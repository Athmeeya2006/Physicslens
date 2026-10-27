"""Double pendulum: two tracked bobs, short-horizon (prediction-error) fitting, and chaos measurement.

A chaotic system can't be fitted by simulating the whole clip: tiny errors grow exponentially and any
long simulation diverges from the data. Instead every short window is started from the measured state and
predicted a fraction of a second ahead; the parameters are fitted to all those short predictions at once.
Chaos itself is measured on the fitted model as the largest Lyapunov exponent (how fast two nearby futures
separate), which turns into a predictability horizon given the tracking noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from ..sigproc.cleanup import differentiate
from ..sigproc.trajectory import Trajectory
from .coords import _radius_std, fit_circle

HORIZON_S = 0.3  # short enough that chaos hasn't amplified the measurement noise yet
STEP_S = 0.15  # a new window this often
TRAIN_FRACTION = 0.75
# Acceptance: held-out windows predicted about as well as training ones, and far better than a naive
# constant-velocity forecast from the same measured states (proof of real predictive skill).
MAX_HELDOUT_VS_TRAIN = 2.0  # chaotic stretches differ a lot in difficulty; the naive test below guards against junk
MAX_VS_NAIVE = 0.5
MAX_LENGTH_REL_STD = 0.03  # same rule as the single pendulum: a flat arc can't measure the inner rod
TOLERANCE_RAD = 0.2  # a ghost this far off in angle is visibly wrong: defines the predictability horizon
CHAOS_LAMBDA = 0.5  # 1/s; below this nearby futures barely separate within a clip
PARAMS = ("g", "mu", "gamma")  # gravity, mass ratio m2/m1, joint damping
LOWER = np.array([1.0, 0.05, 0.0])
UPPER = np.array([40.0, 20.0, 5.0])

LATEX = (r"\ddot\theta_1 = \frac{-g(2+\mu)\sin\theta_1 - \mu g\sin(\theta_1-2\theta_2) - 2\mu\sin\Delta\,"
         r"(\dot\theta_2^2 L_2 + \dot\theta_1^2 L_1\cos\Delta)}{L_1(2+\mu-\mu\cos 2\Delta)} - \gamma\dot\theta_1,\quad "
         r"\ddot\theta_2 = \frac{2\sin\Delta\,((1+\mu)(\dot\theta_1^2 L_1 + g\cos\theta_1) + \mu\dot\theta_2^2 L_2\cos\Delta)}"
         r"{L_2(2+\mu-\mu\cos 2\Delta)} - \gamma\dot\theta_2")


class DoublePendulumError(ValueError):
    pass


def rhs(state: np.ndarray, p: dict[str, float], L1: float, L2: float) -> np.ndarray:
    """Point masses m1 = 1, m2 = mu on massless rods; angles from the downward vertical."""
    t1, t2, w1, w2 = state
    g, mu, gamma = p["g"], p["mu"], p["gamma"]
    d = t1 - t2
    den = 2 + mu - mu * np.cos(2 * d)
    a1 = (-g * (2 + mu) * np.sin(t1) - mu * g * np.sin(t1 - 2 * t2)
          - 2 * np.sin(d) * mu * (w2 * w2 * L2 + w1 * w1 * L1 * np.cos(d))) / (L1 * den)
    a2 = (2 * np.sin(d) * ((1 + mu) * (w1 * w1 * L1 + g * np.cos(t1)) + mu * w2 * w2 * L2 * np.cos(d))) / (L2 * den)
    return np.array([w1, w2, a1 - gamma * w1, a2 - gamma * w2])


def simulate(p: dict[str, float], L1: float, L2: float, y0: np.ndarray, t: np.ndarray) -> np.ndarray | None:
    with np.errstate(all="ignore"):
        sol = solve_ivp(lambda _t, s: rhs(s, p, L1, L2), (t[0], t[-1]), y0, t_eval=t, method="DOP853", rtol=1e-9, atol=1e-11)
    if not sol.success or sol.y.shape[1] != len(t) or not np.all(np.isfinite(sol.y)):
        return None
    return sol.y.T


@dataclass
class DoubleCoords:
    t: np.ndarray
    q: np.ndarray  # (N, 2) unwrapped angles
    pivot: tuple[float, float]
    L1: float
    L2: float
    fps: float
    noise_rad: float
    L1_std: float = 0.0

    @property
    def length_reliable(self) -> bool:
        return self.L1_std <= MAX_LENGTH_REL_STD * self.L1

    def to_xy(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        px, py = self.pivot
        x1, y1 = px + self.L1 * np.sin(q[:, 0]), py - self.L1 * np.cos(q[:, 0])
        return np.column_stack([x1, y1]), np.column_stack([x1 + self.L2 * np.sin(q[:, 1]), y1 - self.L2 * np.cos(q[:, 1])])


def to_double_coords(inner: Trajectory, outer: Trajectory) -> DoubleCoords:
    """Angles of both rods. The inner bob circles the pivot; the outer bob keeps a fixed distance from the inner."""
    if len(inner) != len(outer) or not np.array_equal(inner.frames, outer.frames):
        raise DoublePendulumError("Both bobs must be tracked on the same frames.")
    cx, cy, L1 = fit_circle(inner.xy)
    if cy < inner.xy[:, 1].mean():
        raise DoublePendulumError("The inner bob's path doesn't curve around a pivot above it.")
    rel = outer.xy - inner.xy
    L2 = float(np.median(np.hypot(rel[:, 0], rel[:, 1])))
    t1 = np.unwrap(np.arctan2(inner.xy[:, 0] - cx, -(inner.xy[:, 1] - cy)))
    t2 = np.unwrap(np.arctan2(rel[:, 0], -rel[:, 1]))
    q = np.column_stack([t1, t2])
    kin = differentiate(q, inner.fps)
    noise = float(np.sqrt(np.mean((q - kin.pos) ** 2)))
    return DoubleCoords(inner.t, q, (cx, cy), float(L1), L2, inner.fps, noise, _radius_std(inner.xy))


@dataclass
class DoubleFit:
    params: dict[str, float]
    train_rmse_rad: float
    heldout_rmse_rad: float
    naive_rmse_rad: float
    noise_rad: float
    accepted: bool
    lyapunov_per_s: float
    chaotic: bool
    predictability_s: float | None  # how long a forecast stays within TOLERANCE_RAD given the tracking noise
    y0: list[float] = field(default_factory=list)
    reason: str = ""


def _windows(n: int, fps: float, start: int, stop: int) -> list[int]:
    h, step = int(HORIZON_S * fps), max(1, int(STEP_S * fps))
    return [i for i in range(start, stop - h, step)]


def _prediction_residuals(theta: np.ndarray, c: DoubleCoords, states: np.ndarray, starts: list[int]) -> np.ndarray:
    p = dict(zip(PARAMS, theta, strict=True))
    h = int(HORIZON_S * c.fps)
    out = []
    for i in starts:
        sim = simulate(p, c.L1, c.L2, states[i], c.t[i:i + h + 1])
        out.append(np.full(2 * h, 1.0) if sim is None else (sim[1:, :2] - c.q[i + 1:i + h + 1]).ravel())
    return np.concatenate(out)


def _naive_rmse(c: DoubleCoords, states: np.ndarray, starts: list[int]) -> float:
    h = int(HORIZON_S * c.fps)
    errs = []
    for i in starts:
        dt = (c.t[i + 1:i + h + 1] - c.t[i])[:, None]
        errs.append((states[i, :2] + states[i, 2:] * dt - c.q[i + 1:i + h + 1]).ravel())
    return float(np.sqrt(np.mean(np.concatenate(errs) ** 2))) if errs else float("nan")


def lyapunov(p: dict[str, float], L1: float, L2: float, y0: np.ndarray, seconds: float, tau: float = 0.1) -> float:
    """Largest Lyapunov exponent by Benettin's method: follow a nearby copy, renormalize every tau."""
    d0 = 1e-8
    a = np.array(y0, dtype=float)
    b = a + np.array([d0, 0, 0, 0])
    total = 0.0
    steps = max(1, int(seconds / tau))
    for _ in range(steps):
        sa = simulate(p, L1, L2, a, np.array([0.0, tau]))
        sb = simulate(p, L1, L2, b, np.array([0.0, tau]))
        if sa is None or sb is None:
            break
        a, b = sa[-1], sb[-1]
        dist = float(np.linalg.norm(b - a)) or d0
        total += np.log(dist / d0)
        b = a + (b - a) * (d0 / dist)
    return total / (steps * tau)


def fit_double(c: DoubleCoords) -> DoubleFit:
    n = len(c.t)
    if n < 3 * int(HORIZON_S * c.fps):
        raise DoublePendulumError("The clip is too short to fit a double pendulum.")
    kin = differentiate(c.q, c.fps)
    states = np.column_stack([kin.pos, kin.vel])
    n_train = int(n * TRAIN_FRACTION)
    train = _windows(n, c.fps, 0, n_train)
    held = _windows(n, c.fps, n_train, n)
    theta0 = np.array([9.8, 1.0, 0.05])
    sol = least_squares(_prediction_residuals, theta0, bounds=(LOWER, UPPER), args=(c, states, train), x_scale="jac")
    params = dict(zip(PARAMS, (float(v) for v in sol.x), strict=True))
    train_rmse = float(np.sqrt(np.mean(sol.fun ** 2)))
    held_res = _prediction_residuals(sol.x, c, states, held) if held else np.array([np.nan])
    held_rmse = float(np.sqrt(np.mean(held_res ** 2)))
    naive = _naive_rmse(c, states, held)
    accepted = bool(held and c.length_reliable and held_rmse <= MAX_HELDOUT_VS_TRAIN * train_rmse
                    and held_rmse <= MAX_VS_NAIVE * naive)

    lam = lyapunov(params, c.L1, c.L2, states[0], seconds=float(c.t[-1] - c.t[0]))
    chaotic = lam > CHAOS_LAMBDA
    horizon = float(np.log(TOLERANCE_RAD / max(c.noise_rad, 1e-6)) / lam) if chaotic else None
    if not c.length_reliable:
        reason = "The inner bob's arc is too flat to measure the first rod's length, so g can't be pinned down."
    else:
        reason = (f"Held-out {HORIZON_S:.1f} s predictions are off by {held_rmse:.3f} rad (training {train_rmse:.3f}, "
                  f"naive constant-velocity guess {naive:.3f}).")
    return DoubleFit(params, train_rmse, held_rmse, naive, c.noise_rad, accepted, lam, chaotic, horizon,
                     [float(v) for v in states[0]], reason)


def summary(fit: DoubleFit, c: DoubleCoords) -> dict[str, Any]:
    return {"params": fit.params, "L1": c.L1, "L2": c.L2, "accepted": fit.accepted, "reason": fit.reason,
            "lyapunov_per_s": fit.lyapunov_per_s, "chaotic": fit.chaotic, "predictability_s": fit.predictability_s,
            "heldout_rmse_rad": fit.heldout_rmse_rad, "naive_rmse_rad": fit.naive_rmse_rad, "noise_rad": fit.noise_rad,
            "L1_std": c.L1_std, "length_reliable": c.length_reliable, "latex": LATEX}
