"""Fit each candidate ODE to the measured coordinate by simulate-and-compare least squares."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from ..sigproc.cleanup import differentiate, savgol_window
from .coords import Coordinates
from .models import ModelSpec

TRAIN_FRACTION = 0.75
# Fit on growing windows so an oscillation's frequency is locked in before many cycles can alias it.
CONTINUATION = (0.25, 0.5, 1.0)
FAILED_RESIDUAL = 1e3


@dataclass
class FitResult:
    model_id: str
    params: dict[str, float]
    y0: np.ndarray
    n: int
    k: int
    rss: float
    bic: float
    rmse: float
    heldout_rmse: float
    heldout_nrmse: float
    noise_sigma: float  # model-free tracking-noise estimate, same units as q
    success: bool
    message: str = ""
    sim: np.ndarray = field(default_factory=lambda: np.empty((0, 0)), repr=False)
    # Fitted on the first TRAIN_FRACTION only and run over the whole clip: the honest forecast of the rest.
    sim_train: np.ndarray = field(default_factory=lambda: np.empty((0, 0)), repr=False)
    n_train: int = 0


def simulate(spec: ModelSpec, params: dict[str, float], y0: np.ndarray, t: np.ndarray, ctx: dict[str, float]) -> np.ndarray | None:
    """(N, d) positions, or None if the integration blew up."""
    d = len(y0) // 2
    with np.errstate(all="ignore"):
        sol = solve_ivp(lambda _t, s: spec.rhs(s, params, ctx), (t[0], t[-1]), y0, t_eval=t,
                        method="DOP853", rtol=1e-7, atol=1e-9)
    if not sol.success or sol.y.shape[1] != len(t) or not np.all(np.isfinite(sol.y)):
        return None
    return sol.y[:d].T


def noise_sigma(q: np.ndarray, smoothed: np.ndarray, fps: float, window_s: float = 0.12, polyorder: int = 3) -> float:
    """Tracking noise from the Savitzky-Golay residual, corrected for the degrees of freedom the filter absorbs."""
    w = savgol_window(len(q), fps, window_s, polyorder)
    return float(np.sqrt(np.mean((q - smoothed) ** 2) / max(1 - (polyorder + 1) / w, 0.1)))


def dominant_omega(t: np.ndarray, x: np.ndarray) -> float:
    """Angular frequency of the strongest oscillation (zero-padded FFT peak)."""
    x = x - x.mean()
    n = max(4096, 1 << (len(x) - 1).bit_length())
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n))
    freqs = np.fft.rfftfreq(n, t[1] - t[0])
    spec[0] = 0
    return float(2 * np.pi * freqs[np.argmax(spec)])


def initial_guess(spec: ModelSpec, c: Coordinates) -> dict[str, float]:
    q = c.q[:, 0]
    guess: dict[str, float] = {}
    for p in spec.params:
        if p.guess is not None:
            guess[p.name] = p.guess
        elif p.name == "g":
            if c.family == "pendulum":
                guess["g"] = dominant_omega(c.t, q) ** 2 * c.ctx["length_m"]
            else:
                guess["g"] = 9.8
        elif p.name == "omega":
            guess["omega"] = dominant_omega(c.t, q)
        elif p.name == "kappa":
            guess["kappa"] = dominant_omega(c.t, q) ** 2
        elif p.name == "phi0":
            guess["phi0"] = float(q.mean())
        elif p.name == "s0":
            guess["s0"] = float(q.mean())
        else:
            guess[p.name] = 0.05  # damping/drag: start small, the data pulls it up
    return {p.name: float(np.clip(guess[p.name], p.lower, p.upper)) for p in spec.params}


def _unpack(spec: ModelSpec, theta: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    k = len(spec.params)
    return dict(zip(spec.param_names, theta[:k], strict=True)), theta[k:]


class FitTimeout(RuntimeError):
    pass


# A law with a sign() or abs() (dry friction, drag written by the agent) can make the solver crawl through its
# corners: one such fit took minutes on a real clip. Past this budget the model is reported as not fitted.
MAX_FIT_SECONDS = 45.0


def derivative_match(spec: ModelSpec, c: Coordinates, kin, guess: dict[str, float]) -> dict[str, float]:
    """Parameters that best make the law's acceleration match the smoothed measured acceleration."""
    d = c.q.shape[1]
    idx = np.unique(np.linspace(2, len(c.q) - 3, min(400, len(c.q) - 4)).astype(int))
    states = np.column_stack([kin.pos, kin.vel])[idx]
    acc = kin.acc[idx]
    scale = float(np.std(acc)) + 1e-30
    lower = np.array([p.lower for p in spec.params])
    upper = np.array([p.upper for p in spec.params])
    p0 = np.clip([guess[n] for n in spec.param_names], lower, upper)

    def residual(theta: np.ndarray) -> np.ndarray:
        params = dict(zip(spec.param_names, theta, strict=True))
        with np.errstate(all="ignore"):
            pred = np.array([spec.rhs(s, params, c.ctx)[d:] for s in states])
        r = ((pred - acc) / scale).ravel()
        return np.where(np.isfinite(r), r, FAILED_RESIDUAL)

    try:
        theta = least_squares(residual, p0, bounds=(lower, upper), x_scale="jac", max_nfev=200).x
    except (ValueError, FloatingPointError):
        return guess
    return {**guess, **dict(zip(spec.param_names, (float(v) for v in theta), strict=True))}


def fit_window(spec: ModelSpec, c: Coordinates, stop: int, theta0: np.ndarray, lower: np.ndarray, upper: np.ndarray,
               max_nfev: int | None = None, deadline: float | None = None) -> np.ndarray:
    t, q = c.t[:stop], c.q[:stop]

    def residual(theta: np.ndarray) -> np.ndarray:
        if deadline is not None and time.monotonic() > deadline:
            raise FitTimeout
        params, y0 = _unpack(spec, theta)
        sim = simulate(spec, params, y0, t, c.ctx)
        if sim is None:
            return np.full(q.size, FAILED_RESIDUAL)
        return (sim - q).ravel()

    return least_squares(residual, np.clip(theta0, lower, upper), bounds=(lower, upper), x_scale="jac",
                         max_nfev=max_nfev).x


def fit_model(spec: ModelSpec, c: Coordinates, warm: dict[str, float] | None = None) -> FitResult:
    """warm: parameters from a previous fit of the same model (live mode); skips the slow continuation start."""
    if spec.family != c.family:
        raise ValueError(f"{spec.id} is a {spec.family} model, coordinates are {c.family}.")
    n_samples, d = c.q.shape
    kin = differentiate(c.q, 1.0 / (c.t[1] - c.t[0]))
    noise = noise_sigma(c.q, kin.pos, 1.0 / (c.t[1] - c.t[0]))
    guess = {**initial_guess(spec, c), **(warm or {})}
    if spec.id.startswith("custom_") and not warm:
        # A law the agent wrote comes with only its own rough guesses. Matching the law to the measured accelerations
        # first (no simulation needed) puts the parameters near the right basin; without it, an orbit fit started
        # 3x off locked onto a wrong period and missed GM by half.
        guess = derivative_match(spec, c, kin, guess)
    theta = np.concatenate([[guess[p] for p in spec.param_names], kin.pos[0], kin.vel[0]])
    span = np.ptp(c.q, axis=0).max() + 1e-9
    vspan = np.abs(kin.vel).max() + 1e-9
    lower = np.concatenate([[p.lower for p in spec.params], c.q.min(axis=0) - span, -3 * vspan * np.ones(d)])
    upper = np.concatenate([[p.upper for p in spec.params], c.q.max(axis=0) + span, 3 * vspan * np.ones(d)])

    n_train = int(n_samples * TRAIN_FRACTION)
    deadline = time.monotonic() + MAX_FIT_SECONDS
    try:
        for frac in (1.0,) if warm else CONTINUATION:
            theta = fit_window(spec, c, max(8, int(n_train * frac)), theta, lower, upper, deadline=deadline)
        params_train, y0_train = _unpack(spec, theta)
        sim_train = simulate(spec, params_train, y0_train, c.t, c.ctx)
        theta = fit_window(spec, c, n_samples, theta, lower, upper, deadline=deadline)
    except FitTimeout:
        params, y0 = _unpack(spec, theta)
        return FitResult(spec.id, params, y0, n_samples * d, len(theta), math.inf, math.inf, math.inf, math.inf, math.inf,
                         noise, success=False, message=f"the fit did not finish within {MAX_FIT_SECONDS:.0f} s")
    params, y0 = _unpack(spec, theta)
    sim = simulate(spec, params, y0, c.t, c.ctx)
    if sim is None or sim_train is None:
        return FitResult(spec.id, params, y0, n_samples * d, len(theta), math.inf, math.inf, math.inf, math.inf, math.inf,
                         noise, success=False, message="simulation diverged")

    resid = sim - c.q
    n = resid.size
    rss = float(np.sum(resid**2))
    held = sim_train[n_train:] - c.q[n_train:]
    heldout_rmse = float(np.sqrt(np.mean(held**2)))
    return FitResult(
        model_id=spec.id, params={k: float(v) for k, v in params.items()}, y0=y0, n=n, k=len(theta), rss=rss,
        bic=float(n * math.log(max(rss, 1e-300) / n) + len(theta) * math.log(n)),
        rmse=float(np.sqrt(rss / n)), heldout_rmse=heldout_rmse,
        heldout_nrmse=heldout_rmse / float(np.std(c.q) + 1e-12), noise_sigma=noise, success=True, sim=sim,
        sim_train=sim_train, n_train=n_train,
    )
