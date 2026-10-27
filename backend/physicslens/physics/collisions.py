"""Bouncing: free flight between impacts plus a restitution law at each impact (a hybrid model).

Between impacts x = x0 + vx t and y = y0 + vy t - g t^2 / 2, with one g shared by every flight. That is
linear in (g, x0_k, vx_k, y0_k, vy_k), so it is solved exactly by least squares. The impact time between
flights k and k+1 is where their parabolas cross; with a shared g the t^2 terms cancel, so it is linear too.
At each impact the vertical speed reverses and shrinks by e (the coefficient of restitution).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from ..sigproc.trajectory import Trajectory

MIN_FLIGHT_SAMPLES = 6
EDGE_TRIM = 1  # samples dropped each side of an impact: the ball is deforming or blurred there
BOOTSTRAP_REPS = 200
MAX_FORECAST_NRMSE = 0.15
MAX_BOUNCES = 200  # hard stop for the hybrid simulation, whatever the inputs
E_RANGE = (0.0, 1.05)
MAX_FLIGHT_G_DEVIATION = 0.35  # a short flight's own curvature is noisy, but not this far off  # a real bounce loses energy; a little over 1 tolerates measurement noise


class BounceError(ValueError):
    pass


@dataclass
class BounceFit:
    g: float
    g_std: float
    floor_y: float
    impact_times: list[float]
    restitution: list[float]  # vertical, per impact
    restitution_std: list[float]
    tangential: list[float]  # horizontal speed kept per impact (friction/spin)
    e_mean: float
    e_std: float
    rms_m: float
    settled_at: float | None  # time after which bounces were too small to resolve (or the ball rested)
    forecast_nrmse: float  # later bounces predicted from e and g measured on the earlier ones
    predicted_xy: np.ndarray = field(repr=False)
    flights: list[tuple[int, int]] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.forecast_nrmse <= MAX_FORECAST_NRMSE


def _noise_sigma(y: np.ndarray) -> float:
    """Position noise from second differences (smooth motion barely contributes at video rates)."""
    d2 = np.diff(y, 2)
    return float(np.median(np.abs(d2 - np.median(d2))) * 1.4826 / np.sqrt(6))


def detect_bounces(traj: Trajectory, floor_band: float = 0.3) -> np.ndarray:
    """Floor impacts: V-shaped minima of height near the lowest point, falling fast before and rising fast after.

    More sensitive than generic jump detection: small late bounces matter, and a missed one would merge two
    flights into a single parabola and spoil the shared fit.
    """
    y, dt = traj.xy[:, 1], traj.dt
    if len(y) < 7:
        return np.array([], dtype=int)
    sigma = _noise_sigma(y)
    vthr = max(4 * sigma * np.sqrt(2) / (2 * dt), 0.15)
    low = y.min() + floor_band * (np.ptp(y) or 1.0)
    hits: list[int] = []
    for i in range(2, len(y) - 2):
        before = (y[i] - y[i - 2]) / (2 * dt)
        after = (y[i + 2] - y[i]) / (2 * dt)
        if y[i] <= low and before < -vthr and after > vthr and y[i] <= y[i - 2:i + 3].min():
            if hits and i - hits[-1] < 3:
                if y[i] < y[hits[-1]]:
                    hits[-1] = i
                continue
            hits.append(i)
    return np.array(hits, dtype=int)


def _flights(n: int, impacts: np.ndarray) -> list[tuple[int, int]]:
    bounds = [0, *[int(i) for i in impacts], n]
    out = []
    for k, (a, b) in enumerate(pairwise(bounds)):
        start = a + (EDGE_TRIM if k > 0 else 0)
        stop = b - (EDGE_TRIM if k < len(bounds) - 2 else 0)
        if stop - start >= MIN_FLIGHT_SAMPLES:
            out.append((start, stop))
    return out


def _solve(t: np.ndarray, xy: np.ndarray, flights: list[tuple[int, int]]) -> np.ndarray:
    """Unknowns [g, (x0, vx, y0, vy) per flight] in absolute time."""
    m = len(flights)
    rows, rhs = [], []
    for k, (a, b) in enumerate(flights):
        for i in range(a, b):
            ti = t[i]
            rx = np.zeros(1 + 4 * m)
            rx[1 + 4 * k], rx[2 + 4 * k] = 1.0, ti
            ry = np.zeros(1 + 4 * m)
            ry[0], ry[3 + 4 * k], ry[4 + 4 * k] = -0.5 * ti * ti, 1.0, ti
            rows += [rx, ry]
            rhs += [xy[i, 0], xy[i, 1]]
    sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return sol


def _impacts(sol: np.ndarray, m: int) -> tuple[list[float], list[float], list[float], list[float]]:
    g = sol[0]
    times, e, tang, floors = [], [], [], []
    for k in range(m - 1):
        _, vx_a, y_a, vy_a = sol[1 + 4 * k: 5 + 4 * k]
        _, vx_b, y_b, vy_b = sol[1 + 4 * (k + 1): 5 + 4 * (k + 1)]
        ts = (y_b - y_a) / (vy_a - vy_b)  # parabolas with the same g cross where their linear parts do
        before, after = vy_a - g * ts, vy_b - g * ts
        times.append(float(ts))
        e.append(float(-after / before) if before != 0 else float("nan"))
        tang.append(float(vx_b / vx_a) if vx_a != 0 else float("nan"))
        floors.append(float(y_a + vy_a * ts - 0.5 * g * ts * ts))
    return times, e, tang, floors


def simulate_bounces(t: np.ndarray, state0: tuple[float, float, float, float], t0: float, g: float, floor_y: float,
                     e: float, tangential: float) -> np.ndarray:
    """Analytic hybrid simulation: parabolic flights, restitution at the floor."""
    if not (g > 0 and E_RANGE[0] < e < E_RANGE[1]):
        raise BounceError(f"Unphysical bounce parameters (g = {g:.3g}, e = {e:.3g}).")
    x0, y0, vx, vy = state0
    bounces = 0
    out = np.zeros((len(t), 2))
    tc = t0
    resting = False
    for i, ti in enumerate(t):
        while not resting:
            dt = ti - tc
            if y0 + vy * dt - 0.5 * g * dt * dt >= floor_y - 1e-9:
                break
            # Time of floor contact within this flight (falling root of the quadratic).
            disc = vy * vy + 2 * g * (y0 - floor_y)
            hit = (vy + np.sqrt(max(disc, 0.0))) / g
            bounces += 1
            if hit <= 1e-9 or bounces > MAX_BOUNCES:
                resting = True
                vy = 0.0
                break
            x0, y0 = x0 + vx * hit, floor_y
            vy = -e * (vy - g * hit)
            vx = tangential * vx
            tc += hit
            if abs(vy) < 0.05:  # hops too small to matter: the ball comes to rest instead of looping forever
                resting = True
                vy = 0.0
        dt = ti - tc
        if resting:
            out[i] = (x0 + vx * dt, floor_y)
            continue
        out[i] = (x0 + vx * dt, max(y0 + vy * dt - 0.5 * g * dt * dt, floor_y))
    return out


def fit_bounces(traj: Trajectory, impacts: np.ndarray | None = None, seed: int = 0) -> BounceFit:
    t, xy = traj.t, traj.xy
    impacts = detect_bounces(traj) if impacts is None else impacts
    # A ball resting on the floor is not a flight: dropping it keeps a flat stretch out of the shared parabola fit.
    sigma = _noise_sigma(xy[:, 1])
    flights: list[tuple[int, int]] = []
    g_ref = None
    for a, b in _flights(len(traj), impacts):
        # Each flight must be one free-fall parabola. Unresolvable tiny hops or a ball settling on the floor
        # fail this; impacts are only measured between consecutive clean flights, so analysis stops at the
        # first bad one. The curvature test is physical: a flight's own g must agree with the first flight's.
        tt, yy = t[a:b], xy[a:b, 1]
        coef = np.polyfit(tt, yy, 2)
        g_k = -2 * coef[0]
        rms = np.sqrt(np.mean((yy - np.polyval(coef, tt)) ** 2))
        if np.ptp(yy) <= 6 * sigma + 0.005 or rms > max(4 * sigma, 0.003) or g_k <= 0:
            break
        if g_ref is not None and abs(g_k - g_ref) > MAX_FLIGHT_G_DEVIATION * g_ref:
            break
        g_ref = g_ref if g_ref is not None else g_k
        flights.append((a, b))
    if len(flights) < 2:
        raise BounceError("Need at least two clean flights between impacts to measure a bounce.")
    sol = _solve(t, xy, flights)
    m = len(flights)
    times, e, tang, floors = _impacts(sol, m)
    if sol[0] <= 0 or not all(E_RANGE[0] < x < E_RANGE[1] for x in e):
        raise BounceError(f"The detected impacts don't behave like bounces (g = {sol[0]:.3g}, e = {np.round(e, 2).tolist()}).")

    idx = np.concatenate([np.arange(a, b) for a, b in flights])
    fitted = np.zeros_like(xy)
    for k, (a, b) in enumerate(flights):
        x0, vx, y0, vy = sol[1 + 4 * k: 5 + 4 * k]
        tt = t[a:b]
        fitted[a:b] = np.column_stack([x0 + vx * tt, y0 + vy * tt - 0.5 * sol[0] * tt * tt])
    resid = xy[idx] - fitted[idx]

    # Residual bootstrap: refit on (fit + resampled residuals) for g and e uncertainties.
    rng = np.random.default_rng(seed)
    gs, es = [], []
    for _ in range(BOOTSTRAP_REPS):
        boot = xy.copy()
        boot[idx] = fitted[idx] + resid[rng.integers(0, len(idx), len(idx))]
        s = _solve(t, boot, flights)
        gs.append(s[0])
        es.append(_impacts(s, m)[1])
    es_arr = np.array(es)

    # Forecast: measure e on the first half of the impacts, predict everything from the first flight.
    n_train = max(1, (m - 1) // 2)
    e_train, tang_train = float(np.mean(e[:n_train])), float(np.mean(tang[:n_train]))
    x0, vx, y0, vy = sol[1:5]
    t_first = t[flights[0][0]]
    state0 = (x0 + vx * t_first, y0 + vy * t_first - 0.5 * sol[0] * t_first ** 2, vx, vy - sol[0] * t_first)
    floor_y = float(np.mean(floors))
    predicted = simulate_bounces(t, state0, t_first, float(sol[0]), floor_y, e_train, tang_train)
    later = idx[idx >= flights[min(n_train + 1, m - 1)][0]]
    spread = float(np.std(xy[:, 1])) or 1.0
    forecast = float(np.sqrt(np.mean((predicted[later] - xy[later]) ** 2)) / spread) if len(later) else float("nan")

    full = simulate_bounces(t, state0, t_first, float(sol[0]), floor_y, float(np.mean(e)), float(np.mean(tang)))
    return BounceFit(
        g=float(sol[0]), g_std=float(np.std(gs, ddof=1)), floor_y=floor_y, impact_times=times,
        restitution=e, restitution_std=[float(v) for v in es_arr.std(axis=0, ddof=1)], tangential=tang,
        e_mean=float(np.mean(e)), e_std=float(np.std(e, ddof=1)) if len(e) > 1 else float(es_arr[:, 0].std(ddof=1)),
        rms_m=float(np.sqrt(np.mean(resid ** 2))),
        settled_at=float(t[flights[-1][1]]) if flights[-1][1] < len(t) - 1 else None,
        forecast_nrmse=forecast, predicted_xy=full, flights=flights,
    )
