"""Fit each candidate ODE to the measured coordinate by simulate-and-compare least squares."""

from __future__ import annotations

import atexit
import logging
import math
import multiprocessing
import os
import time
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from ..sigproc.cleanup import differentiate, savgol_window
from .coords import Coordinates
from .models import BUILTIN_MODELS, ModelSpec, derivative

log = logging.getLogger(__name__)

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
    # Shown in the decision trail: e.g. a law that could not improve on a special case it contains.
    notes: tuple[str, ...] = ()
    bic_raw: float | None = None  # BIC with every sample counted as independent (bic then uses n_eff; physics/rigor.py)
    rolling: tuple[dict[str, float], ...] = ()  # forecasts from later origins (physics/rigor.py)


# One simulation's own limits. The fit's time budget is checked between simulations, but a stiff law (an agent's
# written law with a huge coefficient) makes the integrator take ever smaller steps inside one: on a real clip a
# single simulation ran for over an hour. A clip of a few seconds needs a few thousand evaluations.
MAX_RHS_CALLS = 200_000
MAX_SIM_SECONDS = 5.0


class _TooSlow(Exception):
    pass


def simulate(spec: ModelSpec, params: dict[str, float], y0: np.ndarray, t: np.ndarray, ctx: dict[str, float]) -> np.ndarray | None:
    """(N, d) positions, or None if the integration blew up or would not finish (treated the same by every fit)."""
    d = len(y0) // 2
    calls = 0
    stop = time.monotonic() + MAX_SIM_SECONDS

    def rhs(_t: float, s: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls > MAX_RHS_CALLS or (calls % 2000 == 0 and time.monotonic() > stop):
            raise _TooSlow
        return derivative(spec, s, params, ctx, _t)

    try:
        with np.errstate(all="ignore"):
            sol = solve_ivp(rhs, (t[0], t[-1]), y0, t_eval=t, method="DOP853", rtol=1e-7, atol=1e-9)
    except _TooSlow:
        return None
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


# Smoothing of friction corners per continuation window (models.smoothing): wide on the first quarter, narrower on
# the half, the true law on the whole training span.
SMOOTHING = {0.25: 20.0, 0.5: 5.0, 1.0: 1.0}
def derivative_match(spec: ModelSpec, c: Coordinates, kin, guess: dict[str, float]) -> dict[str, float]:
    """Parameters that best make the law's acceleration match the smoothed measured acceleration."""
    d = c.q.shape[1]
    idx = np.unique(np.linspace(2, len(c.q) - 3, min(400, len(c.q) - 4)).astype(int))
    states = np.column_stack([kin.pos, kin.vel])[idx]
    acc = kin.acc[idx]
    scale = float(np.std(acc)) + 1e-30
    lower = np.array([p.lower for p in spec.params], dtype=float)
    upper = np.array([p.upper for p in spec.params], dtype=float)
    # dtype=float throughout: one guess arriving as true/false made the whole vector boolean and the fitter crashed.
    p0 = np.clip(np.array([guess[n] for n in spec.param_names], dtype=float), lower, upper)

    def residual(theta: np.ndarray) -> np.ndarray:
        params = dict(zip(spec.param_names, theta, strict=True))
        with np.errstate(all="ignore"):
            pred = np.array([derivative(spec, s, params, c.ctx, c.t[i])[d:] for s, i in zip(states, idx, strict=True)])
        r = ((pred - acc) / scale).ravel()
        return np.where(np.isfinite(r), r, FAILED_RESIDUAL)

    try:
        theta = least_squares(residual, p0, bounds=(lower, upper), x_scale="jac", max_nfev=200).x
    except (ValueError, FloatingPointError, TypeError):  # only a better start: failing it keeps the law's own guesses
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


def fit_model(spec: ModelSpec, c: Coordinates, warm: dict[str, float] | None = None,
              starts: tuple[dict[str, float], ...] = ()) -> FitResult:
    """warm: parameters from a previous fit of the same model (live mode); skips the slow continuation start.
    starts: extra starting points (fit_family passes the fitted special cases); the best successful fit is kept."""
    if spec.family != c.family:
        raise ValueError(f"{spec.id} is a {spec.family} model, coordinates are {c.family}.")
    kin = differentiate(c.q, 1.0 / (c.t[1] - c.t[0]))
    noise = noise_sigma(c.q, kin.pos, 1.0 / (c.t[1] - c.t[0]))
    guess = {**initial_guess(spec, c), **(warm or {})}
    if (spec.id.startswith("custom_") or starts) and not warm:
        # A law the agent wrote comes with only its own rough guesses. Matching the law to the measured accelerations
        # first (no simulation needed) puts the parameters near the right basin; without it, an orbit fit started
        # 3x off locked onto a wrong period and missed GM by half.
        guess = derivative_match(spec, c, kin, guess)
    if starts and not warm:
        # A special case's optimum is already in the right basin: refit it on the whole training span directly (the
        # short early windows would pull it away), from that fit's own initial state.
        fits = [_fit_from(spec, c, kin, noise, guess, None),
                *(_fit_from(spec, c, kin, noise, st, st) for st in starts)]
        done = [f for f in fits if f.success]
        return min(done, key=lambda f: f.rss) if done else fits[0]
    return _fit_from(spec, c, kin, noise, guess, warm)


def nested_starts(spec: ModelSpec, fitted: dict[str, FitResult]) -> tuple[dict[str, float], ...]:
    """Starts for a law from each fitted special case it contains: the special case's parameters and initial state,
    with the switched-off terms at their reduction value."""
    out: list[dict[str, float]] = []
    bounds = {p.name: (p.lower, p.upper) for p in spec.params}
    for sub_id, off in spec.nests:
        sub = fitted.get(sub_id)
        if sub is None or not sub.success:
            continue
        base = {k: v for k, v in sub.params.items() if k in bounds}
        y0 = {f"_y0_{i}": float(v) for i, v in enumerate(sub.y0)}
        out.append({**base, **dict(off), **y0})
    return tuple(out)


def _fit_from(spec: ModelSpec, c: Coordinates, kin, noise: float, guess: dict[str, float],
              warm: dict[str, float] | None) -> FitResult:
    n_samples, d = c.q.shape
    start_state = np.concatenate([kin.pos[0], kin.vel[0]])
    if "_y0_0" in guess:  # a start from a fitted special case brings its fitted initial state
        start_state = np.array([guess[f"_y0_{i}"] for i in range(2 * d)], dtype=float)
    theta = np.concatenate([np.array([guess[p] for p in spec.param_names], dtype=float), start_state])
    theta_start = theta.copy()
    span = np.ptp(c.q, axis=0).max() + 1e-9
    vspan = np.abs(kin.vel).max() + 1e-9
    lower = np.concatenate([[p.lower for p in spec.params], c.q.min(axis=0) - span, -3 * vspan * np.ones(d)])
    upper = np.concatenate([[p.upper for p in spec.params], c.q.max(axis=0) + span, 3 * vspan * np.ones(d)])

    n_train = int(n_samples * TRAIN_FRACTION)
    deadline = time.monotonic() + MAX_FIT_SECONDS
    try:
        for frac in (1.0,) if warm else CONTINUATION:
            window = replace(c, ctx={**c.ctx, "v_eps_scale": SMOOTHING.get(frac, 1.0)})
            theta = fit_window(spec, window, max(8, int(n_train * frac)), theta, lower, upper, deadline=deadline)
        params_train, y0_train = _unpack(spec, theta)
        sim_train = simulate(spec, params_train, y0_train, c.t, c.ctx)
        if "_y0_0" in guess:
            # From a special case's whole-clip optimum the whole-clip fit can only go down from that special case's
            # RSS (least squares never ends above its start), so the law cannot end worse than what it contains.
            theta = theta_start
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


def fit_workers() -> int:
    """Processes for fitting candidate laws side by side. Each is a full Python process with SciPy loaded, so the
    default stays small (a laptop running several analyses ran out of memory with one per thread);
    PHYSICSLENS_FIT_WORKERS overrides, 1 turns it off."""
    env = os.environ.get("PHYSICSLENS_FIT_WORKERS", "").strip()
    if env.isdigit():
        return max(1, int(env))
    if _configured_workers is not None:
        return max(1, _configured_workers)
    return max(1, min((os.cpu_count() or 1) // 2, 4))


_configured_workers: int | None = None


def configure_fit_workers(n: int | None) -> None:
    """From config/limits.yaml (fitting.workers); the environment variable still wins, for one-off runs."""
    global _configured_workers
    _configured_workers = n


_pool: Executor | None = None


def _get_pool(workers: int) -> Executor:
    global _pool
    if _pool is None:
        # spawn, not fork: the server has threads, and a forked child can inherit a lock held by one of them.
        _pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
        atexit.register(_pool.shutdown, cancel_futures=True)
    return _pool


def _fit_builtin(model_id: str, c: Coordinates, starts: tuple[dict[str, float], ...] = ()) -> FitResult:
    """Top level so a worker process can run it; the law is looked up by id because specs hold closures."""
    return fit_model(BUILTIN_MODELS[model_id], c, starts=starts)


def fit_models(specs: list[ModelSpec], c: Coordinates,
               starts: list[tuple[dict[str, float], ...]] | None = None) -> list[FitResult]:
    """fit_model for every candidate, in order. Each fit is independent and deterministic, so built-in laws are fitted
    in worker processes while the agent's own laws (known only to this process) are fitted here meanwhile: the same
    results as one after another, sooner: the wait becomes that of the slowest law (random motion, where every law
    struggles, went from 214 s to 105 s)."""
    global _pool
    remote = [i for i, s in enumerate(specs) if BUILTIN_MODELS.get(s.id) is s]
    workers = fit_workers()
    starts = starts or [()] * len(specs)
    if workers < 2 or len(specs) < 2 or not remote:
        return [fit_model(s, c, starts=st) for s, st in zip(specs, starts, strict=True)]
    results: list[FitResult | None] = [None] * len(specs)
    try:
        futures = {i: _get_pool(workers).submit(_fit_builtin, specs[i].id, c, starts[i]) for i in remote}
    except (BrokenProcessPool, RuntimeError) as e:
        log.warning("Fit workers unavailable (%s); fitting one after another.", e)
        _pool = None
        return [fit_model(s, c, starts=st) for s, st in zip(specs, starts, strict=True)]
    for i, s in enumerate(specs):
        if i not in futures:
            results[i] = fit_model(s, c, starts=starts[i])
    for i, f in futures.items():
        try:
            results[i] = f.result()
        except BrokenProcessPool as e:
            # A worker died (out of memory, killed): this fit is redone here, and the next call starts a new pool.
            log.warning("A fit worker stopped (%s); fitting %s here instead.", e, specs[i].id)
            _pool = None
            results[i] = fit_model(specs[i], c, starts=starts[i])
    return [r for r in results if r is not None]


NESTED_REL_TOL = 1e-9


def fit_family(specs: list[ModelSpec], c: Coordinates, known: list[FitResult] | None = None,
               warm: dict[str, dict[str, float]] | None = None) -> list[FitResult]:
    """Fit candidate laws so that none ends worse than a special case it contains (ModelSpec.nests).

    Every law is first fitted from its own guess, side by side in worker processes. A law that ends with more
    residual than one of its fitted special cases is then refitted from that special case's solution (its extra terms
    at zero, its fitted initial state); from there least squares can only go down, so the law matches or beats it.
    Doing this only when needed keeps the cost where it was: starting every law from every special case made the
    oscillator tests four times slower. known: fits already made (the referee's laws reuse the first round's).
    warm: live mode's previous parameters per law, refitted in place every second, still checked by the guard.
    Returns the fits in the order of specs."""
    fitted: dict[str, FitResult] = {f.model_id: f for f in known or []}
    if warm is not None:
        for s in specs:
            fitted[s.id] = fit_model(s, c, warm=warm.get(s.id))
        return _nesting_guard(specs, fitted)
    for f in fit_models(specs, c):
        fitted[f.model_id] = f
    losing = [s for s in specs if _loses_to_special_case(s, fitted)]
    if losing:
        for f in fit_models(losing, c, [nested_starts(s, fitted) for s in losing]):
            if f.success and (not fitted[f.model_id].success or f.rss < fitted[f.model_id].rss):
                fitted[f.model_id] = f
    return _nesting_guard(specs, fitted)


def _loses_to_special_case(s: ModelSpec, fitted: dict[str, FitResult]) -> bool:
    f = fitted.get(s.id)
    if f is None or not f.success:
        return False
    return any(sub is not None and sub.success and f.rss > sub.rss * (1 + NESTED_REL_TOL)
               for sub in (fitted.get(sub_id) for sub_id, _ in s.nests))


def _nesting_guard(specs: list[ModelSpec], fitted: dict[str, FitResult]) -> list[FitResult]:
    out = []
    for s in specs:
        f = fitted[s.id]
        for sub_id, off in s.nests:
            sub = fitted.get(sub_id)
            if f.success and sub is not None and sub.success and f.rss > sub.rss * (1 + NESTED_REL_TOL):
                # Started from the special case and still worse. The special case's solution IS this law with its
                # extra terms at zero, so the law is reported there (same paths, same forecast, more parameters
                # paid for in BIC), and the trail says so rather than hiding it.
                sub_title = BUILTIN_MODELS[sub_id].title if sub_id in BUILTIN_MODELS else sub_id
                note = (f"{s.title} could not improve on {sub_title} (a special case it contains) from either start, "
                        "so it is reported at that special case, its extra terms at zero")
                k = len(s.params) + len(sub.y0)
                f = replace(sub, model_id=s.id, params={**sub.params, **dict(off)}, k=k,
                            bic=float(sub.n * math.log(max(sub.rss, 1e-300) / sub.n) + k * math.log(sub.n)),
                            notes=(*f.notes, note))
        out.append(f)
    return out
