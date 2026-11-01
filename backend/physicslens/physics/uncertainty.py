"""Parameter uncertainty by residual block bootstrap around the accepted fit."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .coords import Coordinates
from .fit import FitResult, FitTimeout, fit_window
from .models import ModelSpec


@dataclass(frozen=True)
class ParamEstimate:
    value: float
    std: float
    ci_low: float
    ci_high: float


MAX_BOOTSTRAP_SECONDS = 60.0  # a slow law (an agent-written one with sign()) took minutes for 40 refits
MIN_BOOTSTRAP_REPS = 8


def bootstrap_samples(spec: ModelSpec, c: Coordinates, fit: FitResult, reps: int = 24,
                      seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Refit on (fitted trajectory + resampled residual blocks): (param samples [reps, k], length samples [reps]).

    Blocks, not single samples, because tracking errors are correlated frame to frame.
    For a pendulum the length is redrawn from its own spread each time, since g is proportional to L.
    """
    rng = np.random.default_rng(seed)
    resid = c.q - fit.sim
    n = len(resid)
    block = max(3, n // 20)
    theta_hat = np.concatenate([[fit.params[p] for p in spec.param_names], fit.y0])
    lower = np.concatenate([[p.lower for p in spec.params], np.full(len(fit.y0), -np.inf)])
    upper = np.concatenate([[p.upper for p in spec.params], np.full(len(fit.y0), np.inf)])

    samples, lengths = [], []
    t_end = time.monotonic() + MAX_BOOTSTRAP_SECONDS
    for _ in range(reps):
        # Past the time budget, the refits so far give the spread (never fewer than MIN_BOOTSTRAP_REPS).
        if time.monotonic() > t_end and len(samples) >= MIN_BOOTSTRAP_REPS:
            break
        starts = rng.integers(0, n - block + 1, size=n // block + 1)
        boot_resid = np.concatenate([resid[s:s + block] for s in starts])[:n]
        ctx = dict(c.ctx)
        if "length_std" in ctx:
            ctx["length_m"] = rng.normal(ctx["length_m"], ctx["length_std"])
        boot = Coordinates(c.family, c.t, fit.sim + boot_resid, c.names, ctx)
        try:
            theta = fit_window(spec, boot, n, theta_hat, lower, upper, max_nfev=40,
                               deadline=time.monotonic() + MAX_BOOTSTRAP_SECONDS / MIN_BOOTSTRAP_REPS)
        except FitTimeout:
            continue  # one refit stuck in a law's corners is skipped, not waited for
        samples.append(theta[: len(spec.params)])
        lengths.append(ctx.get("length_m", np.nan))
    return np.array(samples), np.array(lengths)


IDENTIFIABILITY_R = 0.9  # refits this correlated: the clip pins down a combination of the two better than each one

# A filming instruction that separates each pair, from the physics of how the two terms differ.
ADVICE: dict[frozenset[str], str] = {
    frozenset({"zeta", "f"}): ("Film a larger first swing and let it decay to rest: viscous damping shrinks the swings "
                               "by a fixed ratio, dry friction by a fixed amount."),
    frozenset({"c", "k"}): ("Film the spin from fast all the way to a stop: dry friction slows it by the same amount "
                            "each second, viscous friction in proportion to its speed."),
    frozenset({"omega", "beta"}): "Film swings of several sizes: a stiffening spring's period changes with amplitude.",
    frozenset({"beta", "zeta"}): "Film swings of several sizes: a stiffening spring's period changes with amplitude.",
    frozenset({"g", "k"}): "Film a longer, faster throw: drag grows with speed, gravity does not.",
    frozenset({"g", "c"}): "Film a longer, faster throw: drag grows with speed, gravity does not.",
    frozenset({"g", "gamma"}): "Film more swings, so the decay shows apart from the period.",
    frozenset({"g", "beta"}): "Film a larger swing: air drag grows with speed, the period's dependence on g does not.",
}
GENERIC_ADVICE = "Film a longer clip with a larger motion, so each term acts on its own for part of it."


def identifiability(names: list[str], samples: np.ndarray) -> list[dict[str, object]]:
    """Parameter pairs whose bootstrap refits move together (|r| > IDENTIFIABILITY_R), with the recording that would
    separate them. Each value's interval is then honest only as part of the pair."""
    if len(samples) < MIN_BOOTSTRAP_REPS or samples.ndim != 2 or samples.shape[1] < 2:
        return []
    sd = samples.std(axis=0)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if sd[i] == 0 or sd[j] == 0:
                continue
            r = float(np.corrcoef(samples[:, i], samples[:, j])[0, 1])
            if abs(r) > IDENTIFIABILITY_R:
                advice = ADVICE.get(frozenset({names[i], names[j]}), GENERIC_ADVICE)
                out.append({"params": [names[i], names[j]], "corr": round(r, 3), "advice": advice})
    return out


def bootstrap_builtin(model_id: str, c: Coordinates, fit: FitResult, reps: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """bootstrap_samples for a library law looked up by id: a top-level function a worker process can run."""
    from .models import BUILTIN_MODELS
    return bootstrap_samples(BUILTIN_MODELS[model_id], c, fit, reps, seed)


def bootstrap(spec: ModelSpec, c: Coordinates, fit: FitResult, reps: int = 40, seed: int = 0) -> dict[str, ParamEstimate]:
    return bootstrap_full(spec, c, fit, reps, seed).estimates


@dataclass(frozen=True)
class Bootstrap:
    estimates: dict[str, ParamEstimate]
    identifiability: list[dict[str, object]]
    reps: int  # refits that finished: 40 is the target, MIN_BOOTSTRAP_REPS the floor under the time budget


def bootstrap_full(spec: ModelSpec, c: Coordinates, fit: FitResult, reps: int = 40,
                   seed: int = 0) -> Bootstrap:
    """95% interval as value +- 1.96 standard deviations of the refits.

    Percentiles of a few dozen refits would just be their second-smallest and second-largest values; the spread
    is a far steadier estimate at this size, and hundreds of refits would make every analysis minutes slower.
    """
    arr, _ = bootstrap_samples(spec, c, fit, reps, seed)
    out = {}
    for i, name in enumerate(spec.param_names):
        # Too few refits finished in time: the uncertainty is unknown (shown without a +-), never a made-up number.
        v, sd = fit.params[name], float(arr[:, i].std(ddof=1)) if len(arr) >= 2 else float("nan")
        out[name] = ParamEstimate(v, sd, v - 1.96 * sd, v + 1.96 * sd)
    return Bootstrap(out, identifiability(spec.param_names, arr), len(arr))


def agreed_parameters(laws: list[tuple[ModelSpec, FitResult]], c: Coordinates, reps: int = 40) -> list[dict]:
    """When laws tie, the parameters they share and agree on: each law's bootstrap interval for the parameter must
    overlap every other's. Such a parameter is measured whichever law turns out right (a pendulum's g, when the laws
    differ only in damping). The interval reported spans all of theirs."""
    if len(laws) < 2:
        return []
    per = [(spec, bootstrap_full(spec, c, fit, reps=reps).estimates) for spec, fit in laws]
    first = per[0][0]
    out = []
    for p in first.params:
        ests = [e.get(p.name) for _, e in per]
        if any(x is None for x in ests):
            continue
        if max(x.ci_low for x in ests) > min(x.ci_high for x in ests):
            continue  # some law's interval misses another's: they disagree on it
        out.append({"name": p.name, "latex": p.latex, "unit": p.unit, "value": float(np.mean([x.value for x in ests])),
                    "std": float(max(x.std for x in ests)), "ci95": [float(min(x.ci_low for x in ests)), float(max(x.ci_high for x in ests))],
                    "per_law": [{"title": spec.title, "value": e[p.name].value, "std": e[p.name].std} for spec, e in per]})
    return out
