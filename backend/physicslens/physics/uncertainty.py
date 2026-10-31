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


ENTANGLED_R = 0.98  # refits this correlated: the clip measures a combination of the two, not each one


def entangled(names: list[str], samples: np.ndarray) -> list[dict[str, object]]:
    """Parameter pairs whose bootstrap refits move together almost perfectly. Each value's interval is then honest
    only as part of the pair: a different clip could trade one against the other and fit as well."""
    if len(samples) < MIN_BOOTSTRAP_REPS or samples.ndim != 2 or samples.shape[1] < 2:
        return []
    sd = samples.std(axis=0)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if sd[i] == 0 or sd[j] == 0:
                continue
            r = float(np.corrcoef(samples[:, i], samples[:, j])[0, 1])
            if abs(r) >= ENTANGLED_R:
                out.append({"a": names[i], "b": names[j], "r": round(r, 3)})
    return out


def bootstrap(spec: ModelSpec, c: Coordinates, fit: FitResult, reps: int = 40, seed: int = 0) -> dict[str, ParamEstimate]:
    return bootstrap_full(spec, c, fit, reps, seed)[0]


def bootstrap_full(spec: ModelSpec, c: Coordinates, fit: FitResult, reps: int = 40,
                   seed: int = 0) -> tuple[dict[str, ParamEstimate], list[dict[str, object]]]:
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
    return out, entangled(spec.param_names, arr)
