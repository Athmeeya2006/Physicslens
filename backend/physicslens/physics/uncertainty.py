"""Parameter uncertainty by residual block bootstrap around the accepted fit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coords import Coordinates
from .fit import FitResult, fit_window
from .models import ModelSpec


@dataclass(frozen=True)
class ParamEstimate:
    value: float
    std: float
    ci_low: float
    ci_high: float


def bootstrap(spec: ModelSpec, c: Coordinates, fit: FitResult, reps: int = 24, seed: int = 0) -> dict[str, ParamEstimate]:
    """Refit on (fitted trajectory + resampled residual blocks).

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

    samples = []
    for _ in range(reps):
        starts = rng.integers(0, n - block + 1, size=n // block + 1)
        boot_resid = np.concatenate([resid[s:s + block] for s in starts])[:n]
        ctx = dict(c.ctx)
        if "length_std" in ctx:
            ctx["length_m"] = rng.normal(ctx["length_m"], ctx["length_std"])
        boot = Coordinates(c.family, c.t, fit.sim + boot_resid, c.names, ctx)
        theta = fit_window(spec, boot, n, theta_hat, lower, upper, max_nfev=40)
        samples.append(theta[: len(spec.params)])

    arr = np.array(samples)
    out = {}
    for i, name in enumerate(spec.param_names):
        lo, hi = np.percentile(arr[:, i], [2.5, 97.5])
        out[name] = ParamEstimate(fit.params[name], float(arr[:, i].std(ddof=1)), float(lo), float(hi))
    return out
