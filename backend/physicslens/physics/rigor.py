"""Harder checks behind "only an equation that forecasts held-out motion is accepted".

Rolling origins: one 75/25 split can be lucky. The law, with its fitted parameters fixed, is restarted from the data
before several later points of the clip and asked to forecast what follows each one.

Effective sample size: residuals of a smooth model are correlated frame to frame, so N samples carry far less than N
independent pieces of evidence, and BIC computed with N overstates how decisive a difference is.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from .coords import Coordinates
from .fit import FAILED_RESIDUAL, FitResult, simulate
from .models import ModelSpec

ROLLING_ORIGINS = (0.55, 0.65, 0.75, 0.85)
ROLLING_HORIZON = 0.15
MIN_ROLLING_SAMPLES = 4
MIN_N_EFF = 10.0


def rolling_origin(spec: ModelSpec, c: Coordinates, f: FitResult) -> tuple[dict[str, float], ...]:
    """Forecast from each origin: the initial state refitted on the data before it (parameters fixed: cheap), then
    the error over the following ROLLING_HORIZON of the clip, as a share of the motion's spread."""
    n = len(c.t)
    spread = float(np.std(c.q)) + 1e-12
    out = []
    for origin in ROLLING_ORIGINS:
        i0 = int(n * origin)
        i1 = min(n, i0 + max(MIN_ROLLING_SAMPLES, int(n * ROLLING_HORIZON)))
        if i1 - i0 < MIN_ROLLING_SAMPLES or i0 < 8:
            continue

        def resid(y0: np.ndarray, i0: int = i0) -> np.ndarray:
            sim = simulate(spec, f.params, y0, c.t[:i0], c.ctx)
            return np.full(c.q[:i0].size, FAILED_RESIDUAL) if sim is None else (sim - c.q[:i0]).ravel()

        y0 = least_squares(resid, np.asarray(f.y0, float), max_nfev=20).x  # starting near the fit's own state
        sim = simulate(spec, f.params, y0, c.t[:i1], c.ctx)
        if sim is None:
            out.append({"origin": origin, "horizon": (i1 - i0) / n, "nrmse": math.inf, "rmse": math.inf})
            continue
        rmse = float(np.sqrt(np.mean((sim[i0:i1] - c.q[i0:i1]) ** 2)))
        out.append({"origin": origin, "horizon": round((i1 - i0) / n, 3), "nrmse": rmse / spread, "rmse": rmse})
    return tuple(out)


def effective_n(resid: np.ndarray) -> tuple[float, float]:
    """(n_eff, rho): n * (1 - rho) / (1 + rho) from the lag-1 autocorrelation of the residuals, clipped to [10, n]."""
    r = resid.reshape(len(resid), -1)
    n = r.size
    rhos = []
    for j in range(r.shape[1]):
        x = r[:, j] - r[:, j].mean()
        denom = float(np.dot(x, x))
        if denom > 0 and len(x) > 2:
            rhos.append(float(np.dot(x[:-1], x[1:]) / denom))
    rho = float(np.clip(np.mean(rhos), 0.0, 0.999)) if rhos else 0.0
    return float(np.clip(n * (1 - rho) / (1 + rho), min(MIN_N_EFF, n), n)), rho


def bic_with(f: FitResult, n_eff: float) -> float:
    """BIC with the mean squared residual from all samples and the evidence counted as n_eff independent ones."""
    return float(n_eff * math.log(max(f.rss, 1e-300) / f.n) + f.k * math.log(n_eff))
