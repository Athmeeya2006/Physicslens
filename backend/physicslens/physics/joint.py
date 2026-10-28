"""Fit one law to several recordings of the same system at once: the closed experiment loop.

When one clip can't separate two laws (a small swing fits the linear and the air-drag pendulum alike), the agent asks
for a recording that can. Both clips are then fitted together: the physical parameters are shared, each clip keeps
its own starting state, and each clip must forecast its own held-out last quarter. BIC on the joint residuals picks
the law, by the same rules as a single clip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .coords import Coordinates
from .fit import FAILED_RESIDUAL, TRAIN_FRACTION, FitResult, fit_model, simulate
from .models import ModelSpec
from .verify import (
    APPROXIMATES,
    DECISIVE_DELTA_BIC,
    HARD_MAX_HELDOUT_NRMSE,
    MAX_HELDOUT_NOISE_RATIO,
    MAX_HELDOUT_NRMSE,
)


@dataclass
class JointFit:
    model_id: str
    params: dict[str, float]
    y0s: list[np.ndarray]
    n: int
    k: int
    rss: float
    bic: float
    heldout_nrmse: list[float]  # one per clip
    heldout_rmse: list[float]
    noise_sigma: list[float]
    success: bool
    message: str = ""
    per_clip: list[FitResult] = field(default_factory=list, repr=False)

    @property
    def forecasts_well(self) -> bool:
        """Every clip must be forecast, by the single-clip rule: one recording explained badly sinks the law."""
        return self.success and all(
            e <= MAX_HELDOUT_NRMSE or (e <= HARD_MAX_HELDOUT_NRMSE and r <= MAX_HELDOUT_NOISE_RATIO * s)
            for e, r, s in zip(self.heldout_nrmse, self.heldout_rmse, self.noise_sigma, strict=True))


def _residual_fn(spec: ModelSpec, clips: list[Coordinates], stops: list[int]):
    k, d = len(spec.params), clips[0].q.shape[1]

    def residual(theta: np.ndarray) -> np.ndarray:
        params = dict(zip(spec.param_names, theta[:k], strict=True))
        out = []
        for i, (c, stop) in enumerate(zip(clips, stops, strict=True)):
            y0 = theta[k + 2 * d * i: k + 2 * d * (i + 1)]
            # Each clip's residual is scaled by its own spread so a large swing doesn't drown a small one.
            scale = float(np.std(c.q[:stop])) + 1e-12
            sim = simulate(spec, params, y0, c.t[:stop], c.ctx)
            out.append(np.full(c.q[:stop].size, FAILED_RESIDUAL) if sim is None else ((sim - c.q[:stop]) / scale).ravel())
        return np.concatenate(out)

    return residual


def _shared_geometry(clips: list[Coordinates]) -> list[Coordinates]:
    """One pendulum, one length: taken from the widest swing, whose arc measures it best.

    Each clip measures its own length from its arc, and a small swing's arc pins it down poorly. With g shared, two
    different lengths would make the same law look wrong on one of the clips.
    """
    if clips[0].family != "pendulum":
        return clips
    best = max(clips, key=lambda c: float(np.ptp(c.q[:, 0])))
    keys = ("length_m", "length_std")
    return [Coordinates(c.family, c.t, c.q, c.names, {**c.ctx, **{k: best.ctx[k] for k in keys if k in best.ctx}})
            for c in clips]


def fit_joint(spec: ModelSpec, clips: list[Coordinates]) -> JointFit:
    if any(c.family != spec.family for c in clips):
        raise ValueError(f"{spec.id} is a {spec.family} model; every clip must be one too.")
    clips = _shared_geometry(clips)
    d = clips[0].q.shape[1]
    singles = [fit_model(spec, c) for c in clips]
    if not all(f.success for f in singles):
        return JointFit(spec.id, {}, [], 0, 0, math.inf, math.inf, [], [], [], False, "a single-clip fit diverged", singles)
    # Start the shared parameters where the clips agree best: the median of the single-clip fits.
    p0 = [float(np.median([f.params[p] for f in singles])) for p in spec.param_names]
    lower = [p.lower for p in spec.params]
    upper = [p.upper for p in spec.params]
    for f, c in zip(singles, clips, strict=True):
        span = np.ptp(c.q, axis=0).max() + 1e-9
        lower += [*(c.q.min(axis=0) - span), *(-np.inf * np.ones(d))]
        upper += [*(c.q.max(axis=0) + span), *(np.inf * np.ones(d))]
    theta0 = np.concatenate([p0, *[f.y0 for f in singles]])
    lo, hi = np.array(lower), np.array(upper)

    def solve(stops: list[int], start: np.ndarray) -> np.ndarray:
        return least_squares(_residual_fn(spec, clips, stops), np.clip(start, lo, hi), bounds=(lo, hi),
                             x_scale="jac").x

    k = len(spec.params)
    trains = [int(len(c.t) * TRAIN_FRACTION) for c in clips]
    theta_train = solve(trains, theta0)
    theta = solve([len(c.t) for c in clips], theta_train)
    params = dict(zip(spec.param_names, map(float, theta[:k]), strict=True))
    y0s = [theta[k + 2 * d * i: k + 2 * d * (i + 1)] for i in range(len(clips))]
    p_train = dict(zip(spec.param_names, map(float, theta_train[:k]), strict=True))
    rss, n, held_n, held_r = 0.0, 0, [], []
    for i, c in enumerate(clips):
        sim = simulate(spec, params, y0s[i], c.t, c.ctx)
        sim_train = simulate(spec, p_train, theta_train[k + 2 * d * i: k + 2 * d * (i + 1)], c.t, c.ctx)
        if sim is None or sim_train is None:
            return JointFit(spec.id, params, y0s, 0, 0, math.inf, math.inf, [], [], [], False, "simulation diverged", singles)
        scale = float(np.std(c.q)) + 1e-12
        rss += float(np.sum(((sim - c.q) / scale) ** 2))
        n += c.q.size
        held = sim_train[trains[i]:] - c.q[trains[i]:]
        held_r.append(float(np.sqrt(np.mean(held**2))))
        held_n.append(held_r[-1] / scale)
    n_free = k + 2 * d * len(clips)
    bic = n * math.log(max(rss, 1e-300) / n) + n_free * math.log(n)
    return JointFit(spec.id, params, y0s, n, n_free, rss, float(bic), held_n, held_r,
                    [f.noise_sigma for f in singles], True, per_clip=singles)


def decide_joint(fits: list[JointFit]) -> dict:
    """Same rules as one clip: every clip forecast, lowest joint BIC, a decisive margin, approximations folded."""
    ranked = sorted((f for f in fits if f.success), key=lambda f: f.bic)
    passing = [f for f in ranked if f.forecasts_well]
    if not passing:
        return {"status": "no_fit", "winner": None, "ranked": ranked, "contenders": []}
    best = passing[0]
    close = [f for f in passing[1:] if f.bic - best.bic < DECISIVE_DELTA_BIC and f.k <= best.k]
    ids = {f.model_id for f in (best, *close)}
    close = [f for f in close if APPROXIMATES.get(f.model_id, ("",))[0] not in ids]
    if close:
        return {"status": "ambiguous", "winner": None, "ranked": ranked, "contenders": [best, *close]}
    return {"status": "accepted", "winner": best, "ranked": ranked, "contenders": [best]}
