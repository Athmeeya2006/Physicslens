"""Regime-change detection: the moment a fitted model stops predicting the motion.

From each point of the track, the model is started at the measured state and run one horizon ahead.
Within one regime the error stays near the tracking noise; when the dynamics change abruptly (a shorter
string, an impact, a new setup) it jumps. The model is fitted only on an early stretch, so a change can't
contaminate it. Slow changes such as a gradual rise in damping barely move a 1 s prediction; they are
found by comparing fits of whole recordings instead (machine-health drift).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sigproc.cleanup import differentiate
from ..sigproc.trajectory import Trajectory
from .coords import Coordinates, to_coordinates
from .fit import fit_model, simulate
from .models import MODELS

HORIZON_S = 1.0
STEP_S = 0.2
THRESHOLD = 4.0  # times the in-regime error level
CONSECUTIVE = 3  # checks in a row above threshold before calling a change (rejects single glitches)
MIN_REGIME_S = 2.5


@dataclass(frozen=True)
class RegimeChange:
    index: int  # first sample certainly in the new regime (safe start for refitting)
    pre_end: int  # samples before this are certainly from the old regime
    t: float
    t_estimate: float  # best estimate of when the change happened
    error_ratio: float  # post-change prediction error relative to the in-regime level
    model_id: str


def prediction_errors(spec, params: dict[str, float], c: Coordinates, fps: float) -> list[tuple[int, float]]:
    kin = differentiate(c.q, fps)
    horizon = max(2, int(HORIZON_S * fps))
    step = max(1, int(STEP_S * fps))
    out = []
    for i in range(0, len(c.q) - horizon, step):
        y0 = np.concatenate([kin.pos[i], kin.vel[i]])
        t = c.t[i:i + horizon + 1]
        sim = simulate(spec, params, y0, t, c.ctx)
        err = np.inf if sim is None else float(np.sqrt(np.mean((sim - c.q[i:i + horizon + 1]) ** 2)))
        out.append((i, err))
    return out


def detect_change(traj: Trajectory, family: str, model_id: str, min_regime_s: float = MIN_REGIME_S) -> RegimeChange | None:
    fps = traj.fps
    n0 = int(min_regime_s * fps)
    if len(traj) < 2 * n0:
        return None
    try:
        # Geometry (pivot, axis) from the early stretch only: after a change, e.g. a shorter string,
        # a fit over everything would blend two arcs into a meaningless circle.
        early = to_coordinates(traj.slice(0, n0), family)
        c = to_coordinates(traj, family, geometry=early.ctx)
    except ValueError:
        return None
    spec = MODELS[model_id]
    fit = fit_model(spec, early)
    if not fit.success:
        return None
    errors = prediction_errors(spec, fit.params, c, fps)
    horizon = int(HORIZON_S * fps)
    baseline = [e for i, e in errors if i + horizon <= n0]
    if not baseline:
        return None
    level = max(float(np.median(baseline)), fit.noise_sigma)
    run: list[tuple[int, float]] = []
    for i, e in errors:
        if i + horizon <= n0:
            continue
        if e > THRESHOLD * level:
            run.append((i, e))
            if len(run) >= CONSECUTIVE:
                # The first failing window already overlaps the change; its end is the latest the change can be.
                first_fail = run[0][0]
                start = min(first_fail + horizon, len(traj) - 1)
                middle = min(first_fail + horizon // 2, len(traj) - 1)
                # The window before the first failure passed, so the change can't be much before first_fail.
                pre_end = max(first_fail - max(1, int(STEP_S * fps)), n0)
                return RegimeChange(start, pre_end, float(traj.t[start]), float(traj.t[middle]),
                                    float(np.median([x for _, x in run]) / level), model_id)
        else:
            run = []
    return None
