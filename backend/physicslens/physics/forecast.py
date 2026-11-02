"""Predict a future event from the verified law, with an honest timing window, then check it against what happened.

For a swing or a spring the event is the oscillation passing its rest point in a chosen direction: easy to see, and
its time depends on the whole law (frequency, damping, starting state), so a hit many periods ahead is evidence the
law is right. For something rolling or sliding there is no rest point to come back to; the event is it reaching a
point further along its path (where the law puts it `horizon` seconds ahead), whose time depends on the friction or
drag law and the speed. A throw is over before a law can be verified and then tested on what is left, so it gets none.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coords import Coordinates
from .fit import FitResult, simulate
from .models import ModelSpec

# 95% window: 1.96 standard deviations of the predicted time over the bootstrap refits, plus the time resolution
# of the measurement itself (one sample either side, since a crossing is interpolated between two samples).
Z95 = 1.96
FAMILIES = ("pendulum", "oscillator", "line")
OSCILLATING = ("pendulum", "oscillator")
# Below this distance (in the coordinate's units) ahead the object has as good as stopped: nothing left to time.
MIN_AHEAD = 1e-3


@dataclass(frozen=True)
class Prediction:
    t_pred: float  # seconds on the coordinates' clock (coords.t)
    window_s: float  # half-width of the 95% window
    level: float  # rest point the crossing is measured against
    direction: int  # +1: q increasing through the rest point
    made_at: float  # last measured time used for the prediction (same clock)


def crossing_times(t: np.ndarray, q: np.ndarray, level: float, direction: int) -> np.ndarray:
    """Times where q passes `level` going in `direction`, linearly interpolated between samples."""
    d = q - level
    i = np.flatnonzero((d[:-1] * direction < 0) & (d[1:] * direction >= 0))
    frac = d[i] / (d[i] - d[i + 1])
    return t[i] + frac * (t[i + 1] - t[i])


def _rest_level(q: np.ndarray) -> float:
    # The median of a long simulated stretch sits at the equilibrium for any decaying or steady oscillation.
    return float(np.median(q))


def predict_crossing(spec: ModelSpec, fit: FitResult, c: Coordinates, horizon_s: float,
                     param_samples: np.ndarray, direction: int = 1) -> Prediction | None:
    """The first rest-point crossing at least `horizon_s` after the last measured sample.

    param_samples: bootstrap refits [reps, k] in spec.param_names order. Each is simulated from the same starting
    state, and the spread of its crossing time is the timing uncertainty.
    """
    if c.family not in FAMILIES:
        return None
    dt = float(c.t[1] - c.t[0])
    t_end = float(c.t[-1])
    t = np.arange(c.t[0], t_end + horizon_s + 4.0, dt)  # a few seconds past the target so a period always fits
    q = simulate(spec, fit.params, fit.y0, t, c.ctx)
    if q is None:
        return None
    if c.family in OSCILLATING:
        level = _rest_level(q[:, 0])
        after = crossing_times(t, q[:, 0], level, direction)
        after = after[after >= t_end + horizon_s]
        if len(after) == 0:
            return None  # it stops before then; nothing to time
        t_pred = float(after[0])
    else:
        # Where the law puts it `horizon_s` ahead; the event is passing that point, moving the way it moves now.
        k_end = int(np.searchsorted(t, t_end))
        k_tgt = int(np.searchsorted(t, t_end + horizon_s))
        if k_tgt >= len(t):
            return None
        ahead = float(q[k_tgt, 0] - q[k_end, 0])
        if abs(ahead) < MIN_AHEAD * max(1.0, float(np.ptp(c.q[:, 0]))):
            return None  # it will have stopped: a stopped object passes no point
        direction = 1 if ahead > 0 else -1
        level = float(q[k_tgt, 0])
        t_pred = float(t_end + horizon_s)
    times = []
    for row in param_samples:
        p = dict(zip(spec.param_names, map(float, row), strict=True))
        qs = simulate(spec, p, fit.y0, t, c.ctx)
        if qs is None:
            continue
        cs = crossing_times(t, qs[:, 0], _rest_level(qs[:, 0]) if c.family in OSCILLATING else level, direction)
        if len(cs):
            times.append(float(cs[np.argmin(np.abs(cs - t_pred))]))
    if len(times) < max(3, len(param_samples) // 2):
        return None  # most refits no longer cross near then: the prediction has no meaningful window
    window = Z95 * float(np.std(times, ddof=1)) + dt
    return Prediction(t_pred, window, level, direction, t_end)


def observed_crossing(t: np.ndarray, q: np.ndarray, p: Prediction) -> float | None:
    """The measured crossing nearest the predicted time, or None if the data don't reach that far."""
    if t[-1] < p.t_pred + p.window_s or t[0] > p.t_pred - p.window_s:
        return None
    cs = crossing_times(t, q, p.level, p.direction)
    if len(cs) == 0:
        return None
    return float(cs[np.argmin(np.abs(cs - p.t_pred))])


def judge(p: Prediction, observed: float | None) -> dict:
    """Hit when the measured crossing falls inside the window stated before it happened."""
    if observed is None:
        return {"status": "unmeasured"}
    err = observed - p.t_pred
    return {"status": "hit" if abs(err) <= p.window_s else "miss", "error_s": err, "window_s": p.window_s,
            "observed": observed}
