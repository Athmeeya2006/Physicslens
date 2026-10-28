"""Steer a real swing with your hand: the verified law says when to push, the phone says it in rhythm.

A push adds the most energy when the bob is fastest, at the bottom, moving in the push direction; pushing against
the motion at the same moment takes the most energy out. The law predicts when that moment comes next, and the cue
is a rhythm (ticks one period apart, push on the last) because people anticipate a beat far better than they react
to a beep (reaction time is about 0.2 to 0.25 s).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .coords import Coordinates
from .fit import FitResult, simulate
from .forecast import crossing_times
from .models import ModelSpec

Goal = Literal["pump", "stop"]
LEAD_S = 0.6  # never cue a push sooner than this: the rhythm needs time to be heard
TICKS = 3  # ticks before the push, one period apart
# Pushes cued from one fit. Pumping keeps the swing's timing, so a train of pushes one period apart stays in step;
# a push that takes energy out late shifts the timing enough that the next one could land with the motion, so
# stopping re-measures after every push.
TRAIN = {"pump": 3, "stop": 1}
HOLD_BAND = 0.05  # within 5% of the target amplitude: hold (no push) until damping drains it


@dataclass(frozen=True)
class Cue:
    action: Literal["push", "wait", "done"]
    push_t: float | None  # on the coordinates' clock: the first push of the train
    pushes: tuple[float, ...]  # the whole train, one period apart
    period_s: float
    direction: int  # which way to push: +1 toward increasing q (rightward on screen for a pendulum), -1 leftward
    amplitude: float  # current amplitude of q (radians for a pendulum)
    bottom_speed: float  # speed of q at the bottom now: to stop a swing, push about this hard and no harder
    target: float
    reason: str


def amplitude(q: np.ndarray, level: float) -> float:
    return float(np.max(np.abs(q - level)))


def next_cue(spec: ModelSpec, fit: FitResult, c: Coordinates, goal: Goal, target: float, direction: int = 1,
             lead_s: float = 0.0) -> Cue | None:
    """The next moment to push (or not) from the fitted law and the motion seen so far.

    lead_s: how late this person's pushes land after the cue, measured from their earlier pushes (push_delay); the
    cue moves that much earlier so the push itself lands at the bottom.
    """
    dt = float(c.t[1] - c.t[0])
    t_end = float(c.t[-1])
    t = np.arange(c.t[0], t_end + 6.0, dt)
    q = simulate(spec, fit.params, fit.y0, t, c.ctx)
    if q is None:
        return None
    level = float(np.median(q[:, 0]))
    recent = c.q[:, 0][c.t >= t_end - 3.0]  # the last couple of swings set the current amplitude
    amp = amplitude(recent, level)
    near = (t >= t_end) & (t <= t_end + 3.0)
    speed = float(np.max(np.abs(np.gradient(q[near, 0], dt)))) if near.sum() > 2 else 0.0
    ups = crossing_times(t, q[:, 0], level, direction)
    downs = crossing_times(t, q[:, 0], level, -direction)
    allc = np.sort(np.concatenate([ups, downs]))
    period = float(2 * np.median(np.diff(allc))) if len(allc) >= 3 else float("nan")
    if not np.isfinite(period):
        return None
    if goal == "stop":
        if amp < max(target, 0.02):
            return Cue("done", None, (), period, direction, amp, speed, target, "Stopped: the swing is below the target.")
        # Against the motion: at the bottom while it moves the other way, pushing toward `direction`.
        when = downs[downs - lead_s >= t_end + LEAD_S]
        why = "Push against the motion as it passes the bottom, where it is fastest."
    else:
        if amp >= target * (1 - HOLD_BAND):
            return Cue("wait", None, (), period, direction, amp, speed, target, "At the target: no push until damping drains it.")
        when = ups[ups - lead_s >= t_end + LEAD_S]
        why = "Push with the motion as it passes the bottom, where it is fastest."
    if len(when) == 0:
        return None
    first = float(when[0]) - lead_s
    return Cue("push", first, tuple(first + i * period for i in range(TRAIN[goal])), period, direction, amp, speed, target, why)


def push_delay(spec: ModelSpec, fit: FitResult, c: Coordinates, t_obs: np.ndarray, q_obs: np.ndarray,
               cued_t: float) -> float | None:
    """When the push really happened, relative to its cue: where the track leaves the law's prediction.

    t_obs is on the clock of c and runs across the push. A push changes the velocity at one instant, so the gap
    between track and prediction is a slow drift before it and gains a kink at it: a line plus a hinge, with the
    hinge time chosen by least squares, finds it within a frame or so, and slow extrapolation drift can't fake it.
    """
    pred = simulate(spec, fit.params, fit.y0, np.concatenate([c.t[:1], t_obs]), c.ctx)
    if pred is None:
        return None
    keep = (t_obs >= cued_t - 0.6) & (t_obs <= cued_t + 0.8)
    t, r = t_obs[keep], q_obs[keep] - pred[1:, 0][keep]
    if len(t) < 12:
        return None
    best, best_sse = None, np.inf
    for tp in t[3:-3]:
        a = np.column_stack([np.ones_like(t), t - tp, np.maximum(t - tp, 0.0)])
        coef, *_ = np.linalg.lstsq(a, r, rcond=None)
        sse = float(np.sum((a @ coef - r) ** 2))
        if sse < best_sse:
            best, best_sse = float(tp), sse
    return None if best is None else best - cued_t


# A person's lag after a rhythm cue: anything outside this range is a misdetection (or a push skipped), not a lag.
LAG_RANGE_S = (-0.2, 0.6)


def learned_lead(delays: list[float]) -> float:
    """How much earlier to cue, from this person's measured pushes: the median, robust to an odd detection."""
    ok = [d for d in delays if LAG_RANGE_S[0] <= d <= LAG_RANGE_S[1]]
    return float(np.median(ok)) if ok else 0.0
