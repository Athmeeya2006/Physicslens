"""Fast honest failure: recognise noise-like motion before spending a minute of simulation fits on it.

Every check here is cheap (no simulation): the laws' accelerations matched to the measured ones, the spectrum of the
velocity, and how much of the track a smooth trend explains. All three must point to noise; anything with a clear
law, a steady drift or a free fall goes on to the full fits.
"""

from __future__ import annotations

import numpy as np

from ..sigproc.cleanup import differentiate
from .coords import Coordinates
from .fit import derivative_match, initial_guess
from .models import ModelSpec, derivative

# Measured on synthetic tracks (tests/test_prescreen.py): random walks score acceleration R^2 = 0.0 to 0.03, velocity
# flatness 0.53 to 0.62 and a quadratic trend explaining 22% to 87% of the track; a pendulum 0.97 / 0.01 / 1%, a
# spring 0.31 / 0.29 / 0%, constant velocity and free fall 0.00 / 0.31 to 0.44 / 100% (no force to match, which is why
# the trend test is there: a law-free straight line is still a law).
MAX_ACCEL_R2 = 0.2
MIN_FLATNESS = 0.5
MAX_TREND_R2 = 0.95


def acceleration_r2(spec: ModelSpec, c: Coordinates, kin) -> float:
    """How much of the measured acceleration the law explains at its best-matching parameters (no simulation)."""
    guess = derivative_match(spec, c, kin, initial_guess(spec, c)) if spec.params else {}
    d = c.q.shape[1]
    idx = np.arange(2, len(c.q) - 2)
    states = np.column_stack([kin.pos, kin.vel])[idx]
    pred = np.array([derivative(spec, s, guess, c.ctx, c.t[i])[d:] for s, i in zip(states, idx, strict=True)],
                    dtype=float)
    acc = kin.acc[idx]
    if not np.all(np.isfinite(pred)):
        return -np.inf
    return float(1 - np.sum((acc - pred) ** 2) / (np.sum((acc - acc.mean(axis=0)) ** 2) + 1e-30))


def spectral_flatness(c: Coordinates) -> float:
    """Geometric over arithmetic mean of the velocity's power spectrum: near 1 for white noise, near 0 for a tone."""
    v = np.diff(c.q, axis=0)
    out = []
    for j in range(v.shape[1]):
        power = np.abs(np.fft.rfft(v[:, j] - v[:, j].mean()))[1:] ** 2 + 1e-30
        out.append(float(np.exp(np.mean(np.log(power))) / np.mean(power)))
    return float(np.mean(out))


def trend_r2(c: Coordinates) -> float:
    """The smallest share of any coordinate's variance a quadratic in time explains (1 for drift or free fall)."""
    t = c.t - c.t.mean()
    out = []
    for j in range(c.q.shape[1]):
        resid = c.q[:, j] - np.polyval(np.polyfit(t, c.q[:, j], 2), t)
        out.append(1 - float(resid.var()) / (float(c.q[:, j].var()) + 1e-30))
    return float(min(out))


def noise_like(c: Coordinates, specs: list[ModelSpec]) -> str | None:
    """The reason to stop early, or None to go on to the full fits."""
    if len(c.q) < 16:
        return None
    kin = differentiate(c.q, 1.0 / (c.t[1] - c.t[0]))
    r2 = max((acceleration_r2(s, c, kin) for s in specs), default=-np.inf)
    if r2 >= MAX_ACCEL_R2:
        return None
    flat, trend = spectral_flatness(c), trend_r2(c)
    if flat <= MIN_FLATNESS or trend >= MAX_TREND_R2:
        return None
    return (f"The motion looks like noise: no candidate law explains more than {max(r2, 0.0):.0%} of the measured "
            f"acceleration, the velocity's spectrum is flat ({flat:.2f}, where 1 is white noise), and no smooth trend "
            f"explains the track ({trend:.0%}). Simulation fits were skipped.")
