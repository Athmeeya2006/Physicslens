"""Spin-down law of a free rotor: I dw/dt = -tau_c - b w - c w^2.

A rotor left to coast is slowed by three kinds of drag with different physics: dry (Coulomb) friction in bearings
and seals, constant torque; viscous and eddy-current drag, proportional to speed; air drag (windage), proportional to
speed squared. Which terms a coast-down needs, and how large they are, is a health record: worn bearings raise the
constant term, a rubbing seal or a magnet change moves the linear one. Only the ratios to the inertia I are
observable from speed alone, so the parameters are reported per unit inertia.

Pure functions, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

# Nested candidates, simplest first: (name, which of a, b, c are free).
CANDIDATES = (("coulomb", (True, False, False)), ("coulomb+viscous", (True, True, False)),
              ("coulomb+viscous+quadratic", (True, True, True)))
TRAIN_FRACTION = 0.7  # fit on the first 70% of the coast-down, forecast the rest
# A richer law must earn its place: lower BIC and a held-out error at least 10% better than the simpler law.
MIN_HELDOUT_GAIN = 0.10


def _rate(w: float, a: float, b: float, c: float) -> float:
    return -a - b * w - c * w * w


def coast(t: np.ndarray, w0: float, a: float, b: float, c: float, steps_per_sample: int = 20) -> np.ndarray:
    """Integrate dw/dt = -a - b w - c w^2 from w0 (RK4), clamping at standstill where dry friction holds the rotor."""
    out = np.empty_like(t, dtype=float)
    w, out[0] = w0, w0
    for i in range(1, len(t)):
        h = (t[i] - t[i - 1]) / steps_per_sample
        for _ in range(steps_per_sample):
            if w <= 0:
                w = 0.0
                break
            k1 = _rate(w, a, b, c)
            k2 = _rate(w + h * k1 / 2, a, b, c)
            k3 = _rate(w + h * k2 / 2, a, b, c)
            k4 = _rate(w + h * k3, a, b, c)
            w = max(0.0, w + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6)
        out[i] = w
    return out


@dataclass(frozen=True)
class SpinFit:
    model: str
    params: dict[str, float]  # a = tau_c/I (rad/s^2), b = b/I (1/s), c = c/I (1/rad), w0 (rad/s)
    stderr: dict[str, float]
    rmse: float  # rad/s, all points
    heldout_rmse: float  # rad/s, fit on the first TRAIN_FRACTION, error on the rest
    bic: float
    n: int


def _fit(t: np.ndarray, w: np.ndarray, free: tuple[bool, bool, bool]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = [k for k, on in zip("abc", free, strict=True) if on]
    # Start from the mean deceleration as pure Coulomb: always the right sign and order of magnitude.
    a0 = max((w[0] - w[-1]) / max(t[-1] - t[0], 1e-9), 1e-3)
    x0 = np.array([w[0]] + [a0 if k == "a" else 1e-4 for k in names])

    def model(x: np.ndarray, tt: np.ndarray) -> np.ndarray:
        p = dict(zip(names, x[1:], strict=True))
        return coast(tt - tt[0], x[0], p.get("a", 0.0), p.get("b", 0.0), p.get("c", 0.0))

    res = least_squares(lambda x: model(x, t) - w, x0, bounds=(0, np.inf), x_scale="jac")
    dof = max(len(t) - len(x0), 1)
    s2 = float(res.fun @ res.fun) / dof
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac) * s2
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(len(x0), np.nan)
    return res.x, se, res.fun


def fit_spindown(t: np.ndarray, w: np.ndarray) -> list[SpinFit]:
    """Fit every candidate law to one coast-down (t in s, w in rad/s, standstill samples already removed)."""
    fits = []
    split = max(4, round(TRAIN_FRACTION * len(t)))
    for name, free in CANDIDATES:
        names = ["w0"] + [k for k, on in zip("abc", free, strict=True) if on]
        x, se, r = _fit(t, w, free)
        xt, _, _ = _fit(t[:split], w[:split], free)
        p = dict(zip(names, xt, strict=True))
        pred = coast(t - t[0], p["w0"], p.get("a", 0.0), p.get("b", 0.0), p.get("c", 0.0))
        held = float(np.sqrt(np.mean((pred[split:] - w[split:]) ** 2)))
        n, k = len(t), len(x)
        rss = float(r @ r)
        bic = n * np.log(max(rss, 1e-300) / n) + k * np.log(n)
        fits.append(SpinFit(name, dict(zip(names, map(float, x), strict=True)),
                            dict(zip(names, map(float, se), strict=True)), float(np.sqrt(rss / n)), held, float(bic), n))
    return fits


def select_law(fits: list[SpinFit]) -> SpinFit:
    """The simplest law that no richer law beats on both BIC and held-out forecast (by MIN_HELDOUT_GAIN)."""
    best = fits[0]
    for f in fits[1:]:
        if f.bic < best.bic and f.heldout_rmse < (1 - MIN_HELDOUT_GAIN) * best.heldout_rmse:
            best = f
    return best
