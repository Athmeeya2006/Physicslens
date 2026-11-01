"""Mid-times of individual transits from a light curve, for timing (astro/timing.py).

Per transit: find the dip near its predicted time (a box search, so transit-timing variations of hours are found),
detrend locally with a polynomial fitted outside the dip, then fit its mid-time with a transit model (batman) whose
shape (radius ratio, a/R*, inclination, limb darkening) was fitted once on all transits stacked at their own
mid-times. Partial transits and failed fits are rejected with the reason logged, never used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .photometry import event_timing

KEPLER_LONG_CADENCE_DAYS = 0.0204336  # 29.4 min


@dataclass
class Shape:
    period: float
    rp: float  # radius ratio
    a: float  # a / R*
    inc: float  # degrees
    q1: float  # Kipping (2013) limb-darkening parameters, both in [0, 1]
    q2: float


def _batman(t: np.ndarray, mid: float, s: Shape, exp_time: float) -> np.ndarray:
    import batman
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = mid, s.period, s.rp, s.a, s.inc, 0.0, 90.0
    u1, u2 = 2 * math.sqrt(s.q1) * s.q2, math.sqrt(s.q1) * (1 - 2 * s.q2)
    p.u, p.limb_dark = [u1, u2], "quadratic"
    m = batman.TransitModel(p, t, supersample_factor=7 if exp_time > 0.005 else 1, exp_time=exp_time)
    return m.light_curve(p)


def _box_search(t: np.ndarray, f: np.ndarray, duration: float) -> float:
    centres = np.linspace(t.min() + duration / 2, t.max() - duration / 2, 400)
    depth = [np.mean(f[np.abs(t - c) < duration / 2]) if np.any(np.abs(t - c) < duration / 2) else np.inf for c in centres]
    return float(centres[int(np.argmin(depth))])


def cut_transits(t: np.ndarray, f: np.ndarray, e: np.ndarray, t0: float, period: float, duration: float,
                 search_half_width: float, min_coverage: float = 0.7,
                 known: list[tuple[int, float]] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Each epoch's window, detrended, with a first mid-time from a trapezoid; and the rejected epochs with reasons."""
    n_first, n_last = math.ceil((t.min() - t0) / period), math.floor((t.max() - t0) / period)
    half = search_half_width + 1.5 * duration
    cad = float(np.median(np.diff(t)))
    kept, rejected = [], []
    good: list[tuple[int, float]] = []  # (epoch, O-C in days) of the transits found so far
    for n in range(n_first, n_last + 1):
        # Follow the timing curve: the window is centred where the last few transits say this one will be, so
        # variations of many hours (Kepler-88 b reaches 18) never carry a transit out of its window.
        drift = 0.0
        if known and known[0][0] < n < known[-1][0]:
            # A second pass: inside the span already measured, interpolate across gaps between found transits.
            ke, ko = np.array(known).T
            drift = float(np.interp(n, ke, ko))
        elif len(good) >= 3:
            e_, oc_ = np.array(good[-4:]).T
            drift = float(np.polyval(np.polyfit(e_, oc_, 1 if len(good) < 6 else 2), n)) if len(good) >= 3 else 0.0
            drift = float(np.clip(drift, -search_half_width * 4, search_half_width * 4))
        pred = t0 + n * period + drift
        w = np.abs(t - pred) < half
        if w.sum() < 10:
            continue  # no data at all near this transit (a gap between quarters): nothing to report
        tw, fw, ew = t[w], f[w], e[w]
        mid0 = _box_search(tw, fw, duration)
        inside = np.abs(tw - mid0) < duration / 2
        expected = duration / cad
        if inside.sum() < min_coverage * expected:
            rejected.append({"epoch": n, "reason": f"partial: {inside.sum()} of about {expected:.0f} in-transit points"})
            continue
        out = np.abs(tw - mid0) > 0.75 * duration
        if out.sum() < 10:
            rejected.append({"epoch": n, "reason": "too little data around the transit to detrend"})
            continue
        coef = np.polyfit(tw[out] - mid0, fw[out], 2)
        trend = np.polyval(coef, tw - mid0)
        fn, en = fw / trend, ew / trend
        try:
            ev = event_timing(tw, fn, en, guess_mid=mid0)
        except (ValueError, np.linalg.LinAlgError) as err:
            rejected.append({"epoch": n, "reason": f"first fit failed: {err}"})
            continue
        kept.append({"epoch": n, "t": tw, "f": fn, "e": en, "mid": ev.mid_time, "depth": ev.depth})
        if ev.depth > 0:
            good.append((n, ev.mid_time - (t0 + n * period)))
    # A "dip" much shallower than the transits is noise found where no transit was in the window.
    if kept:
        typical = float(np.median([k["depth"] for k in kept]))
        shallow = [k for k in kept if k["depth"] < 0.4 * typical]
        rejected += [{"epoch": k["epoch"], "reason": f"no clear transit (dip {k['depth'] / typical:.0%} of the typical depth)"}
                     for k in shallow]
        kept = [k for k in kept if k["depth"] >= 0.4 * typical]
    return kept, rejected


def fit_shape(transits: list[dict[str, Any]], period: float, duration: float, exp_time: float) -> Shape:
    """One transit shape for all: the transits stacked at their own mid-times."""
    tt = np.concatenate([tr["t"] - tr["mid"] for tr in transits])
    ff = np.concatenate([tr["f"] for tr in transits])
    ee = np.concatenate([tr["e"] for tr in transits])
    depth = float(np.median([tr["depth"] for tr in transits]))
    a0 = max(2.0, period / (math.pi * duration))  # from the duration of a central transit
    p0 = np.array([math.sqrt(max(depth, 1e-5)), a0, 89.0, 0.4, 0.3])

    def resid(p: np.ndarray) -> np.ndarray:
        return (_batman(tt, 0.0, Shape(period, *p), exp_time) - ff) / ee
    r = least_squares(resid, p0, bounds=([0.001, 1.5, 60.0, 0.0, 0.0], [0.5, 200.0, 90.0, 1.0, 1.0]), x_scale="jac")
    return Shape(period, *r.x)


def fit_mid_times(transits: list[dict[str, Any]], shape: Shape, exp_time: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Each transit's mid-time with the shape fixed (free: mid-time and a linear baseline), 1-sigma from the fit's
    covariance scaled by its own residual scatter."""
    rows, rejected = [], []
    for tr in transits:
        t, f, e = tr["t"], tr["f"], tr["e"]

        def resid(p: np.ndarray, t=t, f=f, e=e) -> np.ndarray:
            mid, c0, c1 = p
            return (_batman(t, mid, shape, exp_time) * (c0 + c1 * (t - mid)) - f) / e
        r = least_squares(resid, np.array([tr["mid"], 1.0, 0.0]), x_scale="jac")
        chi2 = float(np.sum(r.fun**2) / max(len(t) - 3, 1))
        try:
            cov = np.linalg.inv(r.jac.T @ r.jac) * max(chi2, 1.0)
            sigma = float(math.sqrt(cov[0, 0]))
        except np.linalg.LinAlgError:
            rejected.append({"epoch": tr["epoch"], "reason": "the mid-time is not constrained"})
            continue
        if not (np.isfinite(sigma) and sigma < 0.05):
            rejected.append({"epoch": tr["epoch"], "reason": f"mid-time uncertainty {sigma * 1440:.0f} min: no clear transit"})
            continue
        rows.append({"event": "tra", "epoch": tr["epoch"], "bjd_tdb": float(r.x[0]), "err_days": sigma,
                     "source": "PhysicsLens fit to Kepler PDCSAP long cadence", "chi2_red": chi2})
    return rows, rejected


def refine_ephemeris(t: np.ndarray, f: np.ndarray, t0: float, period: float, duration: float,
                     tolerance: float = 0.01) -> tuple[float, float]:
    """The mean period and a reference mid-time from the light curve itself (box least squares within tolerance of
    the catalogue period). A catalogue period fitted through large timing variations can be off by enough that
    predicted windows drift days away from the real transits over a mission."""
    from astropy.timeseries import BoxLeastSquares
    bls = BoxLeastSquares(t, f)
    periods = np.linspace(period * (1 - tolerance), period * (1 + tolerance), 4000)
    res = bls.power(periods, duration)
    i = int(np.argmax(res.power))
    p_best, t_best = float(res.period[i]), float(res.transit_time[i])
    n = round((t0 - t_best) / p_best)
    return t_best + n * p_best, p_best


OFF_CURVE_MIN = 60.0  # minutes from the neighbours' timing curve (and 8 sigma) before a time is called a noise dip
OFF_CURVE_TYPICAL = 5.0  # ... and five times the median transit's distance from its own neighbours' line


def off_curve(rows: list[dict[str, Any]], t0: float, period: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Timing variations are smooth from one transit to the next; a time far off its neighbours' line (up to three
    on each side) was a noise dip picked where the transit was missed. "Far" is judged against how far every other
    transit sits from its own neighbours' line, so a fast, real wobble (which bends away from straight lines
    everywhere) is not mistaken for noise."""
    if len(rows) < 7:
        return rows, []
    ep = np.array([r["epoch"] for r in rows])
    oc = np.array([r["bjd_tdb"] - (t0 + r["epoch"] * period) for r in rows]) * 1440
    offs = np.empty(len(rows))
    for i in range(len(rows)):
        nb = [j for j in range(max(0, i - 3), min(len(rows), i + 4)) if j != i]
        # The neighbours' line through this epoch: robust to the curve's slope, unlike a plain median.
        fit = np.polyfit(ep[nb], oc[nb], 1)
        offs[i] = abs(oc[i] - np.polyval(fit, ep[i]))
    typical = float(np.median(offs))
    keep, rejected = [], []
    for r, off in zip(rows, offs, strict=True):
        if off > max(OFF_CURVE_MIN, 8 * r["err_days"] * 1440, OFF_CURVE_TYPICAL * typical):
            rejected.append({"epoch": r["epoch"], "reason": f"off the timing curve of its neighbours by {off:.0f} min"})
        else:
            keep.append(r)
    return keep, rejected


def transit_times(t: np.ndarray, f: np.ndarray, e: np.ndarray, t0: float, period: float, duration: float,
                  search_half_width: float = 0.5, exp_time: float = KEPLER_LONG_CADENCE_DAYS,
                  refine: bool = False) -> dict[str, Any]:
    """The whole step: windows, one shape, every mid-time, and the log of what was rejected and why."""
    f = f / np.nanmedian(f)
    e = e / np.nanmedian(f) if np.nanmedian(e) > 0.5 else e
    if refine:
        t0, period = refine_ephemeris(t, f, t0, period, duration)
    kept, rejected = cut_transits(t, f, e, t0, period, duration, search_half_width)
    if len(kept) >= 6:  # second pass, guided by the first one's timing curve (recovers transits lost after gaps)
        known = sorted((k["epoch"], k["mid"] - (t0 + k["epoch"] * period)) for k in kept)
        kept2, rejected2 = cut_transits(t, f, e, t0, period, duration, search_half_width, known=known)
        if len(kept2) > len(kept):
            kept, rejected = kept2, rejected2
    if len(kept) < 3:
        raise ValueError(f"only {len(kept)} usable transits")
    shape = fit_shape(kept, period, duration, exp_time)
    rows, rejected2 = fit_mid_times(kept, shape, exp_time)
    rows, rejected3 = off_curve(rows, t0, period)
    return {"rows": rows, "rejected": rejected + rejected2 + rejected3, "shape": shape.__dict__,
            "reference": {"t0_bjd": t0, "mean_period_days": period}}
