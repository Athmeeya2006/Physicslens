"""Refreshing a stale published transit ephemeris with new transit mid-times, checked before it is trusted.

A published linear ephemeris (t0, P, each with a 1-sigma) predicts transit N at t0 + N P with an uncertainty that
grows as N sigma_P; years later it can be off by hours. New mid-times (here from TESS) pin it down again.

The refresh is one weighted least-squares fit of t(N) = t0 + N P to the new mid-times plus the published values
as two Gaussian prior rows (t0 = t0_pub +- sigma_t0, P = P_pub +- sigma_P). The published t0 and P are treated as
independent because the archive gives no covariance: that overstates what the prior knows slightly, and the prior
is weak next to years of new transits anyway. The prior is used only when it agrees with the new mid-times alone
(within PRIOR_TENSION_MAX sigma at their mean epoch); otherwise it is dropped and the reason recorded, since
combining two things that disagree gives a number that is right for neither.

The new mid-times' errors are scaled up by their own scatter (sqrt of the reduced chi-square, never down) before the
final fit, so the intervals reflect how well the times really agree with a straight line.

Before a refreshed ephemeris is used to forecast, it must forecast transits it was not fitted on: fitted on the
earlier observing seasons (plus the prior), it predicts the last season's transits, and the chi-square per held-out
transit (each residual over the combined forecast and measurement uncertainty) must be at most timing.HELDOUT_CHI2.

No network and no I/O here: the runner (scripts/ephemeris_refresh.py) fetches and writes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import theilslopes

from .timing import HELDOUT_CHI2

TESS_2MIN_EXP_DAYS = 120.0 / 86400.0  # SPOC 2-minute cadence, about 0.00139 d
PRIOR_TENSION_MAX = 3.0  # sigma: beyond this the published ephemeris and the new times disagree
SEASON_GAP_DAYS = 60.0  # TESS sectors closer than this belong to one observing season
MAX_FIT_CHI2_RED, FIT_CHI2_VS_MEDIAN = 3.0, 3.0  # a transit's own fit this much worse than typical is not used
# Mid-times scattering more than this many times their own errors are not understood (a systematic in the light
# curve or the fit): a held-out check passed only by inflating the errors that much does not earn registration.
MAX_ERROR_SCALE = 3.0
SEASON_OUTLIER_Z = 5.0  # robust sigma off the line through the rest of its season: a bad measurement
Z68, Z95 = 1.0, 1.959964


@dataclass(frozen=True)
class Published:
    """A catalogue ephemeris: reference mid-time and period, each with its 1-sigma (days)."""
    t0: float
    t0_err: float
    period: float
    period_err: float

    def predict(self, epochs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = np.asarray(epochs, float)
        return self.t0 + n * self.period, np.hypot(self.t0_err, n * self.period_err)

    def epoch_of(self, t: float) -> int:
        return round((t - self.t0) / self.period)


@dataclass
class Refreshed:
    """t(N) = t0 + N P with epochs counted from the published reference transit (N = 0)."""
    t0: float
    period: float
    cov: np.ndarray  # 2x2 for (t0, P), days^2, errors already scaled
    scale: float  # factor applied to the new mid-times' errors (>= 1)
    chi2_red: float  # of the new mid-times about the fit, before scaling
    n_times: int
    used_prior: bool
    prior_tension_sigma: float | None
    prior_note: str

    def predict(self, epochs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = np.asarray(epochs, float)
        t = self.t0 + n * self.period
        var = self.cov[0, 0] + 2 * n * self.cov[0, 1] + n * n * self.cov[1, 1]
        return t, np.sqrt(np.maximum(var, 0.0))

    def pivot(self) -> tuple[int, float, float]:
        """The epoch where t0 and P decorrelate (smallest timing uncertainty), its mid-time and 1-sigma: the
        natural reference epoch to publish, since its t0 error is not inflated by the period error."""
        n = round(-self.cov[0, 1] / self.cov[1, 1]) if self.cov[1, 1] > 0 else 0
        t, s = self.predict(np.array([n]))
        return n, float(t[0]), float(s[0])

    def summary(self) -> dict[str, Any]:
        n, tp, sp = self.pivot()
        return {"t0_bjd_tdb": self.t0, "t0_err_days": float(math.sqrt(self.cov[0, 0])), "period_days": self.period,
                "period_err_days": float(math.sqrt(self.cov[1, 1])), "cov_t0_p": float(self.cov[0, 1]),
                "pivot_epoch": n, "pivot_bjd_tdb": tp, "pivot_err_days": sp, "error_scale": self.scale,
                "chi2_red_new_times": self.chi2_red, "n_new_times": self.n_times, "used_prior": self.used_prior,
                "prior_tension_sigma": self.prior_tension_sigma, "prior_note": self.prior_note}


def _wls(epochs: np.ndarray, times: np.ndarray, errs: np.ndarray, pub: Published,
         prior: bool) -> tuple[np.ndarray, np.ndarray]:
    """Solve for (dt0, dP) relative to the published values: well conditioned, unlike raw BJDs near 2.46e6."""
    a = np.column_stack([np.ones_like(epochs, float), epochs.astype(float)]) / errs[:, None]
    y = (times - (pub.t0 + epochs * pub.period)) / errs
    if prior:
        a = np.vstack([a, [1.0 / pub.t0_err, 0.0], [0.0, 1.0 / pub.period_err]])
        y = np.concatenate([y, [0.0, 0.0]])  # the prior says the offsets are zero
    cov = np.linalg.inv(a.T @ a)
    return cov @ a.T @ y, cov


def well_fitted(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop transits whose own light-curve fit is poor: reduced chi-square over MAX_FIT_CHI2_RED and over
    FIT_CHI2_VS_MEDIAN times the planet's median. A transit cut by a momentum dump or a data glitch can fit a
    mid-time hours off with a modest formal error; judged on the light curve alone (no ephemeris involved), so
    removing it cannot favour any forecast."""
    med = float(np.median([r["chi2_red"] for r in rows])) if rows else 1.0
    lim = max(MAX_FIT_CHI2_RED, FIT_CHI2_VS_MEDIAN * med)
    keep = [r for r in rows if r["chi2_red"] <= lim]
    bad = [{"epoch": r.get("epoch"), "bjd_tdb": r["bjd_tdb"], "sector": r.get("sector"),
            "reason": f"poor transit fit (reduced chi-square {r['chi2_red']:.1f}, limit {lim:.1f})"}
           for r in rows if r["chi2_red"] > lim]
    return keep, bad


def season_outliers(epochs: np.ndarray, times: np.ndarray, errs: np.ndarray) -> np.ndarray:
    """True for a mid-time far off the rest of its own observing season.

    A box search that locks onto a systematic dip returns a mid-time tens of minutes to hours off, sometimes with a
    clean-looking fit. Within one season (weeks) no plausible timing variation of these planets moves one transit
    by more than a minute or two against its neighbours, so a time more than SEASON_OUTLIER_Z robust sigma off is a
    bad measurement. The slope (the period) is a Theil-Sen fit (median of pairwise slopes) over all transits, which
    one bad time cannot drag; each season's offset is that season's own median, so a whole season shifted from the
    others (what the held-out forecast tests) is never judged here. Seasons with fewer than four transits are not
    checked."""
    epochs, times, errs = np.asarray(epochs, float), np.asarray(times, float), np.asarray(errs, float)
    bad = np.zeros(len(times), bool)
    if len(times) < 4 or np.ptp(epochs) == 0:
        return bad
    slope = theilslopes(times - times[0], epochs)[0]
    r = times - times[0] - slope * epochs
    lab = seasons(times)
    for s in np.unique(lab):
        idx = np.where(lab == s)[0]
        if len(idx) < 4:
            continue
        z = (r[idx] - np.median(r[idx])) / errs[idx]
        robust = max(1.0, 1.4826 * float(np.median(np.abs(z))))
        bad[idx] = np.abs(z) > SEASON_OUTLIER_Z * robust
    return bad


def assign_epochs(times: np.ndarray, errs: np.ndarray, pub: Published) -> np.ndarray:
    """Cycle numbers for new mid-times, counted from the published reference transit.

    A stale period can be off by enough that rounding (t - t0)/P against the published values picks the wrong cycle
    years later. So the times are numbered season by season: each season against the ephemeris refitted on all
    earlier seasons (the published one for the first). A cycle slip would show as a residual of a large fraction of
    the period; the caller's fit and held-out check would then fail loudly, not silently."""
    order = np.argsort(times)
    t, e = np.asarray(times, float)[order], np.asarray(errs, float)[order]
    groups = np.split(np.arange(len(t)), np.where(np.diff(t) > SEASON_GAP_DAYS)[0] + 1)
    epochs = np.empty(len(t), int)
    t0, per = pub.t0, pub.period
    done: list[int] = []
    for idx in groups:
        epochs[idx] = np.round((t[idx] - t0) / per).astype(int)
        done += list(idx)
        if len(done) >= 2 and np.ptp(epochs[done]) > 0:
            fit = refresh(epochs[done], t[done], e[done], pub)
            t0, per = fit.t0, fit.period
    out = np.empty(len(t), int)
    out[order] = epochs
    return out


def refresh(epochs: np.ndarray, times: np.ndarray, errs: np.ndarray, pub: Published) -> Refreshed:
    """The refreshed ephemeris from new mid-times and the published one (see the module docstring)."""
    epochs, times, errs = np.asarray(epochs, int), np.asarray(times, float), np.asarray(errs, float)
    if len(times) < 2 or np.ptp(epochs) == 0:
        raise ValueError("need at least two transits at different epochs")
    # The new times alone: their scatter sets the error scale, and they are what the prior must agree with.
    x, cov = _wls(epochs, times, errs, pub, prior=False)
    resid = times - (pub.t0 + x[0] + epochs * (pub.period + x[1]))
    dof = max(len(times) - 2, 1)
    chi2_red = float(np.sum((resid / errs) ** 2) / dof) if len(times) > 2 else 1.0
    scale = max(1.0, math.sqrt(chi2_red))
    errs_s = errs * scale
    x, cov = _wls(epochs, times, errs_s, pub, prior=False)
    # Tension: the published prediction at the new times' pivot epoch against what the new times say there.
    n_piv = -cov[0, 1] / cov[1, 1]
    t_new = x[0] + n_piv * x[1]
    s_new = math.sqrt(cov[0, 0] + 2 * n_piv * cov[0, 1] + n_piv**2 * cov[1, 1])
    s_pub = math.hypot(pub.t0_err, n_piv * pub.period_err)
    usable = pub.t0_err > 0 and pub.period_err > 0
    tension = abs(t_new) / math.hypot(s_new, s_pub) if usable else None
    if not usable:
        use, note = False, "published ephemeris lacks an uncertainty on t0 or P: not used as a prior"
    elif tension > PRIOR_TENSION_MAX:
        use, note = False, (f"published ephemeris is {tension:.1f} sigma off the new times at their mean epoch: "
                            "not combined (it would pull the fit toward a value both disagree on)")
    else:
        use, note = True, f"published ephemeris agrees with the new times ({tension:.1f} sigma) and is a prior"
    if use:
        x, cov = _wls(epochs, times, errs_s, pub, prior=True)
    return Refreshed(pub.t0 + float(x[0]), pub.period + float(x[1]), cov, scale, chi2_red, len(times), use,
                     None if tension is None else float(tension), note)


def seasons(times: np.ndarray) -> np.ndarray:
    """Season index per time (sorted order not assumed)."""
    order = np.argsort(times)
    lab = np.empty(len(times), int)
    lab[order] = np.concatenate([[0], np.cumsum(np.diff(np.asarray(times)[order]) > SEASON_GAP_DAYS)])
    return lab


def heldout(epochs: np.ndarray, times: np.ndarray, errs: np.ndarray, pub: Published) -> dict[str, Any]:
    """Fit on all but the last observing season, forecast that season. Passing needs a held-out chi-square per
    transit of at most HELDOUT_CHI2, with each residual over sqrt(forecast sigma^2 + (scaled) measurement sigma^2).
    Needs at least two seasons and three training transits; otherwise there is nothing to hold out and it fails."""
    epochs, times, errs = np.asarray(epochs, int), np.asarray(times, float), np.asarray(errs, float)
    lab = seasons(times)
    test = lab == lab.max()
    train = ~test
    if lab.max() == 0 or train.sum() < 3 or np.ptp(epochs[train]) == 0:
        return {"passed": False, "reason": "only one observing season (or under three earlier transits): no held-out "
                                           "season to test a forecast on", "n_train": int(train.sum()),
                "n_test": int(test.sum())}
    fit = refresh(epochs[train], times[train], errs[train], pub)
    t, s = fit.predict(epochs[test])
    resid = times[test] - t
    z2 = resid**2 / (s**2 + (errs[test] * fit.scale) ** 2)
    chi2 = float(np.mean(z2))
    gap_days = float(times[test].min() - times[train].max())
    passed = chi2 <= HELDOUT_CHI2
    return {"passed": bool(passed), "chi2_per_transit": chi2, "rms_min": float(np.sqrt(np.mean(resid**2)) * 1440),
            "max_abs_z": float(np.sqrt(z2.max())), "inside_95": float(np.mean(np.sqrt(z2) <= Z95)),
            "n_train": int(train.sum()), "n_test": int(test.sum()), "train_until_bjd": float(times[train].max()),
            "test_from_bjd": float(times[test].min()), "forecast_gap_days": gap_days,
            "forecast_sigma_min_median": float(np.median(s) * 1440), "rule": f"chi2 per held-out transit <= {HELDOUT_CHI2}",
            "reason": (f"forecast {int(test.sum())} later transits {gap_days:.0f} days ahead with chi-square "
                       f"{chi2:.2f} per transit" + ("" if passed else f" (over {HELDOUT_CHI2})"))}


def windows(fit: Refreshed, pub: Published, start_bjd: float, end_bjd: float,
            duration_days: float | None = None) -> list[dict[str, Any]]:
    """Every transit between two dates: refreshed mid-time with 68% and 95% intervals, beside the published
    ephemeris' prediction and 95% interval, so the shift is visible."""
    n0, n1 = math.ceil((start_bjd - fit.t0) / fit.period), math.floor((end_bjd - fit.t0) / fit.period)
    ep = np.arange(n0, n1 + 1)
    t, s = fit.predict(ep)
    tp, sp = pub.predict(ep)
    out = []
    for e, ti, si, tpi, spi in zip(ep, t, s, tp, sp, strict=True):
        row = {"epoch": int(e), "value": float(ti), "sigma_days": float(si),
               "ci68": [float(ti - Z68 * si), float(ti + Z68 * si)], "ci95": [float(ti - Z95 * si), float(ti + Z95 * si)],
               "published_ephemeris_value": float(tpi), "published_sigma_days": float(spi),
               "published_ci95": [float(tpi - Z95 * spi), float(tpi + Z95 * spi)],
               "shift_min": float((ti - tpi) * 1440)}
        if duration_days:
            row["duration_days"] = duration_days
        out.append(row)
    return out


def score_transit(pred: dict[str, Any], measured: float, measured_err: float) -> dict[str, Any]:
    """One predicted transit against an independent mid-time. Hit when the measured time lies inside the 95%
    window widened in quadrature by the measurement's own 95% error (a timing good to minutes cannot test a window
    of seconds); the strict test (inside the window as registered) is reported too."""
    sig = float(pred.get("sigma_days", (pred["ci95"][1] - pred["ci95"][0]) / (2 * Z95)))
    half = Z95 * math.hypot(sig, measured_err)
    off = measured - pred["value"]
    return {"label": pred["label"], "predicted": pred["value"], "measured": measured, "measured_err_days": measured_err,
            "offset_min": off * 1440, "hit": bool(abs(off) <= half),
            "inside_registered_ci95": bool(pred["ci95"][0] <= measured <= pred["ci95"][1])}


def exoclock_csv(rows: list[dict[str, Any]]) -> str:
    """Refreshed ephemerides with ExoClock's own field names (its planets database: t0_bjd_tdb, t0_unc, period_days,
    period_unc, duration_hours, depth_mmag), uncertainties in days, the reference epoch at the pivot so t0 and P
    are nearly uncorrelated; then provenance columns."""
    head = ("name,t0_bjd_tdb,t0_unc,period_days,period_unc,duration_hours,depth_mmag,n_tess_transits,"
            "first_tess_bjd,last_tess_bjd,published_t0_bjd_tdb,published_period_days,source")
    lines = [head]
    for r in rows:
        lines.append(",".join([r["name"], f"{r['t0_bjd_tdb']:.6f}", f"{r['t0_unc']:.6f}", f"{r['period_days']:.8f}",
                               f"{r['period_unc']:.8f}", f"{r['duration_hours']:.3f}" if r.get("duration_hours") else "",
                               f"{r['depth_mmag']:.2f}" if r.get("depth_mmag") else "", str(r["n_tess_transits"]),
                               f"{r['first_tess_bjd']:.5f}", f"{r['last_tess_bjd']:.5f}",
                               f"{r['published_t0_bjd_tdb']:.6f}", f"{r['published_period_days']:.8f}",
                               '"' + r["source"] + '"']))
    return "\n".join(lines) + "\n"


def midtimes_csv(name: str, rows: list[dict[str, Any]]) -> str:
    """Individual mid-times (BJD_TDB, 1-sigma in days, as fitted and unscaled), one per line, for an O-C diagram."""
    head = "name,epoch,mid_bjd_tdb,mid_err_days,sector,source"
    body = [f"{name},{r['epoch']},{r['bjd_tdb']:.6f},{r['err_days']:.6f},{r.get('sector', '')},\"{r.get('source', '')}\""
            for r in sorted(rows, key=lambda r: r["bjd_tdb"])]
    return "\n".join([head, *body]) + "\n"
