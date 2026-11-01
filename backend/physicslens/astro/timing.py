"""Transit and eclipse timing: refreshed ephemerides, the next transit windows, and what the timings say.

Laws for event times against epoch N (forms as in Yee et al. 2020, ApJL 888 L5, eqs. 1 to 3, checked in its source):
  linear       t_tra = t0 + N P;                          t_occ = t0 + P/2 + N P
  decay        + (1/2) (dP/dN) N^2 on both                 (the period changing at a steady rate)
  precession   t_tra = t0 + N Ps - (e Pa / pi) cos w(N);  t_occ = t0 + Pa/2 + N Ps + (e Pa / pi) cos w(N),
               w(N) = w0 + (dw/dN) N, Ps = Pa (1 - (dw/dN) / 2 pi)   (an eccentric orbit turning)
  sinusoid     + A sin(2 pi N / N_ttv + phi)                (a companion, or the light-travel time to a third body)

Verified as every PhysicsLens law: fitted on the earlier 75% of the events (by time), it must forecast the later 25%
(held-out chi-square per event at most HELDOUT_CHI2, every law's errors scaled by the same factor: the training
scatter of the best-fitting law), and
among those that do, BIC = chi^2 + k ln n (the paper's definition) decides with a lead of DECISIVE_DELTA_BIC.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import least_squares

TRAIN_FRACTION = 0.75
HELDOUT_CHI2 = 4.0  # per event, with errors scaled by the training scatter: generous, timing errors are often optimistic
DECISIVE_DELTA_BIC = 10.0
MS_PER_DAY = 86400e3
DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class Law:
    id: str
    title: str
    params: tuple[str, ...]
    model: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]  # (theta, epoch, is_occultation) -> time
    latex: str


def _linear(p, n, occ):
    t0, per = p
    return t0 + n * per + occ * per / 2


def _decay(p, n, occ):
    t0, per, dpdn = p
    return t0 + n * per + 0.5 * dpdn * n * n + occ * per / 2


def _precession(p, n, occ):
    t0, ps, e, w0, dwdn = p
    pa = ps / (1 - dwdn / (2 * math.pi))
    sign = np.where(occ, 1.0, -1.0)
    return t0 + n * ps + occ * pa / 2 + sign * (e * pa / math.pi) * np.cos(w0 + dwdn * n)


def _sinusoid(p, n, occ):
    t0, per, amp, nttv, phi = p
    return t0 + n * per + occ * per / 2 + amp * np.sin(2 * math.pi * n / nttv + phi)


LAWS = {
    "linear": Law("linear", "Constant period (nothing happening)", ("t0", "P"), _linear, r"t(N) = t_0 + N P"),
    "decay": Law("decay", "Changing period (orbital decay or growth)", ("t0", "P", "dPdN"), _decay,
                 r"t(N) = t_0 + N P + \tfrac{1}{2}\tfrac{dP}{dN} N^2"),
    "precession": Law("precession", "Apsidal precession (an eccentric orbit turning)", ("t0", "Ps", "e", "w0", "dwdN"),
                      _precession, r"t(N) = t_0 + N P_s \mp \tfrac{e P_a}{\pi}\cos\left(\omega_0 + \tfrac{d\omega}{dN}N\right)"),
    "sinusoid": Law("sinusoid", "Periodic timing variation (a companion, or light-travel time)", ("t0", "P", "A", "Nttv", "phi"),
                    _sinusoid, r"t(N) = t_0 + N P + A \sin\left(2\pi N / N_\mathrm{TTV} + \varphi\right)"),
}


@dataclass
class Timings:
    epoch: np.ndarray  # integer cycle number
    time: np.ndarray  # BJD_TDB
    err: np.ndarray  # days
    occultation: np.ndarray  # bool
    source: list[str] = field(default_factory=list)

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]]) -> Timings:
        return cls(np.array([int(r["epoch"]) for r in rows]), np.array([float(r["bjd_tdb"]) for r in rows]),
                   np.array([float(r["err_days"]) for r in rows]), np.array([r["event"] == "occ" for r in rows]),
                   [r.get("source", "") for r in rows])

    def subset(self, mask: np.ndarray) -> Timings:
        return Timings(self.epoch[mask], self.time[mask], self.err[mask], self.occultation[mask],
                       [s for s, m in zip(self.source, mask, strict=True) if m] if self.source else [])


@dataclass
class LawFit:
    law: Law
    theta: np.ndarray
    cov: np.ndarray
    chi2: float
    bic: float
    n: int
    scale: float  # sqrt of the training chi2 per degree of freedom (>= 1): how optimistic the errors were
    heldout_chi2: float = math.nan  # per held-out event, errors scaled
    heldout_rms_min: float = math.nan

    @property
    def params(self) -> dict[str, float]:
        return dict(zip(self.law.params, (float(v) for v in self.theta), strict=True))

    def sigma(self) -> dict[str, float]:
        return dict(zip(self.law.params, (float(math.sqrt(max(v, 0.0))) for v in np.diag(self.cov)), strict=True))


def _start(law: Law, d: Timings, base: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    t0, per = base
    span = max(abs(d.epoch).max(), 1)
    if law.id == "linear":
        return [base]
    if law.id == "decay":
        return [np.array([t0, per, 0.0])]
    if law.id == "precession":
        return [np.array([t0, per, 0.002, w, 2 * math.pi / s]) for w in (0.0, 2.0, 4.0) for s in (span, 2 * span, 4 * span)]
    # sinusoid: periods from a few cycles to twice the span, phases from the residual's own fit
    return [np.array([t0, per, 1e-4, nt, ph]) for nt in np.geomspace(4, 2 * span, 10) for ph in (0.0, 2.0, 4.0)]


def fit_law(law: Law, d: Timings, base: np.ndarray, seed: int = 0) -> LawFit:
    """Weighted least squares from several starts (periodic laws have many local minima); the best is kept."""
    rng = np.random.default_rng(seed)
    best = None
    for p0 in _start(law, d, base, rng):
        try:
            r = least_squares(lambda p: (law.model(p, d.epoch, d.occultation) - d.time) / d.err, p0, x_scale="jac",
                              method="lm" if len(p0) <= len(d.time) else "trf")
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    assert best is not None
    chi2 = float(2 * best.cost)
    k, n = len(best.x), len(d.time)
    try:
        cov = np.linalg.inv(best.jac.T @ best.jac)
    except np.linalg.LinAlgError:
        cov = np.full((k, k), np.inf)
    scale = max(1.0, math.sqrt(chi2 / max(n - k, 1)))
    return LawFit(law, best.x, cov * scale**2, chi2, chi2 + k * math.log(n), n, scale)


def linear_start(d: Timings) -> np.ndarray:
    tra = ~d.occultation
    coef = np.polyfit(d.epoch[tra], d.time[tra], 1, w=1 / d.err[tra])
    return np.array([coef[1], coef[0]])


def assess(d: Timings, laws: tuple[str, ...] = ("linear", "decay", "precession", "sinusoid")) -> dict[str, Any]:
    """Fit each law on the earlier 75% of events, forecast the later 25%, then refit the survivors on everything
    and let BIC decide."""
    order = np.argsort(d.time)
    cut = d.time[order[int(len(order) * TRAIN_FRACTION)]]
    train, test = d.subset(d.time < cut), d.subset(d.time >= cut)
    base = linear_start(d)
    trained = {lid: fit_law(LAWS[lid], train, base) for lid in laws}
    # One error scale for every law: the scatter of the best-fitting one says how good the timings really are. A law
    # scaled by its own scatter would excuse itself: missing hours of variation, it inflated its errors to match.
    common = min(tr.scale for tr in trained.values())
    rows = []
    for lid in laws:
        law, tr = LAWS[lid], trained[lid]
        resid = (law.model(tr.theta, test.epoch, test.occultation) - test.time) / (test.err * common)
        held = float(np.mean(resid**2))
        full = fit_law(law, d, base)
        full.heldout_chi2 = held
        full.heldout_rms_min = float(np.sqrt(np.mean(((law.model(tr.theta, test.epoch, test.occultation) - test.time)
                                                        * 1440) ** 2)))
        rows.append(full)
    passing = [f for f in rows if f.heldout_chi2 <= HELDOUT_CHI2]
    passing.sort(key=lambda f: f.bic)
    if not passing:
        status, winner, reason = "no_fit", None, "No law forecasts the later timings within their uncertainties."
    elif len(passing) > 1 and passing[1].bic - passing[0].bic < DECISIVE_DELTA_BIC:
        status, winner = "ambiguous", passing[0]
        reason = (f"{passing[0].law.title} and {passing[1].law.title} both forecast the later timings and are within "
                  f"{DECISIVE_DELTA_BIC:.0f} BIC: the timings cannot tell them apart yet.")
    else:
        status, winner = "accepted", passing[0]
        nxt = f"; next best ({passing[1].law.title}) is {passing[1].bic - passing[0].bic:.0f} BIC worse" if len(passing) > 1 else ""
        reason = (f"{winner.law.title} forecasts the later {len(test.time)} timings (chi-square {winner.heldout_chi2:.2f} "
                  f"per event){nxt}.")
    return {"status": status, "reason": reason, "winner": winner, "fits": rows, "train_until_bjd": float(cut),
            "n_train": len(train.time), "n_test": len(test.time)}


def period_change_ms_per_year(fit: LawFit) -> tuple[float, float]:
    """dP/dt from the decay law: (dP/dN) / P, in milliseconds per year, with its 1-sigma."""
    p, s = fit.params, fit.sigma()
    rate = p["dPdN"] / p["P"]  # days per day
    return rate * MS_PER_DAY * DAYS_PER_YEAR, s["dPdN"] / p["P"] * MS_PER_DAY * DAYS_PER_YEAR


def predict(fit: LawFit, epochs: np.ndarray, occultation: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Event times and 1-sigma (days) at future epochs, from the law and its covariance (numerical gradient)."""
    occ = np.full(len(epochs), occultation)
    t = fit.law.model(fit.theta, epochs, occ)
    J = np.zeros((len(epochs), len(fit.theta)))
    for i, v in enumerate(fit.theta):
        h = max(abs(v) * 1e-7, 1e-12)
        dp = fit.theta.copy()
        dp[i] = v + h
        J[:, i] = (fit.law.model(dp, epochs, occ) - t) / h
    var = np.einsum("ij,jk,ik->i", J, fit.cov, J)
    return t, np.sqrt(np.maximum(var, 0.0))


def next_windows(fit: LawFit, after_bjd: float, count: int = 10, duration_days: float | None = None) -> list[dict[str, Any]]:
    """The next transits after a date: mid-time (BJD_TDB and UTC) with 1-sigma, for an observer's schedule."""
    from astropy.time import Time
    p = fit.params
    per = p.get("P", p.get("Ps"))
    n0 = math.ceil((after_bjd - p["t0"]) / per)
    epochs = np.arange(n0, n0 + count)
    t, s = predict(fit, epochs)
    out = []
    for e, ti, si in zip(epochs, t, s, strict=True):
        utc = Time(ti, format="jd", scale="tdb").utc  # BJD_TDB to UTC, ignoring the barycentric light time (< 8.3 min)
        out.append({"epoch": int(e), "bjd_tdb": float(ti), "sigma_min": float(si * 1440),
                    "utc_approx": utc.isot[:19], **({"duration_h": duration_days * 24} if duration_days else {})})
    return out


def windows_csv(name: str, rows: list[dict[str, Any]]) -> str:
    head = "planet,epoch,mid_bjd_tdb,sigma_minutes,mid_utc_approx"
    return "\n".join([head, *(f"{name},{r['epoch']},{r['bjd_tdb']:.6f},{r['sigma_min']:.2f},{r['utc_approx']}" for r in rows)]) + "\n"


def windows_ics(name: str, rows: list[dict[str, Any]], half_window_min: float | None = None) -> str:
    """A calendar file: one event per transit, spanning the 3-sigma window around the predicted mid-time."""
    from datetime import datetime, timedelta
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//PhysicsLens//transit windows//EN"]
    for r in rows:
        mid = datetime.fromisoformat(r["utc_approx"])
        half = timedelta(minutes=half_window_min if half_window_min is not None else 3 * r["sigma_min"])
        fmt = "%Y%m%dT%H%M%SZ"
        lines += ["BEGIN:VEVENT", f"UID:{name.replace(' ', '')}-{r['epoch']}@physicslens", f"DTSTAMP:{mid.strftime(fmt)}",
                  f"DTSTART:{(mid - half).strftime(fmt)}", f"DTEND:{(mid + half).strftime(fmt)}",
                  f"SUMMARY:{name} transit mid-time ±{r['sigma_min']:.1f} min (1 sigma)", "END:VEVENT"]
    return "\r\n".join([*lines, "END:VCALENDAR"]) + "\r\n"


def ephemeris_sigma_minutes(t0: float, t0_err: float, period: float, period_err: float, at_bjd: float) -> float:
    """How uncertain a published linear ephemeris makes a transit at a date: sqrt(sigma_t0^2 + (N sigma_P)^2)."""
    n = abs(at_bjd - t0) / period
    return float(math.hypot(t0_err, n * period_err) * 1440)


def ephemeris_rot(rows: list[dict[str, Any]], at_bjd: float) -> list[dict[str, Any]]:
    """Every planet with a published ephemeris, ranked by how uncertain its transit time is at a date."""
    out = []
    for r in rows:
        try:
            t0, per = float(r["pl_tranmid"]), float(r["pl_orbper"])
            t0e = max(abs(float(r["pl_tranmiderr1"] or 0)), abs(float(r["pl_tranmiderr2"] or 0)))
            pe = max(abs(float(r["pl_orbpererr1"] or 0)), abs(float(r["pl_orbpererr2"] or 0)))
        except (TypeError, ValueError, KeyError):
            continue
        if not (per > 0 and t0 > 0 and (t0e > 0 or pe > 0)):
            continue
        sig = ephemeris_sigma_minutes(t0, t0e, per, pe, at_bjd)
        out.append({"planet": r["pl_name"], "sigma_min": sig, "period_days": per, "t0_bjd": t0,
                    "vmag": float(r["sy_vmag"]) if r.get("sy_vmag") not in (None, "") else None,
                    "over_10_min": sig > 10, "over_1_h": sig > 60})
    return sorted(out, key=lambda x: -x["sigma_min"])
