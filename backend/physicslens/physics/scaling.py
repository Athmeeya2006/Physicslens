"""A law ACROSS recordings: how a measured quantity scales with a variable the experimenter controls.

Each recording gives one point (x, y +/- sigma): a pendulum's period at one string length, a spring's frequency with
one mass. A closed set of laws is fitted to the points; each must predict every recording it was not fitted to
(leave one out), and BIC breaks ties between those that do. The reasoning model may say which law it expects; that
expectation is shown next to the result and never decides it. The best law then predicts the next planned recording
with a 95% interval before it is filmed, and that prediction is checked when the recording arrives.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

MAX_LOO_Z = 3.0  # a law must predict each left-out recording within 3 combined standard deviations
DECISIVE_DELTA_BIC = 6.0  # with a handful of points, "strong" evidence rather than the single-clip 10
Z95 = 1.96


@dataclass(frozen=True)
class Law:
    id: str
    title: str
    latex: str
    n_params: int
    f: Callable[..., np.ndarray]
    p0: Callable[[np.ndarray, np.ndarray], list[float]]


def _ratio_guess(x: np.ndarray, y: np.ndarray, b: float) -> list[float]:
    return [float(np.median(y / np.power(x, b)))]


def _fixed_power(b: float, title: str, latex: str) -> Law:
    return Law(f"power_{b:g}", title, latex, 1, lambda x, a, b=b: a * np.power(x, b), lambda x, y, b=b: _ratio_guess(x, y, b))


LAWS: dict[str, Law] = {law.id: law for law in [
    Law("constant", "No dependence", r"y = a", 1, lambda x, a: np.full_like(x, a, dtype=float),
        lambda x, y: [float(np.mean(y))]),
    _fixed_power(0.5, "Square-root law", r"y = a\,x^{1/2}"),
    _fixed_power(-0.5, "Inverse square-root law", r"y = a\,x^{-1/2}"),
    _fixed_power(1.0, "Proportional", r"y = a\,x"),
    _fixed_power(-1.0, "Inversely proportional", r"y = a\,x^{-1}"),
    _fixed_power(2.0, "Square law", r"y = a\,x^{2}"),
    Law("power", "Power law, exponent fitted", r"y = a\,x^{b}", 2, lambda x, a, b: a * np.power(x, b),
        lambda x, y: [float(np.median(y)), 0.0]),
    Law("linear", "Straight line", r"y = a + b\,x", 2, lambda x, a, b: a + b * x,
        lambda x, y: [float(np.mean(y)), 0.0]),
]}


@dataclass(frozen=True)
class LawFit:
    law: str
    params: list[float]
    cov: np.ndarray
    chi2: float
    bic: float
    loo_z: list[float]  # standardized error predicting each recording from all the others

    def predict(self, x: float) -> tuple[float, float]:
        """Prediction and its 1-sigma uncertainty from the parameter covariance (delta method)."""
        law = LAWS[self.law]
        p = np.asarray(self.params)
        y = float(law.f(np.array([x], float), *p)[0])
        eps = 1e-6 * np.maximum(np.abs(p), 1e-9)
        grad = np.array([(law.f(np.array([x], float), *(p + np.eye(len(p))[i] * eps[i]))[0] - y) / eps[i]
                         for i in range(len(p))])
        return y, float(math.sqrt(max(grad @ self.cov @ grad, 0.0)))


def _fit(law: Law, x: np.ndarray, y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    try:
        p, cov = curve_fit(law.f, x, y, p0=law.p0(x, y), sigma=s, absolute_sigma=True, maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(cov)):
        return None
    chi2 = float(np.sum(((y - law.f(x, *p)) / s) ** 2))
    dof = len(x) - law.n_params
    if dof > 0 and chi2 / dof > 1:
        cov = cov * (chi2 / dof)  # scatter beyond the error bars widens the uncertainty, never narrows it
    return p, cov, chi2


def fit_law(law_id: str, x: np.ndarray, y: np.ndarray, s: np.ndarray) -> LawFit | None:
    law = LAWS[law_id]
    if len(x) < law.n_params + 1:
        return None  # it needs at least one recording it was not fitted to
    full = _fit(law, x, y, s)
    if full is None:
        return None
    p, cov, chi2 = full
    loo = []
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        sub = _fit(law, x[keep], y[keep], s[keep])
        if sub is None:
            return None
        lf = LawFit(law_id, list(sub[0]), sub[1], sub[2], 0.0, [])
        yhat, sp = lf.predict(float(x[i]))
        loo.append(float((y[i] - yhat) / math.hypot(s[i], sp)))
    return LawFit(law_id, [float(v) for v in p], cov, chi2, chi2 + law.n_params * math.log(len(x)), loo)


@dataclass(frozen=True)
class ScalingVerdict:
    status: str  # accepted | ambiguous | no_fit | too_few
    winner: LawFit | None
    contenders: list[LawFit]
    ranked: list[LawFit]
    reason: str


def assess_scaling(x: list[float], y: list[float], sigma: list[float]) -> ScalingVerdict:
    xa, ya, sa = (np.asarray(v, float) for v in (x, y, sigma))
    if len(xa) < 3 or len(set(xa.round(9))) < 3:
        return ScalingVerdict("too_few", None, [], [], "A law across recordings needs at least three different settings.")
    if np.any(xa <= 0) or np.any(sa <= 0):
        return ScalingVerdict("no_fit", None, [], [], "Settings must be positive and every point needs an uncertainty.")
    fits = [f for law_id in LAWS if (f := fit_law(law_id, xa, ya, sa)) is not None]
    ranked = sorted(fits, key=lambda f: f.bic)
    passing = [f for f in ranked if max(abs(z) for z in f.loo_z) <= MAX_LOO_Z]
    if not passing:
        return ScalingVerdict("no_fit", None, [], ranked, "No law predicts every recording from the others.")
    best = passing[0]
    close = [f for f in passing if f.bic - best.bic < DECISIVE_DELTA_BIC]
    # Occam among equivalents: a simpler law that the leader reduces to takes its place.
    simpler = [f for f in close if f is not best and LAWS[f.law].n_params < LAWS[best.law].n_params and _same_law(best, f)]
    if simpler:
        best = min(simpler, key=lambda f: f.bic)
        close = [f for f in passing if f.bic - best.bic < DECISIVE_DELTA_BIC]
    # A free exponent that agrees with a fixed one is the same law with an extra parameter, not a rival.
    rivals = [f for f in close if f is not best and not _same_law(best, f)]
    if rivals:
        names = ", ".join(LAWS[f.law].title for f in [best, *rivals])
        return ScalingVerdict("ambiguous", None, [best, *rivals], ranked,
                              f"{names} all predict the left-out recordings; another setting is needed to separate them.")
    title = LAWS[best.law].title
    worst = max(abs(z) for z in best.loo_z)
    return ScalingVerdict("accepted", best, [best], ranked,
                          f"{title} predicts every recording from the others (worst miss {worst:.1f} sigma) and has the best BIC.")


# (richer law, simpler law) -> (index of the richer law's extra parameter, the value at which it reduces to the simpler)
NESTED: dict[tuple[str, str], tuple[int, float]] = {
    **{("power", lid): (1, float(lid.split("_", 1)[1])) for lid in LAWS if lid.startswith("power_")},
    ("power", "constant"): (1, 0.0),
    ("linear", "constant"): (1, 0.0),
    ("linear", "power_1"): (0, 0.0),
}


def _same_law(a: LawFit, b: LawFit) -> bool:
    """A richer law whose extra parameter is consistent with the value that reduces it to the other is that law."""
    for rich, simple in ((a, b), (b, a)):
        nest = NESTED.get((rich.law, simple.law))
        if nest is not None:
            i, value = nest
            return abs(rich.params[i] - value) <= 2 * math.sqrt(max(rich.cov[i, i], 0.0))
    return False


def next_setting(v: ScalingVerdict, x: list[float]) -> float | None:
    """For a tie: the setting, within half to twice the tested range, where the contenders' predictions differ most."""
    if v.status != "ambiguous" or len(v.contenders) < 2:
        return None
    lo, hi = min(x) / 2, max(x) * 2
    grid = np.geomspace(lo, hi, 200)
    a, b = v.contenders[:2]

    def separation(g: float) -> float:
        ya, sa = a.predict(g)
        yb, sb = b.predict(g)
        return abs(ya - yb) / math.hypot(sa, sb) if math.hypot(sa, sb) > 0 else 0.0
    return float(max(grid, key=separation))


def check_prediction(predicted: float, predicted_std: float, measured: float, measured_std: float) -> dict[str, float | bool]:
    z = (measured - predicted) / math.hypot(predicted_std, measured_std)
    return {"z": float(z), "within_95": bool(abs(z) <= Z95)}
