"""Decide from the fits alone whether one model explains the motion. The language model has no vote here."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from .coords import Coordinates, length_is_reliable
from .fit import FitResult
from .models import MODELS, MODELS_3D
from .rigor import bic_with, effective_n, rolling_origin

# A model must forecast the held-out final quarter within this fraction of the signal's spread,
# or within a few times the tracking noise (a correct model can't beat the noise floor), but never
# beyond the hard limit: on noise-like motion the "noise floor" is the motion itself.
MAX_HELDOUT_NRMSE = 0.15
MAX_HELDOUT_NOISE_RATIO = 3.0
HARD_MAX_HELDOUT_NRMSE = 0.3
# BIC difference needed to call a winner (>10 is "very strong" evidence on the Kass and Raftery scale).
DECISIVE_DELTA_BIC = 10.0
# An extra term must earn its place in the forecast too: a law that wins on BIC but forecasts the held-out motion
# clearly worse (by this factor and this much error) than a simpler passing law has not shown the term is real.
FORECAST_WORSE_REL = 1.2
FORECAST_WORSE_ABS = 0.01

Status = Literal["accepted", "ambiguous", "no_fit"]

# A model that is a limiting case of another law (not rival physics). When both fit equally well the exact law
# is kept: the data simply can't see the approximation, which is not a question worth a new experiment.
APPROXIMATES = {"pendulum_small_angle": ("pendulum", "the small-angle approximation of the same law")}


@dataclass(frozen=True)
class Verdict:
    status: Status
    winner: FitResult | None
    contenders: tuple[FitResult, ...]  # models still in the running (winner first), for experiment requests
    ranked: tuple[FitResult, ...]
    reason: str
    n_eff: float | None = None  # the effective number of independent samples BIC was computed with


def title(f: FitResult) -> str:
    return (MODELS.get(f.model_id) or MODELS_3D[f.model_id]).title


def _within(nrmse: float, rmse: float, noise: float) -> bool:
    if nrmse <= MAX_HELDOUT_NRMSE:
        return True
    return nrmse <= HARD_MAX_HELDOUT_NRMSE and rmse <= MAX_HELDOUT_NOISE_RATIO * noise


def rolling_median(f: FitResult) -> tuple[float, float] | None:
    """(median nrmse, median rmse) of the rolling-origin forecasts, or None when they were not run."""
    if not f.rolling:
        return None
    return (float(np.median([r["nrmse"] for r in f.rolling])), float(np.median([r["rmse"] for r in f.rolling])))


def forecasts_well(f: FitResult) -> bool:
    """The 75/25 forecast, and (when run) the median of the rolling-origin forecasts, by the same rule."""
    if not _within(f.heldout_nrmse, f.heldout_rmse, f.noise_sigma):
        return False
    med = rolling_median(f)
    return med is None or _within(med[0], med[1], f.noise_sigma)


SAME_LAW_REL = 1e-4  # simulated paths closer than this (relative to the motion's size) are one law, written twice


def distinct(fits: list[FitResult]) -> list[FitResult]:
    """Drop fits that are the same law as an earlier one: same parameter count, practically the same simulation.

    Independent proposals often write one law in different notation (GM*x/(x^2+y^2)^1.5 and k*x/hypot(x,y)^3); counted
    separately, they would "tie" with themselves and turn a clear answer into a fake ambiguity.
    """
    kept: list[FitResult] = []
    for f in fits:
        if f.success and f.sim is not None and any(
                g.k == f.k and g.sim is not None and g.sim.shape == f.sim.shape
                and float(np.max(np.abs(g.sim - f.sim))) <= SAME_LAW_REL * (float(np.ptp(g.sim)) + 1e-12)
                for g in kept):
            continue
        kept.append(f)
    return kept


AT_LIMIT = 1e-3  # a fitted value within this share of its range from a limit is "at" the limit


def at_limit(f: FitResult) -> str | None:
    """Which fitted parameter sits at a limit of its allowed range, if any. Such a law only fits by leaning on the
    limit (g pinned at its lowest allowed value while drag takes its highest), so its numbers mean nothing. A limit
    of zero is exempt: "no damping" or "no drag" is a real answer, not a pinned one."""
    spec = _spec(f.model_id)
    if spec is None:
        return None
    for p in spec.params:
        v = f.params.get(p.name)
        if v is None:
            continue
        span = p.upper - p.lower
        if p.lower != 0 and abs(v - p.lower) <= AT_LIMIT * span:
            return f"{p.name} = {v:.4g} sits at its lowest allowed value"
        if p.upper != 0 and abs(v - p.upper) <= AT_LIMIT * span:
            return f"{p.name} = {v:.4g} sits at its highest allowed value"
    return None


def decide(fits: list[FitResult]) -> Verdict:
    fits = distinct(fits)
    ranked = tuple(sorted((f for f in fits if f.success), key=lambda f: f.bic))
    passing = [f for f in ranked if forecasts_well(f) and at_limit(f) is None]
    if not passing:
        best = ranked[0] if ranked else None
        why = f"best forecast error {best.heldout_nrmse:.0%} ({title(best)})" if best else "every simulation failed"
        return Verdict("no_fit", None, (), ranked, f"No candidate predicts the held-out motion: {why}.")

    best = passing[0]
    # Occam: a model with more parameters that still scores worse adds nothing, so it is not a real contender.
    close = tuple(f for f in passing[1:] if f.bic - best.bic < DECISIVE_DELTA_BIC and f.k <= best.k)
    group = (best, *close)
    ids = {f.model_id for f in group}
    folded = [f for f in group if APPROXIMATES.get(f.model_id, ("",))[0] in ids]
    group = tuple(f for f in group if f not in folded)
    note = "".join(f"; {title(f)} is indistinguishable from it at this amplitude" for f in folded)
    if len(group) > 1:
        names = [title(f) for f in group]
        listed = ", ".join(names[:-1]) + " and " + names[-1]
        return Verdict("ambiguous", None, group, ranked,
                       f"{listed} explain the data about equally well (BIC within {DECISIVE_DELTA_BIC:.0f}){note}.")
    best = group[0]
    passing_ids = {f.model_id for f in passing}
    simpler = next((f for f in passing if f.k < best.k and f not in folded
                    and APPROXIMATES.get(f.model_id, ("",))[0] not in passing_ids  # the exact law, not its approximation
                    and best.heldout_nrmse > max(FORECAST_WORSE_REL * f.heldout_nrmse, f.heldout_nrmse + FORECAST_WORSE_ABS)), None)
    if simpler is not None:
        return Verdict("ambiguous", None, (best, simpler), ranked,
                       f"{title(best)} fits the whole clip better ({simpler.bic - best.bic:.0f} BIC) but forecasts the held-out "
                       f"motion worse than the simpler {title(simpler)} ({best.heldout_nrmse:.1%} against "
                       f"{simpler.heldout_nrmse:.1%}): its extra term is not confirmed{note}.")
    runner = next((f for f in passing if f is not best and f not in folded), None)
    margin = f"; next best ({title(runner)}) is {runner.bic - best.bic:.0f} BIC worse" if runner else ""
    return Verdict("accepted", best, (best,), ranked,
                   f"{title(best)} forecasts held-out data within {best.heldout_nrmse:.1%}{margin}{note}.")


# Live mode refits every second and skips the rolling-origin forecasts for speed; recorded clips always run them.
ROLLING_ORIGIN = True


def _spec(model_id: str):
    return MODELS.get(model_id) or MODELS_3D.get(model_id)


def strengthen(c: Coordinates, fits: list[FitResult], rolling: bool = ROLLING_ORIGIN) -> tuple[list[FitResult], float | None]:
    """BIC with the effective sample size (the same n_eff for every law, so ranks stay fair), and rolling-origin
    forecasts for each law that passes the 75/25 forecast."""
    done = [f for f in fits if f.success and f.sim is not None and f.sim.shape == c.q.shape]
    if not done:
        return fits, None
    best = min(done, key=lambda f: f.bic if f.bic_raw is None else f.bic_raw)
    n_eff, _ = effective_n(best.sim - c.q)
    out = [replace(f, bic_raw=f.bic if f.bic_raw is None else f.bic_raw, bic=bic_with(f, n_eff)) if f.success else f
           for f in fits]  # every law with this one n_eff, also those already adjusted in an earlier round
    if not rolling:
        return out, n_eff
    # Rolling forecasts in BIC order, only where they can change the verdict: once one law passes them, a law more
    # than DECISIVE_DELTA_BIC behind it can be neither the winner nor a contender.
    order = sorted((i for i, f in enumerate(out) if f.success and _within(f.heldout_nrmse, f.heldout_rmse, f.noise_sigma)),
                   key=lambda i: out[i].bic)
    leader: float | None = None
    for i in order:
        f = out[i]
        if leader is not None and f.bic - leader >= DECISIVE_DELTA_BIC:
            break
        spec = _spec(f.model_id)
        if spec is not None and not f.rolling:
            out[i] = f = replace(f, rolling=rolling_origin(spec, c, f))
        if leader is None and forecasts_well(f):
            leader = f.bic
    return out, n_eff


def assess(c: Coordinates, fits: list[FitResult], rolling: bool = ROLLING_ORIGIN) -> Verdict:
    """decide(), on fits made stricter by strengthen(), plus checks that depend on the measurement geometry."""
    fits, n_eff = strengthen(c, fits, rolling)
    v = replace(decide(fits), n_eff=n_eff)
    if c.family == "pendulum" and v.status == "accepted" and not length_is_reliable(c):
        # The dynamics only pin down g/L; with a flat arc L (and so g) would be invented.
        return Verdict("ambiguous", None, v.contenders, v.ranked,
                       "The swing's arc is too flat to measure the pendulum length, so g cannot be determined.", v.n_eff)
    return v


def judge_candidates(v: Verdict) -> dict[str, tuple[str, str]]:
    """Per-model outcome and the concrete reason, for the agent's decision trail on screen."""
    out: dict[str, tuple[str, str]] = {}
    best = v.contenders[0] if v.contenders else None
    contenders = {f.model_id for f in v.contenders}
    for f in v.ranked:
        if v.winner is not None and f.model_id == v.winner.model_id:
            med = rolling_median(f)
            rolling = (f"; from {len(f.rolling)} later starting points, median error {med[0]:.1%}" if med else "")
            why = f"forecasts held-out motion within {f.heldout_nrmse:.1%} and has the best BIC{rolling}"
            out[f.model_id] = ("accepted", why)
        elif f.model_id in contenders and len(v.contenders) == 2 and "not confirmed" in v.reason:
            other = next(g for g in v.contenders if g.model_id != f.model_id)
            out[f.model_id] = ("tied", f"fits the whole clip better than {title(other)} but forecasts the held-out motion worse "
                                       f"({f.heldout_nrmse:.1%} against {other.heldout_nrmse:.1%}): the extra term is not confirmed"
                               if f.k > other.k else
                               f"forecasts the held-out motion better ({f.heldout_nrmse:.1%}) with fewer terms than {title(other)}")
        elif f.model_id in contenders:
            out[f.model_id] = ("tied", "fits about as well as the best; the data can't separate them")
        elif APPROXIMATES.get(f.model_id, ("",))[0] in contenders and forecasts_well(f):
            _, what = APPROXIMATES[f.model_id]
            out[f.model_id] = ("approximation", f"{what}: indistinguishable at this amplitude, so the exact law is kept")
        elif not _within(f.heldout_nrmse, f.heldout_rmse, f.noise_sigma):
            out[f.model_id] = ("rejected", f"forecast error {f.heldout_nrmse:.0%} on held-out data")
        elif (why := at_limit(f)) is not None:
            out[f.model_id] = ("rejected", f"{why}: the law only fits by leaning on a limit, so its numbers mean nothing")
        elif not forecasts_well(f):
            med = rolling_median(f)
            why = (f"forecasts the last quarter, but not from later starting points (median error "
                   f"{med[0] if med else math.nan:.0%} over {len(f.rolling)} rolling forecasts)")
            out[f.model_id] = ("rejected", why)
        elif best is not None and f.k > best.k:
            out[f.model_id] = ("rejected", f"extra parameters don't improve on {title(best)}")
        elif best is not None:
            out[f.model_id] = ("rejected", f"{f.bic - best.bic:.0f} BIC worse than {title(best)}")
        else:
            out[f.model_id] = ("rejected", "does not explain the motion")
        if f.notes:
            status, reason = out[f.model_id]
            out[f.model_id] = (status, "; ".join([reason, *f.notes]))
    return out


# The agent may leave out a throw or a catch, but never most of the clip: that would let it pick the part that fits.
MIN_WINDOW_FRACTION = 0.7


def window_indices(t: np.ndarray, start_s: float, end_s: float, min_fraction: float = MIN_WINDOW_FRACTION,
                   min_samples: int = 30) -> tuple[int, int] | str:
    """Sample range for an agent's fit window, or the reason it is refused. Looking needs less than fitting."""
    duration = float(t[-1] - t[0])
    start, end = max(start_s, 0.0) + t[0], min(end_s, duration) + t[0]
    if end <= start:
        return "it ends before it starts"
    if (end - start) < min_fraction * duration:
        return f"it keeps {(end - start) / duration:.0%} of the clip, below the {min_fraction:.0%} minimum"
    i, j = int(np.searchsorted(t, start)), int(np.searchsorted(t, end, side="right"))
    return (i, j) if j - i >= min_samples else "it leaves too few samples"
