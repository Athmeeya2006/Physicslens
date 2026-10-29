"""Decide from the fits alone whether one model explains the motion. The language model has no vote here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .coords import Coordinates, length_is_reliable
from .fit import FitResult
from .models import MODELS, MODELS_3D

# A model must forecast the held-out final quarter within this fraction of the signal's spread,
# or within a few times the tracking noise (a correct model can't beat the noise floor), but never
# beyond the hard limit: on noise-like motion the "noise floor" is the motion itself.
MAX_HELDOUT_NRMSE = 0.15
MAX_HELDOUT_NOISE_RATIO = 3.0
HARD_MAX_HELDOUT_NRMSE = 0.3
# BIC difference needed to call a winner (>10 is "very strong" evidence on the Kass and Raftery scale).
DECISIVE_DELTA_BIC = 10.0

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


def title(f: FitResult) -> str:
    return (MODELS.get(f.model_id) or MODELS_3D[f.model_id]).title


def forecasts_well(f: FitResult) -> bool:
    if f.heldout_nrmse <= MAX_HELDOUT_NRMSE:
        return True
    return f.heldout_nrmse <= HARD_MAX_HELDOUT_NRMSE and f.heldout_rmse <= MAX_HELDOUT_NOISE_RATIO * f.noise_sigma


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


def decide(fits: list[FitResult]) -> Verdict:
    fits = distinct(fits)
    ranked = tuple(sorted((f for f in fits if f.success), key=lambda f: f.bic))
    passing = [f for f in ranked if forecasts_well(f)]
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
    runner = next((f for f in passing if f is not best and f not in folded), None)
    margin = f"; next best ({title(runner)}) is {runner.bic - best.bic:.0f} BIC worse" if runner else ""
    return Verdict("accepted", best, (best,), ranked,
                   f"{title(best)} forecasts held-out data within {best.heldout_nrmse:.1%}{margin}{note}.")


def assess(c: Coordinates, fits: list[FitResult]) -> Verdict:
    """decide(), plus checks that depend on the measurement geometry rather than the fit."""
    v = decide(fits)
    if c.family == "pendulum" and v.status == "accepted" and not length_is_reliable(c):
        # The dynamics only pin down g/L; with a flat arc L (and so g) would be invented.
        return Verdict("ambiguous", None, v.contenders, v.ranked,
                       "The swing's arc is too flat to measure the pendulum length, so g cannot be determined.")
    return v


def judge_candidates(v: Verdict) -> dict[str, tuple[str, str]]:
    """Per-model outcome and the concrete reason, for the agent's decision trail on screen."""
    out: dict[str, tuple[str, str]] = {}
    best = v.contenders[0] if v.contenders else None
    contenders = {f.model_id for f in v.contenders}
    for f in v.ranked:
        if v.winner is not None and f.model_id == v.winner.model_id:
            out[f.model_id] = ("accepted", f"forecasts held-out motion within {f.heldout_nrmse:.1%} and has the best BIC")
        elif f.model_id in contenders:
            out[f.model_id] = ("tied", "fits about as well as the best; the data can't separate them")
        elif APPROXIMATES.get(f.model_id, ("",))[0] in contenders and forecasts_well(f):
            _, what = APPROXIMATES[f.model_id]
            out[f.model_id] = ("approximation", f"{what}: indistinguishable at this amplitude, so the exact law is kept")
        elif not forecasts_well(f):
            out[f.model_id] = ("rejected", f"forecast error {f.heldout_nrmse:.0%} on held-out data")
        elif best is not None and f.k > best.k:
            out[f.model_id] = ("rejected", f"extra parameters don't improve on {title(best)}")
        elif best is not None:
            out[f.model_id] = ("rejected", f"{f.bic - best.bic:.0f} BIC worse than {title(best)}")
        else:
            out[f.model_id] = ("rejected", "does not explain the motion")
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
