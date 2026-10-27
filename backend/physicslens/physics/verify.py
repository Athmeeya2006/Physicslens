"""Decide from the fits alone whether one model explains the motion. The language model has no vote here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .coords import Coordinates, length_is_reliable
from .fit import FitResult
from .models import MODELS

# A model must forecast the held-out final quarter within this fraction of the signal's spread,
# or within a few times the tracking noise (a correct model can't beat the noise floor), but never
# beyond the hard limit: on noise-like motion the "noise floor" is the motion itself.
MAX_HELDOUT_NRMSE = 0.15
MAX_HELDOUT_NOISE_RATIO = 3.0
HARD_MAX_HELDOUT_NRMSE = 0.3
# BIC difference needed to call a winner (>10 is "very strong" evidence on the Kass and Raftery scale).
DECISIVE_DELTA_BIC = 10.0

Status = Literal["accepted", "ambiguous", "no_fit"]


@dataclass(frozen=True)
class Verdict:
    status: Status
    winner: FitResult | None
    contenders: tuple[FitResult, ...]  # models still in the running (winner first), for experiment requests
    ranked: tuple[FitResult, ...]
    reason: str


def title(f: FitResult) -> str:
    return MODELS[f.model_id].title


def forecasts_well(f: FitResult) -> bool:
    if f.heldout_nrmse <= MAX_HELDOUT_NRMSE:
        return True
    return f.heldout_nrmse <= HARD_MAX_HELDOUT_NRMSE and f.heldout_rmse <= MAX_HELDOUT_NOISE_RATIO * f.noise_sigma


def decide(fits: list[FitResult]) -> Verdict:
    ranked = tuple(sorted((f for f in fits if f.success), key=lambda f: f.bic))
    passing = [f for f in ranked if forecasts_well(f)]
    if not passing:
        best = ranked[0] if ranked else None
        why = f"best forecast error {best.heldout_nrmse:.0%} ({title(best)})" if best else "every simulation failed"
        return Verdict("no_fit", None, (), ranked, f"No candidate predicts the held-out motion: {why}.")

    best = passing[0]
    # Occam: a model with more parameters that still scores worse adds nothing, so it is not a real contender.
    close = tuple(f for f in passing[1:] if f.bic - best.bic < DECISIVE_DELTA_BIC and f.k <= best.k)
    if close:
        names = [title(f) for f in (best, *close)]
        listed = ", ".join(names[:-1]) + " and " + names[-1]
        return Verdict("ambiguous", None, (best, *close), ranked,
                       f"{listed} explain the data about equally well (BIC within {DECISIVE_DELTA_BIC:.0f}).")
    runner = passing[1] if len(passing) > 1 else None
    margin = f"; next best ({title(runner)}) is {runner.bic - best.bic:.0f} BIC worse" if runner else ""
    return Verdict("accepted", best, (best,), ranked,
                   f"{title(best)} forecasts held-out data within {best.heldout_nrmse:.1%}{margin}.")


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
        elif not forecasts_well(f):
            out[f.model_id] = ("rejected", f"forecast error {f.heldout_nrmse:.0%} on held-out data")
        elif best is not None and f.k > best.k:
            out[f.model_id] = ("rejected", f"extra parameters don't improve on {title(best)}")
        elif best is not None:
            out[f.model_id] = ("rejected", f"{f.bic - best.bic:.0f} BIC worse than {title(best)}")
        else:
            out[f.model_id] = ("rejected", "does not explain the motion")
    return out
