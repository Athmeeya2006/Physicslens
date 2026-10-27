"""Decide from the fits alone whether one model explains the motion. The language model has no vote here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .coords import Coordinates, length_is_reliable
from .fit import FitResult

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


def forecasts_well(f: FitResult) -> bool:
    if f.heldout_nrmse <= MAX_HELDOUT_NRMSE:
        return True
    return f.heldout_nrmse <= HARD_MAX_HELDOUT_NRMSE and f.heldout_rmse <= MAX_HELDOUT_NOISE_RATIO * f.noise_sigma


def decide(fits: list[FitResult]) -> Verdict:
    ranked = tuple(sorted((f for f in fits if f.success), key=lambda f: f.bic))
    passing = [f for f in ranked if forecasts_well(f)]
    if not passing:
        best = ranked[0] if ranked else None
        why = f"best forecast error {best.heldout_nrmse:.0%} ({best.model_id})" if best else "every simulation failed"
        return Verdict("no_fit", None, (), ranked, f"No candidate predicts the held-out motion: {why}.")

    best = passing[0]
    # Occam: a model with more parameters that still scores worse adds nothing, so it is not a real contender.
    close = tuple(f for f in passing[1:] if f.bic - best.bic < DECISIVE_DELTA_BIC and f.k <= best.k)
    if close:
        names = ", ".join(f.model_id for f in close)
        return Verdict("ambiguous", None, (best, *close), ranked,
                       f"{best.model_id} and {names} explain the data about equally well (BIC within {DECISIVE_DELTA_BIC:.0f}).")
    runner = passing[1] if len(passing) > 1 else None
    margin = f"; next best {runner.model_id} is {runner.bic - best.bic:.0f} BIC worse" if runner else ""
    return Verdict("accepted", best, (best,), ranked,
                   f"{best.model_id} forecasts held-out data within {best.heldout_nrmse:.1%}{margin}.")


def assess(c: Coordinates, fits: list[FitResult]) -> Verdict:
    """decide(), plus checks that depend on the measurement geometry rather than the fit."""
    v = decide(fits)
    if c.family == "pendulum" and v.status == "accepted" and not length_is_reliable(c):
        # The dynamics only pin down g/L; with a flat arc L (and so g) would be invented.
        return Verdict("ambiguous", None, v.contenders, v.ranked,
                       "The swing's arc is too flat to measure the pendulum length, so g cannot be determined.")
    return v
