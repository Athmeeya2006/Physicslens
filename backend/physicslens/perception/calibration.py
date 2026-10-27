"""Pixels per meter from a reference object of known size. Sizes come from this table, never from a model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sam_track import ObjectTrack

# (long side, short side) in meters. Sources: ISO 216, ANSI, ISO/IEC 7810, mint specifications.
REFERENCE_SIZES: dict[str, tuple[float, float]] = {
    "a4_sheet": (0.297, 0.210),
    "us_letter_sheet": (0.2794, 0.2159),
    "credit_card": (0.08560, 0.05398),
    "inr_10_coin": (0.027, 0.027),
    "us_quarter": (0.02426, 0.02426),
    "euro_1_coin": (0.02325, 0.02325),
}
# No rulers: SAM segments the plastic, whose length varies by brand and isn't the printed scale.
# For those, the user types a known length instead (from_known_length).
MIN_FRAMES = 3


class CalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Calibration:
    px_per_m: float
    rel_std: float  # frame-to-frame spread of the estimate, relative
    source: str


def from_reference(track: ObjectTrack, kind: str) -> Calibration:
    if kind not in REFERENCE_SIZES:
        raise CalibrationError(f"Unknown reference object {kind!r}; known: {sorted(REFERENCE_SIZES)}.")
    if len(track.frames) < MIN_FRAMES:
        raise CalibrationError("The reference object was not visible in enough frames.")
    long_m, short_m = REFERENCE_SIZES[kind]
    estimates = track.long_px / long_m
    if short_m > 0:
        # Using both sides averages out slight perspective; a big disagreement means the sheet is tilted.
        aspect = np.median(track.long_px / track.short_px)
        if abs(aspect / (long_m / short_m) - 1) > 0.15:
            raise CalibrationError(
                f"The {kind.replace('_', ' ')} looks tilted or partly hidden (aspect {aspect:.2f}, expected "
                f"{long_m / short_m:.2f}). Put it flat in the plane of motion, facing the camera."
            )
        estimates = np.concatenate([estimates, track.short_px / short_m])
    ppm = float(np.median(estimates))
    spread = float(np.median(np.abs(estimates - ppm)) * 1.4826 / ppm)
    return Calibration(ppm, spread, kind)


def from_known_length(pixels: float, meters: float) -> Calibration:
    """User-supplied scale (e.g. 'the string is 60 cm')."""
    if pixels <= 0 or meters <= 0:
        raise CalibrationError("Known length must be positive.")
    return Calibration(pixels / meters, 0.0, "user")
