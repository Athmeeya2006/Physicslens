"""Rotation angle from mask orientation: unwrap the long axis, check the object is measurable, build coordinates.

A mask's principal axis has no head or tail, so its angle is only known modulo pi. Unwrapping with period pi
works as long as the object turns less than a quarter turn between frames; beyond that the direction of
spin is ambiguous (like a wagon wheel in a film), so such clips are refused instead of misread.
"""

from __future__ import annotations

import numpy as np

from ..sigproc.cleanup import fill_short_gaps
from .coords import Coordinates

MIN_ELONGATION = 1.3  # long/short side of the mask; rounder objects have no measurable orientation
MAX_STEP_RAD = np.pi / 3  # per frame; the hard limit is pi/2, kept below it for a safety margin


class RotationError(ValueError):
    pass


def rotation_coordinates(frames: np.ndarray, angles: np.ndarray, centers_px: np.ndarray, long_px: np.ndarray,
                         short_px: np.ndarray, fps: float, px_per_m: float) -> tuple[Coordinates, np.ndarray]:
    """(coordinates, frames used). Positions in absolute meters (image origin), y up, so the ghost maps back simply."""
    elongation = float(np.median(long_px / np.maximum(short_px, 1e-6)))
    if elongation < MIN_ELONGATION:
        raise RotationError(f"The object looks almost round (long/short sides {elongation:.2f}), so its orientation "
                            "can't be measured. Put a strip of tape across it, or film an elongated object.")
    # Image rows point down; negate so counterclockwise on screen is positive, like the y-up positions elsewhere.
    phi = -np.unwrap(2 * np.asarray(angles, dtype=float)) / 2
    segments = fill_short_gaps(np.asarray(frames), np.column_stack([phi, centers_px]), max_gap=2)
    f, vals = max(segments, key=lambda s: len(s[0]))
    phi = vals[:, 0]
    steps = np.abs(np.diff(phi))
    if len(steps) and np.percentile(steps, 95) > MAX_STEP_RAD:
        raise RotationError(f"It spins too fast for this frame rate (up to {np.percentile(steps, 95):.2f} rad per frame); "
                            "the direction of spin becomes ambiguous. Film at a higher frame rate or a slower spin.")
    center = np.median(vals[:, 1:3], axis=0)
    arm_px = float(np.median(long_px)) / 2
    ctx = {"center_x": float(center[0] / px_per_m), "center_y": float(-center[1] / px_per_m),
           "arm_m": arm_px / px_per_m, "elongation": elongation}
    t = (f - f[0]) / fps
    return Coordinates("rotation", t, phi[:, None], ("phi",), ctx), f
