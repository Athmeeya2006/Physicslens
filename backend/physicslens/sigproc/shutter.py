"""Rolling shutter: a phone reads its sensor row by row, so a fast object's top and bottom are captured at slightly
different times and the object is sheared. The skew across the object is about its horizontal speed times the time
the readout takes to cross it: speed (px/s) x readout (s) x object height / frame height. A typical phone reads a
whole frame in 20 to 35 ms; READOUT_S is the middle of that range, an assumption stated with every warning."""

from __future__ import annotations

import numpy as np

READOUT_S = 0.03
WARN_PX = 1.0


def skew_px(xy_px: np.ndarray, fps: float, object_px: float, frame_height_px: int, readout_s: float = READOUT_S) -> float:
    """The largest shear across the object along the track, in pixels."""
    if len(xy_px) < 2 or fps <= 0 or frame_height_px <= 0:
        return 0.0
    vx = np.abs(np.diff(np.asarray(xy_px, float)[:, 0])) * fps
    return float(np.percentile(vx, 95) * readout_s * object_px / frame_height_px)


def note(skew: float) -> str | None:
    if skew <= WARN_PX:
        return None
    return (f"The object moves fast enough for the phone's rolling shutter to shear it by about {skew:.1f} px "
            f"(assuming a {READOUT_S * 1000:.0f} ms readout): its tracked centre can be off by up to half that. A higher "
            "frame rate or a slower motion avoids it.")
