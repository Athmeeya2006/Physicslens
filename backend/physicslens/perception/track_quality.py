"""Is a SAM track plausibly one solid object moving smoothly? Pure numbers, no model calls.

A box placed slightly off (on a clasp, a shadow, a digit drawn on the video) makes SAM follow the wrong thing;
the signs are teleporting jumps, a mask whose size keeps changing, frames lost, or no motion at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .sam_track import ObjectTrack

MAX_JUMP_RATIO = 8.0  # largest frame-to-frame step against the typical step
MAX_SIZE_SPREAD = 0.35  # mask size variation (robust coefficient of variation)
MIN_COVERAGE = 0.85


@dataclass(frozen=True)
class TrackQuality:
    coverage: float
    jump_ratio: float
    size_spread: float
    travel_px: float
    size_px: float

    @property
    def problems(self) -> list[str]:
        out = []
        if self.coverage < MIN_COVERAGE:
            out.append(f"lost in {1 - self.coverage:.0%} of the frames")
        if self.jump_ratio > MAX_JUMP_RATIO:
            out.append(f"jumps {self.jump_ratio:.0f} times further than its typical step")
        if self.size_spread > MAX_SIZE_SPREAD:
            out.append(f"its outline changes size by {self.size_spread:.0%}")
        if self.travel_px < 0.25 * self.size_px:
            out.append("it barely moves")
        return out

    def score(self) -> float:
        """Higher is better; used to pick between a first track and a repaired one."""
        return self.coverage - 0.05 * min(self.jump_ratio, 40) - self.size_spread + (0.5 if self.travel_px >= 0.25 * self.size_px else 0)


def assess_track(track: ObjectTrack, total_frames: int) -> TrackQuality:
    steps = np.linalg.norm(np.diff(track.centers_px, axis=0), axis=1) if len(track.frames) > 1 else np.zeros(1)
    consecutive = np.diff(track.frames) == 1 if len(track.frames) > 1 else np.zeros(1, bool)
    s = steps[consecutive] if consecutive.any() else steps
    typical = max(float(np.median(s)), 0.5)  # half a pixel: below that the object is effectively still
    size = np.asarray(track.long_px, float)
    q75, q25 = np.percentile(size, [75, 25])
    return TrackQuality(
        coverage=len(track.frames) / max(total_frames, 1),
        jump_ratio=float(np.percentile(s, 99)) / typical,
        size_spread=float((q75 - q25) / max(np.median(size), 1e-9)),
        travel_px=float(np.ptp(track.centers_px, axis=0).max()),
        size_px=float(np.median(size)),
    )


def overlay_frames(clip_path: str, track: ObjectTrack, long_side: int = 640) -> bytes:
    """Start, middle and end frames side by side, each with the tracked region boxed: what the reasoning model checks."""
    cap = cv2.VideoCapture(clip_path)
    picks = [int(track.frames[0]), int(track.frames[len(track.frames) // 2]), int(track.frames[-1])]
    frames: dict[int, np.ndarray] = {}
    try:
        # Read straight through: seeking in an MP4 can land a few frames off, and the box would sit beside the object.
        idx = 0
        while idx <= picks[-1]:
            ok, img = cap.read()
            if not ok:
                break
            if idx in picks:
                frames[idx] = img
            idx += 1
    finally:
        cap.release()
    tiles = []
    for f in picks:
        img = frames.get(f)
        if img is not None:
            i = int(np.searchsorted(track.frames, f))
            if track.boxes is not None:
                x1, y1, x2, y2 = (int(v) for v in track.boxes[i])
            else:
                (cx, cy), half = track.centers_px[i], float(track.long_px[i]) / 2
                x1, y1, x2, y2 = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
            cv2.rectangle(img, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 255, 0), max(2, img.shape[1] // 300), cv2.LINE_AA)
            tiles.append(img)
    strip = cv2.hconcat(tiles) if tiles else np.zeros((10, 10, 3), np.uint8)
    s = long_side * len(tiles) / max(strip.shape[1], 1)
    strip = cv2.resize(strip, (round(strip.shape[1] * s), round(strip.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()
