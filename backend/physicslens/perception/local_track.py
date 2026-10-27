"""Instant on-device tracking between SAM results: normalized template matching in a search window.

SAM is precise but returns seconds late; this keeps the live overlay and fits current. It handles the
mostly-translating rigid objects PhysicsLens measures (bobs, balls, weights); SAM corrects any drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_SCORE = 0.5  # below this the match is not trusted and the frame is skipped
SEARCH_FACTOR = 3.0  # search window is this many object sizes around the last position


@dataclass
class Match:
    x: float  # center, full-frame pixels
    y: float
    score: float


class TemplateTracker:
    def __init__(self, gray: np.ndarray, box: tuple[int, int, int, int]):
        self.reset(gray, box)

    def reset(self, gray: np.ndarray, box: tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = _clip_box(box, gray.shape)
        self.template = gray[y1:y2, x1:x2].copy()
        self.w, self.h = x2 - x1, y2 - y1
        self.cx, self.cy = (x1 + x2) / 2, (y1 + y2) / 2

    def update(self, gray: np.ndarray) -> Match | None:
        if self.w < 4 or self.h < 4:
            return None
        rx, ry = SEARCH_FACTOR * self.w, SEARCH_FACTOR * self.h
        H, W = gray.shape
        x1, y1 = int(max(0, self.cx - rx)), int(max(0, self.cy - ry))
        x2, y2 = int(min(W, self.cx + rx)), int(min(H, self.cy + ry))
        if x2 - x1 <= self.w or y2 - y1 <= self.h:
            return None
        scores = cv2.matchTemplate(gray[y1:y2, x1:x2], self.template, cv2.TM_CCOEFF_NORMED)
        _, best, _, (mx, my) = cv2.minMaxLoc(scores)
        if not np.isfinite(best) or best < MIN_SCORE:
            return None
        self.cx, self.cy = x1 + mx + self.w / 2, y1 + my + self.h / 2
        return Match(self.cx, self.cy, float(best))


def _clip_box(box: tuple[int, int, int, int], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    H, W = shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, min(int(x1), W - 1)), max(0, min(int(y1), H - 1))
    return x1, y1, max(x1 + 1, min(int(x2), W)), max(y1 + 1, min(int(y2), H))


def to_gray(jpeg: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Frame is not a JPEG.")
    return img


def box_around_tap(gray: np.ndarray, x: float, y: float, tolerance: int = 18) -> tuple[int, int, int, int]:
    """Estimate the tapped object's box by growing the region of similar brightness around the tap.

    Falls back to a fixed-size box when the region leaks into the background (low contrast) or is tiny.
    """
    H, W = gray.shape
    xi, yi = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = np.zeros((H + 2, W + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    cv2.floodFill(blurred, mask, (xi, yi), 0, (tolerance,), (tolerance,), flags)
    ys, xs = np.nonzero(mask[1:-1, 1:-1])
    default = max(16, int(0.08 * min(H, W)))
    if len(xs) < 30:
        side = default
    else:
        side = max(int(np.ptp(xs)), int(np.ptp(ys))) + 1
        if side > 0.4 * min(H, W):  # leaked into the background
            side = default
    half = side / 2
    return int(xi - half), int(yi - half), int(xi + half) + 1, int(yi + half) + 1
