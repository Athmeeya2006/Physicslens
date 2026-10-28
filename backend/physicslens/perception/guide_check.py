"""Checks a picture guide with math before anyone sees it: the edit may only draw where it was asked to.

The image model proposes, this decides. Outside the (padded) mark boxes the edited frame must still be the user's
own scene, measured by structural similarity; inside them something must actually have been drawn.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels, right/bottom exclusive


@dataclass(frozen=True)
class GuideCheck:
    accepted: bool
    ssim_outside: float
    change_inside: float
    reason: str
    resized: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel SSIM of two grayscale images (Wang et al. 2004, 11x11 Gaussian window, sigma 1.5)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blur(x: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(x, (11, 11), 1.5)

    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a**2
    var_b = blur(b * b) - mu_b**2
    cov = blur(a * b) - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))


def box_mask(shape: tuple[int, int], boxes: list[Box], pad_fraction: float = 0.0) -> np.ndarray:
    h, w = shape
    px, py = round(pad_fraction * w), round(pad_fraction * h)
    mask = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        mask[max(y1 - py, 0):min(y2 + py, h), max(x1 - px, 0):min(x2 + px, w)] = True
    return mask


def match_size(original: np.ndarray, edited: np.ndarray) -> tuple[np.ndarray, bool]:
    """The image model picks its own output resolution, so the edit is resized back onto the original's pixels."""
    h, w = original.shape[:2]
    if edited.shape[:2] == (h, w):
        return edited, False
    return cv2.resize(edited, (w, h), interpolation=cv2.INTER_AREA), True


def check_edit(original: np.ndarray, edited: np.ndarray, boxes: list[Box], *, ssim_min: float,
               change_min: float, pad_fraction: float) -> tuple[GuideCheck, np.ndarray]:
    """original and edited are BGR uint8. Returns the verdict and the edit at the original's size."""
    edited, resized = match_size(original, edited)
    shape = original.shape[:2]
    inside = box_mask(shape, boxes)
    outside = ~box_mask(shape, boxes, pad_fraction)
    if not outside.any() or not inside.any():
        return GuideCheck(False, 0.0, 0.0, "The marks cover the whole frame, so the scene can't be checked.", resized), edited
    gray_o = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_e = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)
    ssim_out = float(np.mean(ssim_map(gray_o, gray_e)[outside]))
    # Color, not gray: a red ring on a gray wall can keep almost the same brightness.
    change_in = float(np.mean(np.abs(original.astype(np.int16) - edited.astype(np.int16))[inside]))
    if ssim_out < ssim_min:
        reason = (f"The picture changed the scene itself (similarity outside the marks {ssim_out:.2f}, "
                  f"needs {ssim_min:.2f}), so it is not shown.")
        return GuideCheck(False, ssim_out, change_in, reason, resized), edited
    if change_in < change_min:
        return GuideCheck(False, ssim_out, change_in, "Nothing was added where the marks should be.", resized), edited
    return GuideCheck(True, ssim_out, change_in, "Scene unchanged outside the marks, and the marks were drawn.", resized), edited
