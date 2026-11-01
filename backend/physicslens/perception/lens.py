"""Lens calibration for one phone: film a printed checkerboard, get the focal length and the lens distortion.

OpenCV's calibrateCamera on the board's inner corners, found in frames sampled from a short video of the board seen
from several angles. The result is used two ways: the track points (never whole frames: cheap, and SAM still sees the
original video) are undistorted before fitting, and the focal length feeds the 3D mode instead of the A4-sheet method.

A calibration belongs to one camera at one video setting. It is stored by resolution and applied to a clip of the same
shape scaled to the clip's size; a clip of another aspect ratio (a different crop of the sensor) is left as it is.
Pure local math: no model calls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PATTERN = (9, 6)  # inner corners (columns, rows) of the printable board: 10 x 7 squares
SQUARE_MM = 25.0  # on A4 at 100% scale
MIN_VIEWS = 8
MAX_VIEWS = 25  # calibrateCamera grows slowly with views; past this, more frames add time, not accuracy
MAX_RMS_PX = 1.5  # reprojection error above this: blurred frames or a board that was not flat


class LensError(ValueError):
    pass


@dataclass(frozen=True)
class Lens:
    width: int  # the video size the calibration was made at
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, ...]  # k1, k2, p1, p2, k3
    rms_px: float = 0.0
    views: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Lens:
        try:
            lens = cls(width=int(d["width"]), height=int(d["height"]), fx=float(d["fx"]), fy=float(d["fy"]),
                       cx=float(d["cx"]), cy=float(d["cy"]), dist=tuple(float(v) for v in d["dist"]),
                       rms_px=float(d.get("rms_px", 0.0)), views=int(d.get("views", 0)))
        except (KeyError, TypeError, ValueError) as e:
            raise LensError(f"the lens calibration is incomplete ({e})") from e
        values = (lens.fx, lens.fy, lens.cx, lens.cy, *lens.dist)
        if not (lens.width > 0 and lens.height > 0 and len(lens.dist) in (4, 5) and all(np.isfinite(values))
                and lens.fx > 0 and lens.fy > 0 and all(abs(k) < 5 for k in lens.dist)):
            raise LensError("the lens calibration has values no phone camera has")
        return lens

    def matrices(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Camera matrix and distortion at another resolution of the same shape, or None for a different crop."""
        if abs(width / height - self.width / self.height) > 0.01 * self.width / self.height:
            return None
        s = width / self.width
        k = np.array([[self.fx * s, 0, self.cx * s], [0, self.fy * s, self.cy * s], [0, 0, 1.0]])
        return k, np.asarray(self.dist, dtype=float)


def checkerboard_svg(pattern: tuple[int, int] = PATTERN, square_mm: float = SQUARE_MM) -> str:
    """A printable board (A4 landscape, 100% scale): pattern inner corners means one more square each way."""
    cols, rows = pattern[0] + 1, pattern[1] + 1
    w, h = cols * square_mm, rows * square_mm
    squares = "".join(f'<rect x="{c * square_mm:g}" y="{r * square_mm:g}" width="{square_mm:g}" height="{square_mm:g}"/>'
                      for r in range(rows) for c in range(cols) if (r + c) % 2 == 0)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="297mm" height="210mm" viewBox="0 0 297 210">'
            f'<rect width="297" height="210" fill="#fff"/>'
            f'<g transform="translate({(297 - w) / 2:g} {(210 - h) / 2:g})" fill="#000">{squares}</g>'
            f'<text x="148.5" y="204" font-family="sans-serif" font-size="4" text-anchor="middle" fill="#555">'
            f'PhysicsLens lens calibration: print at 100% scale; each square {square_mm:g} mm. Tape it flat.</text></svg>')


def find_corners(gray: np.ndarray, pattern: tuple[int, int] = PATTERN) -> np.ndarray | None:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    return cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit)


def calibrate_views(corners: list[np.ndarray], size: tuple[int, int], pattern: tuple[int, int] = PATTERN,
                    square_m: float = SQUARE_MM / 1000) -> Lens:
    """calibrateCamera on corner sets already found (one per view); size is (width, height)."""
    if len(corners) < MIN_VIEWS:
        raise LensError(f"the board was found in {len(corners)} frames; at least {MIN_VIEWS} are needed. Film it "
                        "slowly from several angles, the whole board in view and in focus.")
    if len(corners) > MAX_VIEWS:
        corners = [corners[i] for i in np.linspace(0, len(corners) - 1, MAX_VIEWS).round().astype(int)]
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_m
    rms, k, dist, _, _ = cv2.calibrateCamera([obj] * len(corners), corners, size, None, None)
    if not np.isfinite(rms) or rms > MAX_RMS_PX:
        raise LensError(f"the corners do not fit one lens well (reprojection error {rms:.2f} px): the board may be "
                        "bent, or frames blurred. Tape it flat and move the phone slowly.")
    return Lens(width=size[0], height=size[1], fx=float(k[0, 0]), fy=float(k[1, 1]), cx=float(k[0, 2]),
                cy=float(k[1, 2]), dist=tuple(float(v) for v in dist.ravel()[:5]), rms_px=float(rms), views=len(corners))


def calibrate_video(video: Path, pattern: tuple[int, int] = PATTERN, square_m: float = SQUARE_MM / 1000,
                    samples: int = 60) -> Lens:
    """Sample frames from a video of the board, keep those where it is found, and calibrate."""
    cap = cv2.VideoCapture(str(video))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise LensError("the calibration video could not be read")
        found, size = [], None
        for i in np.linspace(0, total - 1, min(samples, total)).round().astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = (gray.shape[1], gray.shape[0])
            c = find_corners(gray, pattern)
            if c is not None:
                found.append(c)
    finally:
        cap.release()
    if size is None:
        raise LensError("the calibration video had no readable frames")
    return calibrate_views(found, size, pattern, square_m)


def undistort_px(points: np.ndarray, lens: Lens, width: int, height: int) -> np.ndarray | None:
    """Pixel points (N, 2) as a perfect pinhole camera would have seen them, same camera matrix; None if not applicable."""
    m = lens.matrices(width, height)
    if m is None:
        return None
    k, dist = m
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, k, dist, P=k).reshape(-1, 2)


def undistort_track(track: Any, lens: Lens, width: int, height: int) -> Any:
    """An ObjectTrack with its centres undistorted (sizes are local and barely change); unchanged if not applicable."""
    if len(track.centers_px) == 0:
        return track
    fixed = undistort_px(track.centers_px, lens, width, height)
    return track if fixed is None else replace(track, centers_px=fixed)


def focal_px_at(lens: Lens, width: int, height: int) -> float | None:
    m = lens.matrices(width, height)
    return None if m is None else float((m[0][0, 0] + m[0][1, 1]) / 2)


__all__ = ["PATTERN", "SQUARE_MM", "Lens", "LensError", "calibrate_video", "calibrate_views", "checkerboard_svg",
           "find_corners", "focal_px_at", "undistort_px", "undistort_track"]
