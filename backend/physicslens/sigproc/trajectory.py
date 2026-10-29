"""Measured trajectories in SI units. Pure data, no I/O."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    """Object positions over time. x to the right, y UP (image rows point down), meters, seconds.

    `frames` are source-video frame indices, kept so results can be drawn back onto the video.
    """

    t: np.ndarray
    xy: np.ndarray
    frames: np.ndarray
    fps: float

    def __post_init__(self) -> None:
        if self.xy.ndim != 2 or self.xy.shape[1] != 2 or len(self.t) != len(self.xy) or len(self.frames) != len(self.t):
            raise ValueError("Trajectory needs t (N,), xy (N, 2) and frames (N,) of equal length.")

    def __len__(self) -> int:
        return len(self.t)

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    def slice(self, start: int, stop: int) -> Trajectory:
        return Trajectory(self.t[start:stop], self.xy[start:stop], self.frames[start:stop], self.fps)


def from_pixels(frames: np.ndarray, centers_px: np.ndarray, fps: float, px_per_m: float,
                origin: np.ndarray | None = None) -> Trajectory:
    """Pixel centers (col, row) to meters with y up, origin at the first sample unless a fixed origin is given."""
    frames = np.asarray(frames, dtype=int)
    c = np.asarray(centers_px, dtype=float)
    o = c[0] if origin is None else np.asarray(origin, dtype=float)
    xy = np.column_stack([(c[:, 0] - o[0]) / px_per_m, -(c[:, 1] - o[1]) / px_per_m])
    return Trajectory(t=(frames - frames[0]) / fps, xy=xy, frames=frames, fps=fps)
