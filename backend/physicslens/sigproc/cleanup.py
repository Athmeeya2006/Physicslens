"""Gap filling, smoothing, derivatives and impact detection. Pure functions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.signal import savgol_filter

from .trajectory import Trajectory


def fill_short_gaps(frames: np.ndarray, values: np.ndarray, max_gap: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    """Linearly fill gaps of up to max_gap missing frames; split into segments at longer gaps.

    Long gaps are never bridged: inventing positions there would invent physics.
    """
    frames = np.asarray(frames, dtype=int)
    values = np.asarray(values, dtype=float)
    order = np.argsort(frames)
    frames, values = frames[order], values[order]
    breaks = np.nonzero(np.diff(frames) > max_gap + 1)[0] + 1
    segments = []
    for f, v in zip(np.split(frames, breaks), np.split(values, breaks), strict=True):
        full = np.arange(f[0], f[-1] + 1)
        filled = np.column_stack([np.interp(full, f, v[:, i]) for i in range(v.shape[1])]) if v.ndim == 2 else np.interp(full, f, v)
        segments.append((full, filled))
    return segments


def savgol_window(n: int, fps: float, window_s: float, polyorder: int) -> int:
    w = round(window_s * fps) | 1  # odd
    w = max(w, polyorder + 2 + (polyorder % 2 == 0))
    return min(w, n if n % 2 else n - 1)


@dataclass(frozen=True)
class Kinematics:
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray


def differentiate(values: np.ndarray, fps: float, window_s: float = 0.12, polyorder: int = 3) -> Kinematics:
    """Savitzky-Golay smoothed position, velocity and acceleration along axis 0 (uniform sampling required)."""
    values = np.asarray(values, dtype=float)
    w = savgol_window(len(values), fps, window_s, polyorder)
    if w <= polyorder:
        raise ValueError(f"Too few samples ({len(values)}) to differentiate.")
    dt = 1.0 / fps
    return Kinematics(
        pos=savgol_filter(values, w, polyorder, axis=0),
        vel=savgol_filter(values, w, polyorder, deriv=1, delta=dt, axis=0),
        acc=savgol_filter(values, w, polyorder, deriv=2, delta=dt, axis=0),
    )


def detect_impacts(traj: Trajectory, threshold_mads: float = 8.0, min_separation_s: float = 0.1) -> np.ndarray:
    """Indices where velocity jumps abruptly (collisions), from frame-to-frame raw velocity changes.

    Smooth turning points (a pendulum at max swing) change velocity gradually, so they don't trigger.
    """
    if len(traj) < 5:
        return np.array([], dtype=int)
    v = np.diff(traj.xy, axis=0) * traj.fps
    dv = np.linalg.norm(np.diff(v, axis=0), axis=1)
    mad = np.median(np.abs(dv - np.median(dv))) + 1e-12
    candidates = np.nonzero(dv > np.median(dv) + threshold_mads * 1.4826 * mad)[0] + 1
    impacts: list[int] = []
    for i in candidates:
        if not impacts or (i - impacts[-1]) / traj.fps >= min_separation_s:
            impacts.append(int(i))
        elif dv[i - 1] > dv[impacts[-1] - 1]:
            impacts[-1] = int(i)
    return np.array(impacts, dtype=int)


def split_at(traj: Trajectory, indices: np.ndarray, min_len: int = 10) -> list[Trajectory]:
    bounds = [0, *[int(i) for i in indices], len(traj)]
    return [traj.slice(a, b) for a, b in pairwise(bounds) if b - a >= min_len]
