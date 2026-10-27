"""Compact numeric description of a trajectory: what the language model sees instead of video."""

from __future__ import annotations

from typing import Any

import numpy as np

from .cleanup import detect_impacts, differentiate
from .trajectory import Trajectory


def summarize(traj: Trajectory) -> dict[str, Any]:
    kin = differentiate(traj.xy, traj.fps)
    xy = kin.pos
    centered = xy - xy.mean(axis=0)
    _, sv, vt = np.linalg.svd(centered, full_matrices=False)
    main = centered @ vt[0]

    # Path shape: a circle fit tells an arc (pendulum) from a line (spring) or a parabola (projectile).
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    (cx, cy, c), *_ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)
    radius = float(np.sqrt(max(c + cx**2 + cy**2, 0)))
    arc_resid = float(np.std(np.hypot(x - cx, y - cy) - radius))

    spec = np.abs(np.fft.rfft((main - main.mean()) * np.hanning(len(main))))
    freqs = np.fft.rfftfreq(len(main), traj.dt)
    spec[0] = 0
    sign_changes = int(np.sum(np.diff(np.sign(main - np.median(main))) != 0))
    peaks = np.abs(main[np.r_[False, (np.diff(np.sign(np.diff(main))) != 0), False]])

    r3 = lambda v: float(round(float(v), 3))
    return {
        "duration_s": r3(traj.t[-1] - traj.t[0]),
        "fps": r3(traj.fps),
        "samples": len(traj),
        "extent_m": {"x": r3(np.ptp(x)), "y": r3(np.ptp(y))},
        "linearity": r3(sv[0] / (sv.sum() + 1e-12)),  # 1.0 = motion along one straight line
        "main_axis_direction": {"x": r3(vt[0][0]), "y": r3(vt[0][1])},
        "circle_fit": {"radius_m": r3(radius), "center_above_path": bool(cy > y.mean()), "residual_m": r3(arc_resid)},
        "dominant_frequency_hz": r3(freqs[np.argmax(spec)]),
        "zero_crossings_of_main_axis": sign_changes,
        "amplitude_first_vs_last_peak": [r3(peaks[0]), r3(peaks[-1])] if len(peaks) >= 2 else None,
        "mean_acceleration_m_s2": {"x": r3(kin.acc[:, 0].mean()), "y": r3(kin.acc[:, 1].mean())},
        "max_speed_m_s": r3(np.linalg.norm(kin.vel, axis=1).max()),
        "impacts_detected": len(detect_impacts(traj)),
    }
