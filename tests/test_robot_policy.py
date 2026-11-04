"""The AI-policy comparison (scripts/robot_policy.py): its fair baseline (one command delay and per-joint offsets, fitted
on training episodes) recovers a known delay and offset, and scores a held-out episode it describes exactly near zero."""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("robot_policy", ROOT / "scripts" / "robot_policy.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def episodes(delay_frames: int, offset: np.ndarray, n: int = 6, frames: int = 300, seed: int = 0):
    """Smooth commands on six joints; the arm follows them delay_frames later, plus a constant offset per joint."""
    rng = np.random.default_rng(seed)
    us, ys = {}, {}
    t = np.arange(frames) * P.DT
    for e in range(n):
        u = np.column_stack([20 * np.sin(2 * np.pi * (0.2 + 0.07 * j) * t + rng.uniform(0, 6)) for j in range(6)])
        y = np.vstack([np.repeat(u[:1], delay_frames, axis=0), u[:-delay_frames]]) if delay_frames else u.copy()
        y[:, :5] += offset
        us[e], ys[e] = u, y
    return us, ys


def test_baseline_recovers_delay_and_offset():
    off = np.array([1.5, -2.0, 0.5, 0.0, 3.0])
    us, ys = episodes(delay_frames=3, offset=off)
    err, d = P.delay_offset(us, ys, [0, 1, 2, 3], [4, 5])
    assert abs(d - 3 * P.DT) < 1e-6  # 3 frames at 30 per second: 0.1 s, on the grid of whole frames
    assert np.all(err < 0.05)  # the baseline describes these episodes exactly, up to the start-up frames


def test_baseline_is_worse_when_the_arm_lags_rather_than_waits():
    """A lagging arm (first-order, not a pure delay) is what a delay and offsets cannot describe."""
    us, ys = episodes(delay_frames=0, offset=np.zeros(5))
    for e in ys:  # a 0.1 s first-order lag on every joint
        y = ys[e].copy()
        for k in range(1, len(y)):
            y[k] = y[k - 1] + (P.DT / 0.1) * (us[e][k] - y[k - 1])
        ys[e] = y
    err, _ = P.delay_offset(us, ys, [0, 1, 2, 3], [4, 5])
    assert np.all(err > 0.05)
