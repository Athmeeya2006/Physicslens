import numpy as np
import pytest
from physicslens.sigproc.cleanup import (
    detect_impacts,
    differentiate,
    fill_short_gaps,
    split_at,
)
from physicslens.sigproc.trajectory import Trajectory, from_pixels


def test_short_gaps_filled_long_gaps_split():
    frames = np.array([0, 1, 2, 4, 5, 20, 21, 22])
    values = np.column_stack([frames * 1.0, frames * 2.0])
    segments = fill_short_gaps(frames, values, max_gap=3)
    assert len(segments) == 2
    f0, v0 = segments[0]
    assert list(f0) == [0, 1, 2, 3, 4, 5]
    assert v0[3].tolist() == [3.0, 6.0]


def test_derivatives_of_a_sine():
    fps = 60
    t = np.arange(0, 4, 1 / fps)
    k = differentiate(np.sin(2 * t), fps)
    inner = slice(10, -10)
    assert np.allclose(k.vel[inner], 2 * np.cos(2 * t[inner]), atol=1e-3)
    assert np.allclose(k.acc[inner], -4 * np.sin(2 * t[inner]), atol=1e-2)


def test_pixels_to_meters_flips_y():
    traj = from_pixels(np.array([10, 11]), np.array([[100.0, 200.0], [110.0, 190.0]]), fps=30, px_per_m=100)
    assert traj.xy[1].tolist() == pytest.approx([0.1, 0.1])
    assert traj.t.tolist() == pytest.approx([0, 1 / 30])


def bouncing_ball(fps=60, seconds=2.0, e=0.8):
    t = np.arange(0, seconds, 1 / fps)
    y, vy, ys = 1.0, 0.0, []
    for _ in t:
        ys.append(y)
        vy -= 9.81 / fps
        y += vy / fps
        if y < 0:
            y, vy = -y, -e * vy
    return Trajectory(t, np.column_stack([0.3 * t, ys]), np.arange(len(t)), fps)


def test_impacts_found_on_bounces_not_on_smooth_swings():
    ball = bouncing_ball()
    impacts = detect_impacts(ball)
    assert 2 <= len(impacts) <= 4
    assert all(ball.xy[i, 1] < 0.1 for i in impacts)
    assert len(split_at(ball, impacts)) == len(impacts) + 1

    t = np.arange(0, 4, 1 / 30)
    swing = Trajectory(t, np.column_stack([0.5 * np.sin(3 * t), -0.5 * np.cos(3 * t)]), np.arange(len(t)), 30)
    assert len(detect_impacts(swing)) == 0
