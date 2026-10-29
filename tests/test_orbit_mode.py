"""Orbit mode plumbing: a known image scale, a fixed origin (the Sun) and time-lapse time, with the law still fitted."""

import numpy as np
from physicslens.agent.schemas import Scene
from physicslens.perception.sam_track import ObjectTrack
from physicslens.pipeline import RunOptions, calibrate, full_track
from physicslens.sigproc.trajectory import from_pixels

from .test_pipeline import SCENE


def track(points: np.ndarray) -> ObjectTrack:
    n = len(points)
    return ObjectTrack("1", np.arange(n), points, np.zeros(n), np.full(n, 20.0), np.full(n, 20.0), np.ones(n))


def test_fixed_origin_puts_the_sun_at_zero():
    px = np.array([[640.0, 540.0], [540.0, 440.0]])
    traj = from_pixels(np.array([0, 1]), px, 0.4, 440.0, origin=np.array([540.0, 540.0]))
    assert np.allclose(traj.xy, [[100 / 440, 0.0], [0.0, 100 / 440]])  # y up: a row above the Sun is positive y
    assert np.allclose(from_pixels(np.array([0, 1]), px, 0.4, 440.0).xy[0], [0, 0])  # default: first sample


def test_known_scale_wins_and_time_lapse_sets_physics_time():
    t = track(np.column_stack([540 + 100 * np.cos(np.linspace(0, 3, 40)), 540 - 100 * np.sin(np.linspace(0, 3, 40))]))
    opts = RunOptions(known_px_per_unit=440.0, origin_px=(540.0, 540.0), slow_motion=1 / 75, units={"length": "AU", "time": "day"})
    cal = calibrate(Scene(**SCENE), None, t, opts)
    assert cal.px_per_m == 440.0 and cal.source == "known_scale" and cal.rel_std == 0.0
    traj, origin = full_track(t, 30 * opts.slow_motion, cal.px_per_m, opts.origin_px)
    assert np.allclose(origin, [540, 540])
    assert np.allclose(np.hypot(*traj.xy.T), 100 / 440)  # every point the same distance from the Sun
    assert np.isclose(traj.t[1] - traj.t[0], 2.5)  # 30 fps at 1/75: one frame is 2.5 days
