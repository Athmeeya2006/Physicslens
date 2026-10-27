"""Bouncing balls: shared-g flights plus restitution at each impact, recovered from noisy tracks."""

import time

import numpy as np
import pytest
from physicslens.physics.collisions import (
    BounceError,
    detect_bounces,
    fit_bounces,
    simulate_bounces,
)
from physicslens.pipeline import analyze_bounces, bounce_trail_step, longest_flight
from physicslens.sigproc.trajectory import Trajectory

G = 9.81


def bouncing(fps: int, e: float, noise: float, seed: int = 1, seconds: float = 2.5) -> Trajectory:
    t = np.arange(0, seconds, 1 / fps)
    xy = simulate_bounces(t, (0.0, 1.0, 0.4, 0.0), 0.0, G, 0.0, e, 0.95)
    return Trajectory(t, xy + np.random.default_rng(seed).normal(0, noise, xy.shape), np.arange(len(t)), fps)


@pytest.mark.parametrize("fps,e,noise", [(60, 0.75, 0.002), (30, 0.8, 0.002), (30, 0.6, 0.004), (60, 0.9, 0.003), (30, 0.7, 0.008)])
@pytest.mark.parametrize("seed", [1, 2])
def test_recovers_g_and_restitution(fps, e, noise, seed):
    b = fit_bounces(bouncing(fps, e, noise, seed))
    assert len(b.restitution) >= 2
    assert b.g == pytest.approx(G, rel=0.02)
    assert all(abs(x - e) < 0.02 for x in b.restitution)
    assert b.accepted and b.forecast_nrmse < 0.05


def test_refuses_when_bounces_cannot_be_resolved():
    with pytest.raises(BounceError, match="two clean flights"):
        fit_bounces(bouncing(30, 0.5, 0.006, seed=2))


def test_no_bounces_found_in_a_swing():
    t = np.arange(0, 4, 1 / 30)
    xy = np.column_stack([0.5 * np.sin(3 * t), -0.5 * np.cos(3 * t)]) + np.random.default_rng(2).normal(0, 0.002, (len(t), 2))
    assert len(detect_bounces(Trajectory(t, xy, np.arange(len(t)), 30))) == 0


@pytest.mark.parametrize("g,e", [(-9.81, 0.7), (9.81, 1.5), (9.81, -0.2), (0.0, 0.7)])
def test_simulator_rejects_unphysical_inputs(g, e):
    with pytest.raises(BounceError):
        simulate_bounces(np.arange(0, 2, 0.01), (0, 1, 0, 0), 0, g, 0, e, 1.0)


def test_simulator_always_terminates():
    start = time.monotonic()
    out = simulate_bounces(np.arange(0, 30, 1 / 60), (0, 1, 0.1, 0), 0, G, 0, 0.98, 1.0)  # nearly elastic, long
    assert time.monotonic() - start < 2 and np.all(out[:, 1] >= -1e-9)


def test_pipeline_bounce_analysis_and_ghost():
    traj = bouncing(60, 0.75, 0.002)
    ppm, origin = 400.0, np.array([100.0, 50.0])
    b = analyze_bounces(traj, origin, ppm)
    assert b is not None and b["count"] >= 4 and abs(b["e_mean"] - 0.75) < 0.01
    real_px = origin + np.column_stack([traj.xy[:, 0], -traj.xy[:, 1]]) * ppm
    ghost = np.array([p[1:] for p in b["ghost"]])
    assert np.median(np.linalg.norm(ghost - real_px, axis=1)) < 3  # the hybrid ghost follows every bounce
    step = bounce_trail_step(b)
    assert step["step"] == "collision" and "e = 0.75" in step["text"] and len(step["items"]) == b["count"]


def test_longest_flight_splits_at_small_bounces():
    traj = bouncing(30, 0.6, 0.004)
    flight, _ = longest_flight(traj, np.array([0.0, 0.0]), 400.0)
    assert flight.t[-1] - flight.t[0] < 0.5  # the drop before the first impact, not several merged hops
