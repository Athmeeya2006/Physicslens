"""Double pendulum: short-horizon fitting, chaos measurement, and the two-object pipeline path."""

import json

import numpy as np
import pytest
from physicslens.agent.schemas import Scene
from physicslens.perception.media import VideoInfo
from physicslens.perception.sam_track import ObjectTrack
from physicslens.physics.double import fit_double, simulate, to_double_coords
from physicslens.pipeline import AnalysisError, double_analysis, track_objects
from physicslens.sigproc.trajectory import Trajectory
from pydantic import ValidationError

from .test_pipeline import SCENE

FPS, L1, L2, MU = 60, 0.3, 0.25, 0.8


def bobs(th1: float, th2: float, seconds: float = 10, noise: float = 0.002, seed: int = 1):
    t = np.arange(0, seconds, 1 / FPS)
    q = simulate({"g": 9.81, "mu": MU, "gamma": 0.02}, L1, L2, np.array([th1, th2, 0, 0]), t)
    x1, y1 = 0.1 + L1 * np.sin(q[:, 0]), 0.9 - L1 * np.cos(q[:, 0])
    x2, y2 = x1 + L2 * np.sin(q[:, 1]), y1 - L2 * np.cos(q[:, 1])
    rng = np.random.default_rng(seed)
    f = np.arange(len(t))
    inner = Trajectory(t, np.column_stack([x1, y1]) + rng.normal(0, noise, (len(t), 2)), f, FPS)
    outer = Trajectory(t, np.column_stack([x2, y2]) + rng.normal(0, noise, (len(t), 2)), f, FPS)
    return inner, outer


@pytest.fixture(scope="module")
def chaotic():
    return fit_double(to_double_coords(*bobs(2.0, 2.5)))


def test_chaotic_swing_is_fitted_and_flagged(chaotic):
    assert chaotic.accepted
    assert chaotic.params["g"] == pytest.approx(9.81, rel=0.02)
    assert chaotic.params["mu"] == pytest.approx(MU, rel=0.1)
    assert chaotic.chaotic and chaotic.lyapunov_per_s > 1.0
    assert 0.5 < chaotic.predictability_s < 5
    assert chaotic.heldout_rmse_rad < 0.2 * chaotic.naive_rmse_rad  # real predictive skill, not luck


def test_regular_swing_is_not_chaotic():
    f = fit_double(to_double_coords(*bobs(0.6, 0.6)))
    assert f.accepted and not f.chaotic and f.predictability_s is None
    assert f.params["g"] == pytest.approx(9.81, rel=0.02)


def test_flat_arc_refuses_to_report_g():
    c = to_double_coords(*bobs(0.15, 0.1))
    f = fit_double(c)
    assert not c.length_reliable and not f.accepted and "too flat" in f.reason


def as_track(oid: str, traj: Trajectory, ppm: float = 1000.0, origin=(400.0, 50.0)) -> ObjectTrack:
    px = np.column_stack([origin[0] + traj.xy[:, 0] * ppm, origin[1] - traj.xy[:, 1] * ppm])
    n = len(traj)
    return ObjectTrack(oid, traj.frames, px, np.zeros(n), np.full(n, 30.0), np.full(n, 30.0), np.ones(n))


def test_pipeline_double_analysis():
    inner, outer = bobs(2.0, 2.5, seconds=8)
    out, ghosts, steps = double_analysis(as_track("1", inner), as_track("3", outer), FPS, 1000.0)
    assert out["accepted"] and out["chaotic"] and len(ghosts) == 3
    assert [s["step"] for s in steps] == ["verification", "chaos"] and "Chaotic" in steps[1]["text"]
    real = {p[0]: np.array(p[1:]) for p in out["outer_track_px"]}
    anchored = {p[0]: np.array(p[1:]) for p in ghosts[1]["points"]}
    free = {p[0]: np.array(p[1:]) for p in ghosts[2]["points"]}
    err_anchored = np.array([np.linalg.norm(anchored[f] - real[f]) for f in sorted(anchored)])
    err_free = np.array([np.linalg.norm(free[f] - real[f]) for f in sorted(free)])
    real = np.array(list(real.values()))
    sweep = float(np.ptp(real, axis=0).max())
    assert np.median(err_anchored) < 0.03 * sweep  # restarted predictions stay on the bob
    assert err_free[-FPS:].mean() > 5 * err_free[:FPS].mean()  # one long prediction drifts away: chaos on screen


def test_scene_requires_outer_bob_for_double():
    with pytest.raises(ValidationError):
        Scene.model_validate({**SCENE, "motion_family": "double_pendulum"})
    ok = Scene.model_validate({**SCENE, "motion_family": "double_pendulum", "second_object": "lower bob",
                               "second_object_box": {"x1": 600, "y1": 700, "x2": 650, "y2": 760}})
    assert ok.second_object == "lower bob"


def test_both_bobs_tracked_in_one_sam_request(harness, tmp_path):
    scene = Scene.model_validate({**SCENE, "motion_family": "double_pendulum", "second_object": "lower bob",
                                  "second_object_box": {"x1": 600, "y1": 700, "x2": 650, "y2": 760}})
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake")
    info = VideoInfo(640, 360, 30.0, 2.0, 60, "h264")
    with pytest.raises(AnalysisError):  # the mock SAM finds nothing; we only inspect the request
        track_objects(harness.client, clip, info, scene, "r", "dev")
    prompt = next(c for c in harness.sdk().calls if c["kind"] == "stream")["input"][0]["content"][0]["text"]
    assert "1<|box" in prompt and "3<|box" in prompt and json.dumps(prompt).count("<|box") == 2


def test_junk_motion_is_not_accepted():
    rng = np.random.default_rng(7)
    t = np.arange(0, 8, 1 / FPS)
    f = np.arange(len(t))
    # An inner "bob" wandering on an arc and an outer one jittering at random: no double-pendulum physics.
    th = np.cumsum(rng.normal(0, 0.02, len(t)))
    inner = Trajectory(t, np.column_stack([0.3 * np.sin(th), 0.9 - 0.3 * np.cos(th)]), f, FPS)
    outer = Trajectory(t, inner.xy + np.column_stack([0.25 * np.sin(np.cumsum(rng.normal(0, 0.15, len(t)))),
                                                      -0.25 * np.ones(len(t))]), f, FPS)
    assert not fit_double(to_double_coords(inner, outer)).accepted
