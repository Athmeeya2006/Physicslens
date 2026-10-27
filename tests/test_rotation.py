"""Spin-down: orientation from masks (measured exactly as for SAM masks), friction law identified."""

import subprocess
import sys

import cv2
import numpy as np
import pytest
from physicslens.agent.schemas import Scene
from physicslens.export.mjcf import check_against_fit
from physicslens.export.script import python_script
from physicslens.perception.sam_track import ObjectTrack, _shape
from physicslens.physics.fit import fit_model, simulate
from physicslens.physics.models import MODELS, models_for
from physicslens.physics.rotation import RotationError, rotation_coordinates
from physicslens.physics.verify import assess
from physicslens.pipeline import AnalysisError, rotation_summary, rotation_track
from scipy.integrate import solve_ivp

from .test_pipeline import SCENE

FPS = 30


def spin(c: float, k: float, q: float, w0: float = 18.0, seconds: float = 6.0):
    t = np.arange(0, seconds, 1 / FPS)

    def f(_t, s):
        return [s[1], -c * np.tanh(s[1] / 0.05) - k * s[1] - q * s[1] * abs(s[1])]

    return t, solve_ivp(f, (0, seconds), [0.3, w0], t_eval=t, rtol=1e-10, atol=1e-12).y[0]


def bar_track(phi: np.ndarray, length: float = 120, width: float = 30, size: int = 400) -> ObjectTrack:
    """Render a bar at each angle and measure it with the same mask-shape code used for SAM masks."""
    rows = []
    for p in phi:
        img = np.zeros((size, size), np.uint8)
        box = cv2.boxPoints(((size / 2, size / 2), (length, width), -np.degrees(p)))
        cv2.fillPoly(img, [np.round(box).astype(np.int32)], 1)
        rows.append(_shape(img, 0, 0))
    m = np.array(rows)
    n = len(phi)
    return ObjectTrack("1", np.arange(n), m[:, 0:2], m[:, 4], m[:, 2], m[:, 3], np.ones(n))


@pytest.mark.parametrize("law,ckq,model", [
    ("dry", (3.0, 0, 0), "spin_dry"),
    ("viscous", (0, 0.45, 0), "spin_viscous"),
    ("dry+viscous", (1.5, 0.25, 0), "spin_dry_viscous"),
    ("air", (0, 0, 0.05), "spin_air"),
])
def test_friction_law_identified(law, ckq, model):
    _, phi = spin(*ckq)
    coords, _, _ = rotation_track(bar_track(phi), FPS, 1000.0)
    verdict = assess(coords, [fit_model(s, coords) for s in models_for("rotation")])
    assert verdict.status == "accepted" and verdict.winner.model_id == model
    truth = dict(zip(("c", "k", "q"), ckq, strict=True))
    for name, value in verdict.winner.params.items():
        assert value == pytest.approx(truth[name], rel=0.03, abs=0.005)


def test_round_object_is_refused():
    _, phi = spin(3.0, 0, 0)
    with pytest.raises(AnalysisError, match="round"):
        rotation_track(bar_track(phi, length=60, width=56), FPS, 1000.0)


def test_spin_too_fast_for_frame_rate_is_refused():
    t = np.arange(0, 2, 1 / FPS)
    phi = 50.0 * t  # 50 rad/s: 1.67 rad per frame, beyond the quarter-turn limit
    with pytest.raises(RotationError, match="too fast"):
        rotation_coordinates(np.arange(len(t)), -((phi + np.pi / 2) % np.pi - np.pi / 2), np.zeros((len(t), 2)),
                             np.full(len(t), 120.0), np.full(len(t), 30.0), FPS, 1000.0)


def test_rotation_summary_describes_the_spin_down():
    _, phi = spin(3.0, 0, 0)
    s = rotation_summary(rotation_track(bar_track(phi), FPS, 1000.0)[0])
    speeds = s["angular_speed_rad_s_at_0_25_50_75_100_percent"]
    assert s["kind"] == "rotation" and speeds[0] > speeds[2] > speeds[-1] >= -0.5
    assert s["turns"] == pytest.approx((18 * 6 - 0.5 * 3 * 36) / (2 * np.pi), rel=0.05)


@pytest.mark.parametrize("model,ckq", [("spin_viscous", (0, 0.45, 0)), ("spin_dry_viscous", (1.5, 0.25, 0))])
def test_spin_exports_reproduce_the_fit(tmp_path, model, ckq):
    _, phi = spin(*ckq)
    coords, traj, _ = rotation_track(bar_track(phi), FPS, 1000.0)
    f = fit_model(MODELS[model], coords)
    fs = {"model_id": model, "params": f.params, "y0": [float(v) for v in f.y0], "ctx": coords.ctx, "fps": FPS,
          "origin_px": [0.0, 0.0], "px_per_m": 1000.0, "first_frame": 0}
    assert check_against_fit(fs, len(traj) / FPS)["rms_vs_fit_m"] < 1e-4
    result = {"fit_state": fs, "status": "accepted", "clip": {"frames": len(traj), "fps": FPS}, "parameters": []}
    (tmp_path / "sim.py").write_text(python_script(result))
    subprocess.run([sys.executable, "-I", "sim.py"], cwd=tmp_path, check=True, capture_output=True)
    data = np.loadtxt(tmp_path / "physicslens_trajectory.csv", delimiter=",", skiprows=1)
    ours = simulate(MODELS[model], f.params, np.array(f.y0), data[:, 0], coords.ctx)
    assert np.max(np.abs(data[:, 1:] - ours)) < 1e-9


def test_scene_accepts_rotation():
    assert Scene.model_validate({**SCENE, "motion_family": "rotation"}).motion_family == "rotation"
