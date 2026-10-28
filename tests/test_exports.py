"""Exports must reproduce the fitted model: MuJoCo MJCF, the standalone Python script, and the JSON file."""

import json
import subprocess
import sys

import mujoco
import numpy as np
import pytest
from physicslens.export.mjcf import build_mjcf, check_against_fit
from physicslens.export.script import parameters_json, python_script
from physicslens.physics.coords import to_coordinates
from physicslens.physics.fit import fit_model, simulate
from physicslens.physics.models import MODELS
from physicslens.sigproc.trajectory import Trajectory
from scipy.integrate import solve_ivp

from .test_regime import FPS, pendulum


def state_for(traj: Trajectory, family: str, model_id: str) -> dict:
    c = to_coordinates(traj, family)
    f = fit_model(MODELS[model_id], c)
    return {"model_id": model_id, "params": f.params, "y0": [float(v) for v in f.y0],
            "ctx": {k: float(v) for k, v in c.ctx.items()}, "fps": traj.fps, "origin_px": [320.0, 100.0],
            "px_per_m": 500.0, "first_frame": 0}


def spring() -> Trajectory:
    t = np.arange(0, 6, 1 / FPS)
    s = 0.05 * np.exp(-0.15 * t) * np.cos(2 * np.pi * 1.3 * t)
    xy = np.column_stack([0.3 * s, s + 0.3]) + np.random.default_rng(0).normal(0, 0.001, (len(t), 2))
    return Trajectory(t, xy, np.arange(len(t)), FPS)


def throw() -> Trajectory:
    t = np.arange(0, 0.9, 1 / FPS)
    xy = solve_ivp(lambda _t, s: [s[2], s[3], 0, -9.81], (0, 0.9), [0, 0, 2, 3], t_eval=t).y[:2].T
    return Trajectory(t, xy, np.arange(len(t)), FPS)


EXACT = [
    ("pendulum", "pendulum", lambda: pendulum([(8, 0.6, 0.0)], theta0=0.9)),
    ("pendulum", "pendulum_linear_damping", lambda: pendulum([(8, 0.6, 0.08)], theta0=0.9)),
    ("oscillator", "damped_harmonic", spring),
    ("projectile", "free_fall", throw),
]


@pytest.mark.parametrize("family,model_id,make", EXACT)
def test_mujoco_reproduces_exactly_representable_models(family, model_id, make):
    traj = make()
    fs = state_for(traj, family, model_id)
    xml, notes = build_mjcf(fs)
    mujoco.MjModel.from_xml_string(xml)  # valid MJCF
    check = check_against_fit(fs, len(traj) / FPS)
    assert notes == [] and check["rms_vs_fit_m"] < 1e-4  # under 0.1 mm over the whole clip
    assert len(check["points"]) == len(traj)


def test_air_drag_is_approximated_and_says_so():
    traj = pendulum([(8, 0.6, 0.08)], theta0=0.9)
    check = check_against_fit(state_for(traj, "pendulum", "pendulum_quadratic_damping"), 8)
    assert check["notes"] and "energy-equivalent" in check["notes"][0]
    assert check["rms_vs_fit_m"] < 0.015


@pytest.mark.parametrize("family,model_id,make", [*EXACT, ("pendulum", "pendulum_quadratic_damping",
                                                             lambda: pendulum([(8, 0.6, 0.08)], theta0=0.9))])
def test_python_script_runs_standalone_and_matches(tmp_path, family, model_id, make):
    traj = make()
    fs = state_for(traj, family, model_id)
    result = {"fit_state": fs, "status": "accepted", "clip": {"frames": len(traj), "fps": FPS},
              "parameters": [{"name": k, "value": v, "std": 0.01, "unit": "", "ci95": [v - 0.02, v + 0.02]}
                             for k, v in fs["params"].items()]}
    script = tmp_path / "sim.py"
    script.write_text(python_script(result))
    subprocess.run([sys.executable, "-I", str(script)], cwd=tmp_path, check=True, capture_output=True)
    data = np.loadtxt(tmp_path / "physicslens_trajectory.csv", delimiter=",", skiprows=1)
    t = data[:, 0]
    ours = simulate(MODELS[model_id], fs["params"], np.array(fs["y0"]), t, fs["ctx"])
    assert np.max(np.abs(data[:, 1:] - ours)) < 1e-9


def test_parameters_json_is_complete():
    traj = pendulum([(8, 0.6, 0.08)])
    fs = state_for(traj, "pendulum", "pendulum_linear_damping")
    doc = json.loads(parameters_json({"fit_state": fs, "status": "accepted", "reason": "r"}))
    assert doc["model"]["id"] == "pendulum_linear_damping" and doc["geometry"]["length_m"] > 0
    assert len(doc["initial_state"]) == 2 and doc["units"].startswith("SI")


@pytest.mark.parametrize("kind,marker", [("json", '"generator": "PhysicsLens"'), ("py", "from scipy.integrate import solve_ivp"),
                                         ("mjcf", "<mujoco")])
def test_export_endpoints(kind, marker):
    from fastapi.testclient import TestClient
    from physicslens import app as app_module

    from .test_whatif import finished_run
    fs = state_for(pendulum([(6, 0.6, 0.05)]), "pendulum", "pendulum_linear_damping")
    run_id = finished_run(app_module, fs)
    r = TestClient(app_module.app).get(f"/api/runs/{run_id}/export/{kind}")
    assert r.status_code == 200 and marker in r.text
    assert f"physicslens-{run_id}" in r.headers["content-disposition"]


def test_mujoco_replays_linear_drag_exactly():
    from physicslens.export.mjcf import build_mjcf, check_against_fit

    fs = {"model_id": "linear_drag", "params": {"g": 9.81, "k": 2.6}, "y0": [0.0, 0.0, 2.0, 3.0], "ctx": {},
          "fps": 30.0, "origin_px": [100.0, 400.0], "px_per_m": 500.0, "first_frame": 0}
    xml, notes = build_mjcf(fs)
    assert 'type="free" damping=' in xml and not notes  # joint damping = k * mass is exactly linear drag
    assert check_against_fit(fs, 1.5)["rms_vs_fit_m"] < 1e-4


def test_report_covers_every_verdict():
    import json
    from pathlib import Path

    from physicslens.export.report import report_markdown

    accepted = json.loads(Path("gallery/gallery-paperbag/result.json").read_text())
    md = report_markdown(accepted, "Paper bag")
    for section in ("# Paper bag", "## Setup", "## Method", "## Result", "## Every candidate tested", "## Limitations",
                    "## Provenance", "slow motion x20", "s⁻¹"):
        assert section in md
    no_fit = {"status": "no_fit", "reason": "Nothing forecasts the held-out motion.", "trail": [], "ranking": [],
              "llm_calls": [], "clip": {"width": 640, "height": 360, "fps": 30.0, "frames": 60},
              "calibration": {"source": "none", "px_per_m": 1000.0, "rel_std": 0.0}}
    md = report_markdown(no_fit)
    assert "No clean governing equation" in md and "## Result" not in md
