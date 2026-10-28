"""Sim-to-real gap: textbook physics against the identified model on measured motion, and a simulated control study."""

import json

import mujoco
import numpy as np
import pytest
from physicslens.export.sim2real import (
    Sim2RealError,
    actuated_mjcf,
    compare_to_video,
    control_study,
    measured_coordinates,
    pendulum_constants,
    sim2real,
)
from physicslens.physics.coords import to_coordinates
from physicslens.physics.fit import fit_model
from physicslens.physics.models import MODELS

from .test_exports import spring, state_for
from .test_regime import FPS, PIVOT, pendulum

L = 0.6
GAMMA = 0.3  # 1/s: the swing loses about a third of its amplitude per second and a half
GEOMETRY = {"pivot_x": PIVOT[0], "pivot_y": PIVOT[1], "length_m": L, "length_std": 0.0, "length_given": 1.0}
ORIGIN_PX, PPM = (320.0, 80.0), 400.0


@pytest.fixture(scope="module")
def damped():
    """A damped pendulum filmed for 8 s with 2 mm tracking noise, fitted as the pipeline would with a given length."""
    traj = pendulum([(8, L, GAMMA)], theta0=1.0)
    c = to_coordinates(traj, "pendulum", GEOMETRY)
    f = fit_model(MODELS["pendulum_linear_damping"], c)
    fs = {"model_id": f.model_id, "params": f.params, "y0": [float(v) for v in f.y0], "ctx": dict(GEOMETRY),
          "fps": float(FPS), "origin_px": list(ORIGIN_PX), "px_per_m": PPM, "first_frame": 0}
    return traj, c, fs


@pytest.fixture(scope="module")
def spring_fit():
    traj = spring()
    return traj, {**state_for(traj, "oscillator", "damped_harmonic"), "origin_px": list(ORIGIN_PX), "px_per_m": PPM}


def stored_result(traj, fs, status="accepted", source="user") -> dict:
    """What result.json holds: the pixel track, with the fit's geometry measured from its first point."""
    xy = traj.xy - traj.xy[0]
    px = [[int(f), ORIGIN_PX[0] + x * PPM, ORIGIN_PX[1] - y * PPM] for f, (x, y) in zip(traj.frames, xy, strict=True)]
    ctx = dict(fs["ctx"])
    for k_x, k_y in (("pivot_x", "pivot_y"), ("origin_x", "origin_y")):
        if k_x in ctx:
            ctx[k_x] -= traj.xy[0, 0]
            ctx[k_y] -= traj.xy[0, 1]
    return {"status": status, "family": MODELS[fs["model_id"]].family, "real_track_px": px,
            "calibration": {"source": source, "px_per_m": PPM, "rel_std": 0.0}, "fit_state": {**fs, "ctx": ctx}}


def test_textbook_pendulum_misses_held_out_motion_that_the_identified_model_forecasts(damped):
    _, c, fs = damped
    v = compare_to_video(c, fs, calibrated=True)
    tb, pl = v["textbook"]["errors"], v["physicslens"]["errors"]
    assert v["textbook"]["model_id"] == "pendulum" and v["textbook"]["params"] == {"g": 9.81}
    # Refitted without the last quarter, the identified model still forecasts it to within a few millimetres.
    assert pl["heldout_m"] < 0.005 and abs(v["physicslens"]["params"]["gamma"] - GAMMA) < 0.02
    assert tb["heldout_m"] > 0.05 and v["ratio_heldout"] > 20 and v["ratio_whole"] > 10
    assert len(v["t"]) == len(v["measured"]) == len(v["textbook_series"]) == len(v["physicslens_series"])
    assert abs(v["heldout_start"] - 6.0) < 0.1


def test_textbook_tuned_swing_up_falls_short_on_a_damped_pendulum(damped):
    _, _, fs = damped
    rng = np.random.default_rng(1)
    plants = [pendulum_constants({"g": fs["params"]["g"] + rng.normal(0, 0.02),
                                  "gamma": fs["params"]["gamma"] + rng.normal(0, 0.01)}, L) for _ in range(5)]
    cs = control_study(fs, plants, calibrated=True)
    tb, pl = cs["summary"]["textbook"], cs["summary"]["physicslens"]
    assert cs["label"].startswith("Simulation.") and cs["plants"] == 5
    # Ignoring damping, the energy controller settles where pumping and loss balance: short of the target.
    assert tb["final_error_deg"]["mean"] < -1.5 and tb["settled"] == 0
    assert abs(pl["final_error_deg"]["mean"]) < 0.3 and pl["settled"] == 5
    assert pl["settling_s"]["mean"] < 4.0 and pl["overshoot_pct"]["mean"] < 1.0


def test_motor_torque_is_clamped_to_its_limit(damped):
    _, _, fs = damped
    model = mujoco.MjModel.from_xml_string(actuated_mjcf(fs, 0.7))
    data = mujoco.MjData(model)
    data.ctrl[0] = 5.0
    mujoco.mj_step(model, data)
    assert model.nu == 1 and data.actuator_force[0] == pytest.approx(0.7)


def test_stored_result_round_trip_and_both_parts(damped):
    traj, c, fs = damped
    result = stored_result(traj, fs)
    q = measured_coordinates(result).q
    assert np.max(np.abs(q - c.q)) < 1e-4  # the angle rebuilt from pixels is the one that was fitted
    out = sim2real(result, reps=4)
    assert out["video"]["ratio_heldout"] > 20
    assert out["control"]["plants"] == 4 and out["control"]["summary"]["physicslens"]["settled"] == 4


def test_spring_without_a_scale_compares_but_skips_control(spring_fit):
    traj, fs = spring_fit
    out = sim2real(stored_result(traj, fs, source="none"))
    v = out["video"]
    assert v["textbook"]["model_id"] == "harmonic" and "heldout_m" not in v["textbook"]["errors"]
    assert v["ratio_heldout"] > 2 and out["control"] is None


def test_needs_an_accepted_model(damped):
    traj, _, fs = damped
    with pytest.raises(Sim2RealError):
        sim2real(stored_result(traj, fs, status="ambiguous"))


def test_endpoint_computes_once_and_caches(spring_fit):
    from fastapi.testclient import TestClient
    from physicslens import app as app_module

    traj, fs = spring_fit
    run_dir = app_module.runs.new_run_dir()
    (run_dir / "result.json").write_text(json.dumps(stored_result(traj, fs, source="none")))
    client = TestClient(app_module.app)
    r = client.get(f"/api/runs/{run_dir.name}/sim2real")
    assert r.status_code == 200 and r.json()["video"]["textbook"]["model_id"] == "harmonic"
    assert (run_dir / "sim2real.json").is_file()
    assert client.get(f"/api/runs/{run_dir.name}/sim2real").json() == r.json()

    (run_dir / "result.json").write_text(json.dumps(stored_result(traj, fs, status="ambiguous", source="none")))
    (run_dir / "sim2real.json").unlink()
    assert client.get(f"/api/runs/{run_dir.name}/sim2real").status_code == 409
