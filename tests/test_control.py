"""Controllers written by the model, accepted only by simulation. All model answers are mocked: zero spend."""

import json
import time

import numpy as np
import pytest
import yaml
from physicslens.agent.schemas import ControlTask
from physicslens.control.codecheck import check_controller
from physicslens.control.harness import SandboxLimits, run_controller, summarize
from physicslens.control.loop import FAILED_MESSAGE, LoopConfig, solve
from physicslens.control.manager import ControlManager
from physicslens.control.system import (
    ControlUnsupported,
    build_plants,
    check_supported,
    evaluation_job,
    robustness_params,
)
from physicslens.costguard import BudgetExceeded
from physicslens.costguard.mock import responses_result
from physicslens.physics.coords import to_coordinates
from physicslens.physics.fit import fit_model
from physicslens.physics.models import MODELS

from .conftest import Harness
from .test_regime import FPS, pendulum
from .test_sim2real import GEOMETRY, stored_result

L = 0.6
FS = {"model_id": "pendulum_linear_damping", "params": {"g": 9.79, "gamma": 0.3}, "y0": [1.0, 0.0],
      "ctx": {"length_m": L, "length_std": 0.0, "pivot_x": 0.0, "pivot_y": 0.0, "length_given": 1.0}, "fps": 30.0}
RESULT = {"status": "accepted", "family": "pendulum", "fit_state": FS, "calibration": {"source": "user"}}
HOLD = ControlTask(family="pendulum", goal="reach_and_hold", target=0.5, tolerance=0.02, actuator_limit=4.0, time_limit_s=3.0)
FAST = SandboxLimits(cpu_seconds=20, wall_seconds=30, memory_mb=1536)

GOOD = """import math

WN = 4.0


def control(t, q, qdot, params):
    # PD on the angle error plus gravity and damping compensation from the identified model.
    err = (q[0] - params["target"] + math.pi) % (2.0 * math.pi) - math.pi
    hold = params["mass"] * params["g"] * params["L"] * math.sin(q[0])
    return [hold + params["inertia"] * (params["gamma"] * qdot[0] - WN * WN * err - 2.0 * WN * qdot[0])]
"""
WRONG_SIGN = GOOD.replace("- WN * WN * err", "+ WN * WN * err")


def variants(n: int = 10, seed: int = 0) -> list[dict[str, float]]:
    """Nominal first, then small spreads like a bootstrap gives; cheaper than refitting in every test."""
    rng = np.random.default_rng(seed)
    nominal = {"g": 9.79, "gamma": 0.3, "beta": 0.0, "L": L}
    return [nominal] + [{"g": 9.79 + rng.normal(0, 0.05), "gamma": max(0.0, 0.3 + rng.normal(0, 0.03)), "beta": 0.0,
                         "L": L + rng.normal(0, 0.003)} for _ in range(n)]


def answer(code: str, approach: str = "PD with gravity compensation.") -> object:
    return responses_result(text=json.dumps({"code": code, "approach": approach}), input_tokens=900, output_tokens=700)


@pytest.fixture
def wonder(settings) -> Harness:
    """A mocked client whose budget includes the approved Muse Wonder caps."""
    path = settings.config_dir / "budget.yaml"
    budget = yaml.safe_load(path.read_text())
    budget.update(model_daily_caps_usd={"muse-wonder-1.0": 0.40}, control_task_cap_usd=0.60,
                  meta_api={"hard_cap_usd": 2.0, "daily_cap_usd": 1.0})
    path.write_text(yaml.safe_dump(budget))
    return Harness(settings)


def run_solve(h: Harness, tmp_path, task: ControlTask = HOLD, n: int = 10, **kw):
    cfg = LoopConfig(limits=FAST)
    return solve(h.client, RESULT, task, variants(n), tmp_path / "task", ledger_run_id="control-test", audience="dev",
                 calibrated=True, cfg=cfg, **kw)


# ---------- static checks ----------

@pytest.mark.parametrize("code, why", [
    ("import os\n\ndef control(t, q, qdot, params):\n    return [0.0]\n", "import 'os'"),
    ("def control(t, q, qdot, params):\n    open('x', 'w')\n    return [0.0]\n", "open"),
    ("def control(t, q, qdot, params):\n    m = __import__('os')\n    return [0.0]\n", "__import__"),
    ("def control(t, q, qdot, params):\n    return [eval('1')]\n", "eval"),
    ("def control(t, q, qdot, params):\n    k = q.__class__\n    return [0.0]\n", "__class__"),
    ("import numpy as np\n\ndef control(t, q, qdot, params):\n    np.save('x', q)\n    return [0.0]\n", ".save"),
    ("import numpy as np\n\ndef control(t, q, qdot, params):\n    np.abs = None\n    return [0.0]\n", "assigning"),
    ("def control(t, q, qdot, params):\n    return ['{0.__class__}'.format(q)]\n", ".format"),
    ("def control(t, q, qdot):\n    return [0.0]\n", "exactly (t, q, qdot, params)"),
    ("def control(t, q, qdot, params):\n    return [0.0]\n\ndef helper():\n    return 1\n", "exactly one"),
    ("X = abs(-1)\n\ndef control(t, q, qdot, params):\n    return [0.0]\n", "simple constants"),
    ("def control(t, q, qdot, params):\n" + "    x = 1\n" * 300 + "    return [0.0]\n", "lines"),
])
def test_static_check_rejects_unsafe_code(code, why):
    r = check_controller(code)
    assert not r.ok and why in r.reason


def test_static_check_accepts_the_contract():
    assert check_controller(GOOD).ok
    assert check_controller("import numpy as np\nSTATE = {'i': 0.0}\n\ndef control(t, q, qdot, params):\n"
                            "    STATE['i'] += q[0] * 0.002\n    return [float(np.clip(-STATE['i'], -1, 1))]\n").ok


def test_unsafe_code_is_rejected_before_it_runs(wonder, tmp_path):
    wonder.sdk().script.extend([answer("import os\n\ndef control(t, q, qdot, params):\n    os.system('x')\n    return [0.0]\n"),
                                answer(GOOD)])
    out = run_solve(wonder, tmp_path)
    first, _ = out["attempts"]
    assert first["check"]["ok"] is False and first["summary"] is None  # never executed
    assert "static check" in wonder.sdk().calls[1]["input"][0]["content"][0]["text"]
    assert out["outcome"] == "accepted" and out["accepted_attempt"] == 2


# ---------- the sandbox ----------

def test_correct_controller_passes_and_is_accepted(wonder, tmp_path):
    wonder.sdk().script.append(answer(GOOD))
    out = run_solve(wonder, tmp_path)
    a = out["attempts"][0]
    assert out["outcome"] == "accepted" and a["summary"]["success_rate"] >= 0.9 and a["summary"]["nominal_success"]
    assert out["message"].startswith("Muse Wonder's controller: 11 of 11 succeed; textbook controller:")
    header = (tmp_path / "task" / "controller.py").read_text()
    assert "muse-wonder-1.0" in header and "11 of 11" in header and "Goal: reach_and_hold" in header
    assert json.loads((tmp_path / "task" / "metrics.json").read_text())["outcome"] == "accepted"
    steps = [(s["step"], s["provider"], s["model"]) for s in out["trail"]]
    assert steps == [("controller", "meta_api", "muse-wonder-1.0"), ("control_test", None, "MuJoCo simulation")]
    assert out["trail"][1]["label"] == "Tests it on 11 variants" and out["label"].startswith("Simulation.")
    # The model is told the contract, effort is low, and the call goes through the Responses API without storage.
    call = wonder.sdk().calls[0]
    assert call["model"] == "muse-wonder-1.0" and call["reasoning"] == {"effort": "low"} and call["store"] is False
    assert call["text"]["format"]["strict"] is True


def test_wrong_sign_fails_and_the_report_fixes_it(wonder, tmp_path):
    wonder.sdk().script.extend([answer(WRONG_SIGN), answer(GOOD)])
    out = run_solve(wonder, tmp_path)
    first, second = out["attempts"]
    assert first["summary"]["successes"] == 0 and not first["summary"]["passed"]
    assert len(first["worst"]) == 3 and first["worst"][0]["success"] is False
    report = wonder.sdk().calls[1]["input"][0]["content"][0]["text"]
    assert "success on 0 of 11" in report and "Worst variants" in report and WRONG_SIGN in report
    assert "/" not in report.split("Worst variants")[1].replace("1/s", "")  # no file paths go back to the model
    assert second["summary"]["passed"] and out["accepted_attempt"] == 2
    assert (tmp_path / "task" / "attempt_1.py").read_text() == WRONG_SIGN
    assert (tmp_path / "task" / "attempt_2.py").read_text() == GOOD


def test_three_failures_give_the_honest_result(wonder, tmp_path):
    wonder.sdk().script.extend([answer(WRONG_SIGN)] * 3)
    out = run_solve(wonder, tmp_path, n=4)
    assert out["outcome"] == "failed" and out["message"] == FAILED_MESSAGE
    assert len(out["attempts"]) == 3 and out["best_attempt"] == 1 and len(wonder.sdk().calls) == 3
    assert not (tmp_path / "task" / "controller.py").exists()


def test_infeasible_task_is_reported_without_a_call(wonder, tmp_path):
    # Holding 70 degrees needs m g L sin(1.2) = 5.5 N m per kg; the motor gives 1.
    task = HOLD.model_copy(update={"target": 1.2, "actuator_limit": 1.0})
    out = run_solve(wonder, tmp_path, task=task, n=3)
    assert out["outcome"] == "infeasible" and "m g L sin(target)" in out["infeasibility_note"]
    assert out["message"].startswith(FAILED_MESSAGE) and wonder.sdk().calls == []


def test_budget_cap_means_no_call(settings, tmp_path):
    path = settings.config_dir / "budget.yaml"
    budget = yaml.safe_load(path.read_text())
    budget.update(model_daily_caps_usd={"muse-wonder-1.0": 0.001}, control_task_cap_usd=0.60)
    path.write_text(yaml.safe_dump(budget))
    h = Harness(settings)
    with pytest.raises(BudgetExceeded, match="muse-wonder-1.0 daily budget"):
        run_solve(h, tmp_path, n=2)
    assert h.sdk().calls == []


def test_endless_loop_is_killed_and_this_process_carries_on():
    loop = "def control(t, q, qdot, params):\n    while True:\n        t = t + 1.0\n"
    plants = build_plants(FS, HOLD, variants(2))
    started = time.monotonic()
    res = run_controller(loop, evaluation_job(FS, HOLD, plants), SandboxLimits(cpu_seconds=2, wall_seconds=10))
    assert time.monotonic() - started < 10
    assert all(not s["success"] and "CPU time limit" in s["error"] for s in res)
    assert summarize(run_controller(GOOD, evaluation_job(FS, HOLD, plants), FAST), 0.9)["passed"]


def test_huge_allocation_hits_the_memory_limit():
    # About 3.2 GB: fine for an unlimited process on a big machine, refused under the 1.5 GB address-space limit.
    hog = "import numpy as np\n\ndef control(t, q, qdot, params):\n    big = np.ones(400000000)\n    return [float(big[0])]\n"
    res = run_controller(hog, evaluation_job(FS, HOLD, build_plants(FS, HOLD, variants(1))), FAST)
    assert all(not s["success"] and "MemoryError" in s["error"] for s in res)


def test_controller_output_cannot_fake_the_metrics():
    fake = 'def control(t, q, qdot, params):\n    print(\'{"samples": []}\')\n    return [0.0]\n'
    res = run_controller(fake, evaluation_job(FS, HOLD, build_plants(FS, HOLD, variants(1))), FAST)
    assert len(res) == 2 and not res[0]["success"] and "print" in res[0]["error"]  # print is not even available


def test_command_is_clamped_and_saturation_reported():
    bang = "def control(t, q, qdot, params):\n    return [100.0]\n"
    res = run_controller(bang, evaluation_job(FS, HOLD, build_plants(FS, HOLD, variants(0))), FAST)
    assert res[0]["peak_u"] == pytest.approx(HOLD.actuator_limit) and res[0]["saturation"] == 1.0


# ---------- determinism, support, the API ----------

@pytest.fixture(scope="module")
def measured():
    """A real stored result (pixel track and fit), for the bootstrap variants."""
    traj = pendulum([(4, L, 0.3)], theta0=0.8)
    f = fit_model(MODELS["pendulum_linear_damping"], to_coordinates(traj, "pendulum", GEOMETRY))
    fs = {"model_id": f.model_id, "params": f.params, "y0": [float(v) for v in f.y0], "ctx": dict(GEOMETRY),
          "fps": float(FPS), "origin_px": [320.0, 80.0], "px_per_m": 400.0, "first_frame": 0}
    return stored_result(traj, fs)


def test_same_seed_same_variants_and_metrics(measured):
    a = robustness_params(measured, 3, seed=7)
    assert a == robustness_params(measured, 3, seed=7) and a != robustness_params(measured, 3, seed=8)
    assert len(a) == 4 and a[0]["g"] == pytest.approx(measured["fit_state"]["params"]["g"])
    job = evaluation_job(measured["fit_state"], HOLD, build_plants(measured["fit_state"], HOLD, a))
    assert run_controller(GOOD, job, FAST) == run_controller(GOOD, job, FAST)


def test_unsupported_families_and_mismatches_are_honest():
    with pytest.raises(ControlUnsupported, match="not supported yet"):
        check_supported({**RESULT, "fit_state": {**FS, "model_id": "spin_free", "params": {}}},
                        ControlTask(family="rotation", goal="stop", tolerance=0.1, actuator_limit=1, time_limit_s=3))
    with pytest.raises(ControlUnsupported, match="identified a pendulum"):
        check_supported(RESULT, HOLD.model_copy(update={"family": "oscillator"}))
    with pytest.raises(ControlUnsupported, match="accepted model"):
        check_supported({**RESULT, "status": "ambiguous"}, HOLD)


def test_endpoint_access_background_run_and_free_rerun(measured, wonder, monkeypatch, tmp_path):
    from dataclasses import replace

    from fastapi.testclient import TestClient
    from physicslens import app as app_module

    wonder.client.limits["control"] = {**wonder.client.limits["control"], "bootstrap_samples": 3,
                                       "sandbox_cpu_seconds": 20}
    manager = ControlManager(wonder.client, tmp_path / "runs")
    monkeypatch.setattr(app_module, "controls", manager)
    run_dir = app_module.runs.new_run_dir()
    (run_dir / "result.json").write_text(json.dumps(measured))
    http = TestClient(app_module.app)
    url = f"/api/runs/{run_dir.name}/control"
    body = HOLD.model_dump()

    monkeypatch.setattr(app_module, "settings", replace(app_module.settings, app_env="demo"))
    assert http.post(url, json=body).status_code == 403  # public visitors: no code, no Wonder
    monkeypatch.setattr(app_module, "settings", replace(app_module.settings, app_env="dev"))

    rotation = {**body, "family": "rotation"}
    assert http.post(url, json=rotation).status_code == 409
    wonder.sdk().script.append(answer(GOOD))
    first = http.post(url, json=body).json()
    task_id = first["task_id"]
    for _ in range(300):
        view = http.get(f"{url}/{task_id}").json()
        if view["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert view["status"] == "done", view
    assert view["result"]["outcome"] == "accepted" and view["result"]["samples"] == 4
    assert (tmp_path / "runs" / run_dir.name / "control" / "samples-seed7-n3.json").is_file()
    calls = len(wonder.sdk().calls)
    again = http.post(url, json=body).json()
    assert again["status"] == "done" and again["task_id"] == task_id and len(wonder.sdk().calls) == calls


def test_unusable_answer_counts_as_an_attempt(wonder, tmp_path):
    wonder.sdk().script.extend([responses_result(text="not json"), answer(GOOD)])
    out = run_solve(wonder, tmp_path, n=3)
    assert out["attempts"][0]["check"]["ok"] is False and out["accepted_attempt"] == 2
    assert "not JSON" in wonder.sdk().calls[1]["input"][0]["content"][0]["text"]
