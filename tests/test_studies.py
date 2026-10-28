"""Goal-driven studies: the agent plans and reviews, the math combines and judges agreement."""

import json

import pytest
from physicslens.costguard.mock import responses_result
from physicslens.physics.combine import combine
from physicslens.studies import Studies, StudyError

PLAN = {"deliverable": "simulation_model", "quantities": ["natural frequency", "damping ratio"],
        "recordings": [{"label": "Push 1", "how_to_film": "Phone on a tripod, side-on."},
                       {"label": "Push 2", "how_to_film": "Same again."}],
        "done_when": "Two recordings agree within their error bars.", "reasoning_summary": "Repeat to trust the number."}
MORE = {"action": "record_more", "next_recording": {"label": "Push 2", "how_to_film": "Same again."},
        "message": "One more push to confirm the damping.", "maintenance_note": None, "next_check": None,
        "reasoning_summary": "Only one recording so far."}
FINISH = {**MORE, "action": "finish", "next_recording": None, "message": "Both pushes agree; the model is ready."}


def result(zeta: float, zeta_std: float, status: str = "accepted") -> dict:
    return {"status": status, "family": "oscillator", "reason": "ok", "equation": {"title": "Damped harmonic oscillator"},
            "parameters": [{"name": "omega", "latex": r"\omega", "unit": "rad/s", "value": 8.0, "std": 0.05, "ci95": [7.9, 8.1]},
                           {"name": "zeta", "latex": r"\zeta", "unit": "", "value": zeta, "std": zeta_std, "ci95": [0, 1]}],
            "ranking": [{"heldout_nrmse": 0.05}],
            "fit_state": {"model_id": "damped_harmonic", "params": {"omega": 8.0, "s0": 0.0, "zeta": zeta},
                          "y0": [0.02, 0.0], "ctx": {"origin_x": 0, "origin_y": 0, "axis_x": 0, "axis_y": 1}, "fps": 30.0,
                          "origin_px": [100.0, 100.0], "px_per_m": 1000.0, "first_frame": 0}}


def test_combine_weights_and_tests_agreement():
    same = combine([("a", result(0.20, 0.01)), ("b", result(0.22, 0.01))])
    z = next(c for c in same if c["name"] == "zeta")
    assert z["value"] == pytest.approx(0.21) and z["n"] == 2 and z["consistent"]
    apart = next(c for c in combine([("a", result(0.20, 0.005)), ("b", result(0.30, 0.005))]) if c["name"] == "zeta")
    assert not apart["consistent"] and apart["p_value"] < 0.01
    assert combine([("a", result(0.2, 0.01, status="no_fit"))]) == []


def test_study_plans_reviews_and_finishes(harness, settings):
    results = {}
    studies = Studies(settings.var_dir / "studies", harness.client)
    harness.sdk("meta_api").script.append(responses_result(text=json.dumps(PLAN)))
    s = studies.create("Get the damping of my car's front left shock for a simulation", "dev")
    assert s["plan"]["deliverable"] == "simulation_model" and s["trail"][0]["step"] == "plan"

    studies.attach(s["id"], "run1", "Push 1")
    assert studies.review(s["id"], results.get, lambda ids: None, "dev")["turns"] == []  # still analyzing
    results["run1"] = result(0.20, 0.01)
    harness.sdk("meta_api").script.append(responses_result(text=json.dumps(MORE)))
    s = studies.review(s["id"], results.get, lambda ids: None, "dev")
    assert s["status"] == "recording" and s["turns"][-1]["action"] == "record_more"

    studies.attach(s["id"], "run2", "Push 2")
    results["run2"] = result(0.22, 0.01)
    harness.sdk("meta_api").script.append(responses_result(text=json.dumps(FINISH)))
    s = studies.review(s["id"], results.get, lambda ids: None, "dev")
    assert s["status"] == "finished" and s["trail"][-1]["step"] == "study_turn"
    zeta = next(c for c in s["combined"] if c["name"] == "zeta")
    assert zeta["value"] == pytest.approx(0.21)
    xml = studies.mjcf(s["id"], results.get)
    assert "combined over 2 recording(s)" in xml
    md = studies.report(s["id"], results.get)
    assert "## Combined" in md and "consistent" in md and "Both pushes agree" in md
    with pytest.raises(StudyError):
        studies.attach(s["id"], "run3", "late")


def test_agent_cannot_finish_without_a_verified_recording(harness, settings):
    studies = Studies(settings.var_dir / "studies", harness.client)
    harness.sdk("meta_api").script.append(responses_result(text=json.dumps(PLAN)))
    s = studies.create("Is this fan's bearing wearing out over the month?", "dev")
    studies.attach(s["id"], "bad", "Week 1")
    harness.sdk("meta_api").script.append(responses_result(text=json.dumps(FINISH)))
    s = studies.review(s["id"], {"bad": result(0.2, 0.01, status="no_fit")}.get, lambda ids: None, "dev")
    assert s["status"] == "recording" and "overruled" in s["turns"][-1]


def test_health_study_gets_a_wear_verdict(harness, settings):
    studies = Studies(settings.var_dir / "studies", harness.client)
    plan = {**PLAN, "deliverable": "health_check"}
    harness.sdk("meta_api").script.append(responses_result(text=json.dumps(plan)))
    s = studies.create("Is this suspension wearing out between checks?", "dev")
    for rid in ("w1", "w2"):
        studies.attach(s["id"], rid, rid)
    seen = []
    note = {**FINISH, "maintenance_note": "Damping dropped; inspect the shock.", "next_check": "in two weeks"}
    harness.sdk("meta_api").script.append(responses_result(text=json.dumps(note)))
    s = studies.review(s["id"], {"w1": result(0.20, 0.005), "w2": result(0.12, 0.005)}.get,
                       lambda ids: seen.append(ids) or {"drift": True, "summary": "Damping fell 40%."}, "dev")
    assert seen == [["w1", "w2"]] and s["health"]["drift"]
    md = studies.report(s["id"], {"w1": result(0.20, 0.005), "w2": result(0.12, 0.005)}.get)
    assert "Wear check" in md and "inspect the shock" in md and "in two weeks" in md


def test_studies_need_the_access_code_in_demo(settings, monkeypatch):
    from fastapi.testclient import TestClient
    from physicslens import app as app_module

    monkeypatch.setattr(app_module, "audience_for", lambda code: "public")
    r = TestClient(app_module.app).post("/api/studies", json={"goal": "Measure my door hinge damping please"})
    assert r.status_code == 403
