"""The agent's tool steps: it may test more models or inspect part of the motion; only code changes the verdict."""

import json

import yaml
from physicslens.costguard.mock import responses_result
from physicslens.pipeline import RunOptions, analyze

from .conftest import Harness
from .test_pipeline import (
    CLIP,
    SCENE,
    SKEPTIC_NONE,
    TRACK_OK,
    sam_result,
    sam_text_as_object_1,
    script,
)

NARROW = {"family": "pendulum", "free_models": [], "candidates": [{"model_id": "pendulum_small_angle", "rationale": "r"}]}
EXPLAIN = {"action": "explain", "message": "Done.", "next_family": None, "reasoning_summary": "s"}


def tool(action, **kw):
    return responses_result(text=json.dumps({"action": action, "message": f"Checking with {action}.", "next_family": None,
                                             "reasoning_summary": "s", "expectation": "The full pendulum should do better.",
                                             **kw}))


def run(h, tmp_path, *agent_replies):
    script(h, responses_result(text=json.dumps(SCENE)), sam_result(frames=60, text=sam_text_as_object_1()),
           responses_result(text=json.dumps(TRACK_OK)), responses_result(text=json.dumps(NARROW)), *agent_replies)
    return analyze(h.client, CLIP, tmp_path / "run", "run-tools", RunOptions(pendulum_length_m=0.6))


def agent_inputs(h):
    return [json.loads(c["input"][0]["content"][0]["text"]) for c in h.sdk().calls
            if c.get("kind") == "responses" and "tool_steps_left" in c["input"][0]["content"][0]["text"]]


def test_tested_models_are_fitted_and_the_result_reaches_the_next_turn(harness, tmp_path):
    finals = [responses_result(text=json.dumps(EXPLAIN))]  # a final action that mismatches the verdict is only noted
    result = run(harness, tmp_path, tool("test_models", model_ids=["pendulum_linear_damping"]), *finals,
                 responses_result(text=json.dumps(SKEPTIC_NONE)))
    step = next(t for t in result["trail"] if t.get("action") == "test_models")
    item = step["items"][0]
    assert step["expectation"] and item["model_id"] == "pendulum_linear_damping" and item["status"] != "untestable"
    assert "pendulum_linear_damping" in {r["model_id"] for r in result["ranking"]}  # really fitted and ranked
    first, second = agent_inputs(harness)[:2]
    assert first["tool_steps_left"] == 4 and second["tool_steps_left"] == 3
    assert second["notebook"][0]["tested"] == ["pendulum_linear_damping"]
    assert "pendulum_linear_damping" not in {m["model_id"] for m in second["untested_models"]}


def test_inspect_returns_numbers_for_the_window(harness, tmp_path):
    finals = [responses_result(text=json.dumps(EXPLAIN))]  # a final action that mismatches the verdict is only noted
    result = run(harness, tmp_path, tool("inspect", window={"start_s": 0.2, "end_s": 1.2, "reason": "early swings"}),
                 *finals, responses_result(text=json.dumps(SKEPTIC_NONE)))
    assert any(t.get("action") == "inspect" and "0.20 to 1.20 s" in t["text"] for t in result["trail"])
    entry = agent_inputs(harness)[1]["notebook"][0]
    assert entry["window"] == [0.2, 1.2] and entry["numbers"]


def test_step_budget_is_enforced_by_code(settings, tmp_path):
    path = settings.config_dir / "limits.yaml"
    limits = yaml.safe_load(path.read_text())
    limits["agent"]["max_tool_steps"] = 1
    path.write_text(yaml.safe_dump(limits))
    h = Harness(settings)
    result = run(h, tmp_path, tool("test_models", model_ids=["pendulum_quadratic_damping"]),
                 tool("test_models", model_ids=["pendulum_linear_damping"]),
                 responses_result(text=json.dumps(SKEPTIC_NONE)))
    texts = [t["text"] for t in result["trail"] if t["step"] == "agent"]
    assert any("step budget is used up" in t for t in texts)
    assert "pendulum_linear_damping" not in {r["model_id"] for r in result["ranking"]}  # the over-budget test never ran
    assert result["agent"]["action"] in ("explain", "request_experiment", "give_up")  # final action follows the verdict


def test_models_from_another_family_are_untestable(harness, tmp_path):
    finals = [responses_result(text=json.dumps(EXPLAIN))]  # a final action that mismatches the verdict is only noted
    result = run(harness, tmp_path, tool("test_models", model_ids=["damped_harmonic"]), *finals,
                 responses_result(text=json.dumps(SKEPTIC_NONE)))
    step = next(t for t in result["trail"] if t.get("action") == "test_models")
    assert step["items"][0]["status"] == "untestable" and "verdict is unchanged" in step["text"]
