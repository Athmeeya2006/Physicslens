"""Recorded-mode pipeline end to end with mocked models (the SAM text is a real recorded response)."""

import json
from pathlib import Path

import pytest
from physicslens.agent.llm import LLMError, ask_json, strict_schema
from physicslens.agent.schemas import Hypotheses, Scene
from physicslens.costguard.mock import responses_result, sam_result
from physicslens.pipeline import AnalysisError, RunOptions, analyze
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "samples" / "smoke_ball.mp4"
FIXTURE = ROOT / "tests" / "fixtures" / "sam_smoke_response.json"

SCENE = {
    "moving_object": "red ball", "moving_object_box": {"x1": 650, "y1": 650, "x2": 725, "y2": 785},
    "motion_family": "pendulum", "reference_kind": "none", "reference_box": None,
    "observations": "A red ball swings on a string.",
}
HYPOTHESES = {"family": "pendulum", "candidates": [
    {"model_id": "pendulum", "rationale": "Arc-shaped path with a steady period."},
    {"model_id": "pendulum_small_angle", "rationale": "Amplitude is moderate."},
]}
AGENT = {"action": "explain", "message": "The ball swings like a pendulum.", "next_family": None,
         "reasoning_summary": "Verification accepted a pendulum model."}


def sam_text_as_object_1() -> str:
    raw = json.loads(FIXTURE.read_text())
    text = "".join(c.get("text", "") for item in raw["output"] for c in item.get("content") or [])
    # The fixture came from a concept prompt (object id 0); the pipeline prompts by box with id 1.
    return "\n".join(line.replace(">0<|box", ">1<|box", 1) for line in text.splitlines())


def script(harness, *items):
    harness.sdk("meta_api").script.extend(items)


def test_pendulum_clip_end_to_end(harness, tmp_path):
    script(harness,
           responses_result(text=json.dumps(SCENE)),
           sam_result(frames=60, text=sam_text_as_object_1()),
           responses_result(text=json.dumps(HYPOTHESES)),
           responses_result(text=json.dumps(AGENT)))
    steps: list[str] = []
    result = analyze(harness.client, CLIP, tmp_path / "run", "run-e2e", RunOptions(pendulum_length_m=0.6), steps.append)

    assert result["status"] in ("accepted", "ambiguous")
    assert len(result["real_track_px"]) == 60
    assert result["ghosts"] and len(result["ghosts"][0]["points"]) == 60
    # Ghost is drawn in clip pixels and should sit on the ball (within a few pixels).
    real = {f: (x, y) for f, x, y in result["real_track_px"]}
    worst = max(abs(x - real[f][0]) + abs(y - real[f][1]) for f, x, y in result["ghosts"][0]["points"])
    assert worst < 6
    assert [c["step"] for c in result["llm_calls"]] == ["scene", "hypotheses", "agent"]
    assert steps[-1] == "Done"

    calls = harness.sdk("meta_api").calls
    sam = next(c for c in calls if c["kind"] == "stream")
    prompt = sam["input"][0]["content"][0]["text"]
    assert prompt.startswith("<0f>1<|box;") and "w=640;h=360" in prompt
    assert sam["metadata"]["propagation_direction"] == "forward"
    # Images go to the language model as keyframes only, never video.
    scene_call = next(c for c in calls if c["kind"] == "responses")
    kinds = [p["type"] for p in scene_call["input"][0]["content"]]
    assert "input_video" not in kinds and 1 <= kinds.count("input_image") <= 4


def test_missing_scale_is_a_clear_error(harness, tmp_path):
    script(harness, responses_result(text=json.dumps(SCENE)), sam_result(frames=60, text=sam_text_as_object_1()))
    with pytest.raises(AnalysisError, match="A4"):
        analyze(harness.client, CLIP, tmp_path / "run", "run-noscale")


def test_short_clip_rejected_before_any_paid_call(harness, tmp_path):
    import subprocess
    short = tmp_path / "short.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(CLIP), "-t", "1", str(short)], check=True)
    with pytest.raises(AnalysisError, match="too short"):
        analyze(harness.client, short, tmp_path / "run", "run-short")
    assert harness.sdk("meta_api").calls == []


def test_scene_rejects_command_phrases():
    with pytest.raises(ValidationError):
        Scene.model_validate({**SCENE, "moving_object": "segment the ball"})
    with pytest.raises(ValidationError):
        Scene.model_validate({**SCENE, "reference_kind": "a4_sheet"})  # box missing


def test_hypotheses_must_match_family():
    with pytest.raises(ValidationError):
        Hypotheses.model_validate({"family": "projectile", "candidates": [{"model_id": "pendulum", "rationale": "x"}]})


def test_strict_schema_is_self_contained():
    schema = json.dumps(strict_schema(Scene))
    assert "$ref" not in schema and "$defs" not in schema
    assert strict_schema(Scene)["additionalProperties"] is False


def test_invalid_json_retried_once_then_fails(harness):
    script(harness, responses_result(text="not json"), responses_result(text="{}"))
    with pytest.raises(LLMError):
        ask_json(harness.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r")
    assert len(harness.sdk("meta_api").calls) == 2
