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
    "observations": "A red ball swings on a string.", "second_object": None, "second_object_box": None,
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
    assert len(result["mujoco"]["points"]) == 60
    assert result["mujoco"]["rms_vs_fit_m"] is not None
    trail = result["trail"]
    expected = ["scene", "tracking", "hypotheses", "verification", "agent"]
    if result["status"] == "ambiguous":  # a tie always comes with a simulated experiment design
        expected.insert(4, "design")
        assert result["experiment_design"]["recommended"]["label"]
    assert [t["step"] for t in trail] == expected
    assert all(t["model"] for t in trail)
    verification = trail[3]
    assert {i["model_id"] for i in verification["items"]} == {f["model_id"] for f in result["ranking"]}
    assert all(i["status"] in ("accepted", "tied", "rejected") and i["why"] for i in verification["items"])
    assert steps[-1] == "Done"

    # Shareable video: same length as the clip, H.264, and the ghost really drawn where the simulation says.
    from physicslens.export.mp4 import render_share_video
    share = render_share_video(result, tmp_path / "run" / "clip.mp4", tmp_path / "run" / "share.mp4")
    import subprocess
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-show_entries", "stream=codec_name,nb_read_packets",
                            "-of", "csv=p=0", str(share)], capture_output=True, text=True, check=True).stdout.strip()
    assert probe == "h264,60"
    import cv2
    cap = cv2.VideoCapture(str(share))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ok, frame = cap.read()
    _, gx, gy = next(p for p in result["ghosts"][0]["points"] if p[0] == 30)
    r = int(result["object_size_px"] / 2)
    b, g, red = (int(v) for v in frame[round(gy), round(gx) + r])  # outline at angle 0 (inside the first dash)
    assert ok and b > 150 and g > 150 and red < 120

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
    schema = strict_schema(Scene)
    assert schema["additionalProperties"] is False
    # Every required field must actually be defined (an empty "properties" made the API return 500).
    assert set(schema["required"]) == set(schema["properties"]) == set(Scene.model_fields)
    box = schema["properties"]["moving_object_box"]
    assert set(box["properties"]) == {"x1", "y1", "x2", "y2"} and box["additionalProperties"] is False


def test_invalid_json_retried_once_then_fails(harness):
    script(harness, responses_result(text="not json"), responses_result(text="{}"))
    with pytest.raises(LLMError):
        ask_json(harness.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r")
    assert len(harness.sdk("meta_api").calls) == 2


# ---------- Open Weight: no silent Meta fallback outside dev ----------

def demo_harness(settings):
    from physicslens.settings import Settings

    from .conftest import Harness
    demo = Settings(app_env="demo", repo_root=settings.repo_root, config_dir=settings.config_dir, var_dir=settings.var_dir)
    return Harness(demo)


def test_demo_refuses_meta_fallback_when_open_weight_missing(settings, monkeypatch):
    from physicslens.agent.llm import OpenWeightUnavailable
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    h = demo_harness(settings)
    with pytest.raises(OpenWeightUnavailable):
        ask_json(h.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r", audience="judge")
    assert h.sdk("meta_api").calls == []


def test_demo_does_not_fall_back_when_baseten_fails(settings, monkeypatch):
    from physicslens.agent.llm import OpenWeightUnavailable
    from physicslens.costguard.mock import api_error

    from .conftest import set_baseten_price
    set_baseten_price(settings)
    monkeypatch.setenv("BASETEN_API_KEY", "test-not-a-real-key")
    h = demo_harness(settings)
    h.sdk("baseten").script.extend([api_error(503)] * 3)
    with pytest.raises(OpenWeightUnavailable):
        ask_json(h.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r", audience="judge")
    assert h.sdk("meta_api").calls == []


def test_dev_falls_back_to_meta_when_baseten_fails(harness, settings, monkeypatch):
    from physicslens.costguard.mock import api_error

    from .conftest import Harness, set_baseten_price
    set_baseten_price(settings)
    monkeypatch.setenv("BASETEN_API_KEY", "test-not-a-real-key")
    h = Harness(settings)
    h.sdk("baseten").script.extend([api_error(503)] * 3)
    h.sdk("meta_api").script.append(responses_result(text=json.dumps(SCENE)))
    scene, call = ask_json(h.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r")
    assert call.provider == "meta_api" and scene.moving_object == "red ball"


def test_baseten_used_when_configured(settings, monkeypatch):
    from physicslens.costguard.mock import chat_result

    from .conftest import set_baseten_price
    set_baseten_price(settings)
    monkeypatch.setenv("BASETEN_API_KEY", "test-not-a-real-key")
    h = demo_harness(settings)
    h.sdk("baseten").script.append(chat_result(text=json.dumps(SCENE)))
    _, call = ask_json(h.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r", audience="judge")
    assert call.provider == "baseten" and call.model == "meta-models/muse-spark-open-1026"
    assert h.sdk("baseten").calls[0]["reasoning_effort"] == "low"
