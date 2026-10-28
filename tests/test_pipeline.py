"""Recorded-mode pipeline end to end with mocked models (the SAM text is a real recorded response)."""

import json
from pathlib import Path

import numpy as np
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
], "free_models": []}
TRACK_OK = {"on_target": True, "tracked_thing": "the red ball", "reasoning_summary": "The box surrounds the ball in all frames."}
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
           responses_result(text=json.dumps(TRACK_OK)),
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
    assert [c["step"] for c in result["llm_calls"]] == ["scene", "track_check", "hypotheses", "agent"]
    assert len(result["mujoco"]["points"]) == 60
    assert result["mujoco"]["rms_vs_fit_m"] is not None
    trail = result["trail"]
    expected = ["scene", "tracking", "track_check", "hypotheses", "verification", "agent"]
    if result["status"] == "ambiguous":  # a tie always comes with a simulated experiment design
        expected.insert(5, "design")
        assert result["experiment_design"]["recommended"]["label"]
    assert [t["step"] for t in trail] == expected
    fc = result["forecast"]  # the verification, as plottable data
    assert len(fc["t"]) == len(fc["measured"]) == len(fc["forecast"]) and fc["t"][0] < fc["heldout_start"] < fc["t"][-1]
    assert all(t["model"] for t in trail)
    verification = trail[4]
    assert {i["model_id"] for i in verification["items"]} == {f["model_id"] for f in result["ranking"]}
    assert all(i["status"] in ("accepted", "tied", "rejected", "approximation") and i["why"] for i in verification["items"])
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
    b, g, red = (int(v) for v in frame[round(gy), round(gx)])  # the ghost's centre dot
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


def test_missing_scale_still_analyzes_scale_free_quantities(harness, tmp_path):
    script(harness,
           responses_result(text=json.dumps(SCENE)),
           sam_result(frames=60, text=sam_text_as_object_1()),
           responses_result(text=json.dumps(TRACK_OK)),
           responses_result(text=json.dumps(HYPOTHESES)),
           responses_result(text=json.dumps(AGENT)))
    result = analyze(harness.client, CLIP, tmp_path / "run", "run-noscale")  # no A4 sheet, no length given
    if result["status"] == "accepted":
        # The verified pendulum law contains g, so Earth's gravity sets the scale and the string length follows.
        assert result["calibration"]["source"] == "gravity" and "scale_note" not in result
        params = {p["name"]: p for p in result["parameters"]}
        assert params["g"]["value"] == 9.81 and "assumed" in params["g"]["derived"]
        omega = 2 * np.pi * result["summary"]["dominant_frequency_hz"]
        assert params["L"]["value"] == pytest.approx(9.81 / omega**2, rel=0.1)
        assert any(t["step"] == "plausibility" for t in result["trail"])
    else:
        assert result["calibration"]["source"] == "none" and "arbitrary units" in result["scale_note"]


def test_unit_scale_detection():
    from physicslens.pipeline import needs_scale
    assert all(needs_scale(u) for u in ("m", "m/s^2", "1/m"))
    assert not any(needs_scale(u) for u in ("1/s", "rad/s", "rad/s^2", "1/rad", ""))


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
        Hypotheses.model_validate({"family": "projectile", "candidates": [{"model_id": "pendulum", "rationale": "x"}],
                                   "free_models": []})


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
    monkeypatch.delenv("REASONING_PROVIDER", raising=False)
    h = demo_harness(settings)
    with pytest.raises(OpenWeightUnavailable):
        ask_json(h.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r", audience="judge")
    assert h.sdk("meta_api").calls == []


def test_demo_does_not_fall_back_when_glimmer_fails(settings, monkeypatch):
    from physicslens.agent.llm import OpenWeightUnavailable
    from physicslens.costguard.mock import api_error

    monkeypatch.setenv("REASONING_PROVIDER", "glimmer")
    monkeypatch.setenv("GLIMMER_BASE_URL", "http://127.0.0.1:8080/v1")
    h = demo_harness(settings)
    h.sdk("glimmer").script.extend([api_error(503)] * 3)
    with pytest.raises(OpenWeightUnavailable):
        ask_json(h.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r", audience="judge")
    assert h.sdk("meta_api").calls == []


def test_dev_uses_meta_when_no_open_weight_server(harness, monkeypatch):
    monkeypatch.delenv("REASONING_PROVIDER", raising=False)
    harness.sdk("meta_api").script.append(responses_result(text=json.dumps(SCENE)))
    scene, call = ask_json(harness.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r")
    assert call.provider == "meta_api" and scene.moving_object == "red ball"


def test_slow_motion_factor_puts_the_physics_in_real_time(harness, tmp_path):
    """The same frames declared as 1.2x slow motion: real time is shorter by 1.2, so g comes out 1.44 times larger."""
    out = {}
    script(harness, responses_result(text=json.dumps(SCENE)), sam_result(frames=60, text=sam_text_as_object_1()),
           responses_result(text=json.dumps(TRACK_OK)))
    for factor in (1.0, 1.2):
        # Scene and tracking replay from the cache the second time; the time-scaled summary is a new prompt.
        script(harness, responses_result(text=json.dumps(HYPOTHESES)), responses_result(text=json.dumps(AGENT)))
        out[factor] = analyze(harness.client, CLIP, tmp_path / f"run{factor}", f"run-slow-{factor}",
                              RunOptions(pendulum_length_m=0.6, slow_motion=factor))
    g1 = out[1.0]["ranking"][0]
    g2 = next(f for f in out[1.2]["ranking"] if f["model_id"] == g1["model_id"])
    assert g2["params"]["g"] == pytest.approx(1.44 * g1["params"]["g"], rel=0.02)
    assert out[1.2]["clip"]["fps"] == out[1.0]["clip"]["fps"]  # drawing on the video keeps the file's rate
    assert out[1.2]["forecast"]["t"][-1] == pytest.approx(out[1.0]["forecast"]["t"][-1] / 1.2, rel=0.02)


def test_scale_error_widens_only_length_parameters():
    from physicslens.pipeline import add_scale_error

    params = [{"name": "g", "unit": "m/s^2", "value": 9.8, "std": 0.05, "ci95": [9.7, 9.9]},
              {"name": "zeta", "unit": "", "value": 0.02, "std": 0.001, "ci95": [0.018, 0.022]},
              {"name": "L", "unit": "m", "value": 0.5, "std": 0.0, "ci95": [0.5, 0.5], "derived": "given by you"}]
    add_scale_error(params, 0.02)
    g, zeta, L = params
    assert g["std"] == pytest.approx(np.hypot(0.05, 9.8 * 0.02)) and g["includes_scale_error"]
    assert g["ci95"][0] < 9.7 and g["ci95"][1] > 9.9
    assert zeta["std"] == 0.001 and "includes_scale_error" not in zeta
    assert L["std"] == 0.0  # a length the user gave is not re-measured


def test_gravity_is_the_ruler_when_nothing_else_gives_a_scale():
    from physicslens.pipeline import apply_gravity_ruler

    result = {
        "family": "projectile", "trail": [],
        "parameters": [{"name": "g", "unit": "m/s^2", "value": 3.5, "std": 0.07, "ci95": [3.36, 3.64]},
                       {"name": "k", "unit": "1/s", "value": 2.6, "std": 0.07, "ci95": [2.46, 2.74]}],
        "ranking": [{"model_id": "linear_drag", "params": {"g": 3.5, "k": 2.6},
                     "param_meta": [{"name": "g", "unit": "m/s^2"}, {"name": "k", "unit": "1/s"}]}],
        "fit_state": {"model_id": "linear_drag", "params": {"g": 3.5, "k": 2.6}, "y0": [0.0, 0.0, 1.0, 2.0],
                      "ctx": {}, "px_per_m": 1000.0},
        "forecast": {"unit": "m", "measured": [0.1, 0.2], "forecast": [0.1, 0.2]},
    }
    cal = apply_gravity_ruler(result)
    f = 9.81 / 3.5
    g, k = result["parameters"]
    assert g["value"] == 9.81 and "assumed" in g["derived"]
    assert k["value"] == 2.6 and k["std"] == 0.07  # a rate needs no scale
    assert result["ranking"][0]["params"]["g"] == pytest.approx(9.81)
    assert result["fit_state"]["y0"] == pytest.approx([0.0, 0.0, f, 2 * f])
    assert result["fit_state"]["px_per_m"] == pytest.approx(1000.0 / f)
    assert result["forecast"]["measured"] == pytest.approx([0.1 * f, 0.2 * f])
    assert cal.source == "gravity" and cal.rel_std == pytest.approx(0.02)
    assert result["trail"][-1]["step"] == "plausibility" and "Earth" in result["scale_basis"]


def test_wrong_track_is_repaired_by_asking_sam_by_name(harness, tmp_path):
    """The reasoning model sees the box is on the wrong thing; SAM is asked again by the object's name."""
    off = {"on_target": False, "tracked_thing": "the clasp above the ball", "reasoning_summary": "The box sits above it."}
    script(harness,
           responses_result(text=json.dumps(SCENE)),
           sam_result(frames=60, text=sam_text_as_object_1()),
           responses_result(text=json.dumps(off)),
           sam_result(frames=60, text=sam_text_as_object_1()),  # the concept-prompt request
           responses_result(text=json.dumps(HYPOTHESES)),
           responses_result(text=json.dumps(AGENT)))
    result = analyze(harness.client, CLIP, tmp_path / "run", "run-repair", RunOptions(pendulum_length_m=0.6))
    sams = [c for c in harness.sdk("meta_api").calls if c["kind"] == "stream"]
    assert len(sams) == 2
    named = sams[1]["input"][0]["content"][0]["text"]
    assert named == SCENE["moving_object"] and "propagation_direction" not in sams[1]["metadata"]
    steps = [t["step"] for t in result["trail"]]
    assert steps[:4] == ["scene", "tracking", "track_check", "tracking"] and "by name" in result["trail"][3]["text"]
    assert result["trail"][2]["status"] == "rejected"


def test_track_check_failure_keeps_the_first_track(harness, tmp_path):
    script(harness,
           responses_result(text=json.dumps(SCENE)),
           sam_result(frames=60, text=sam_text_as_object_1()),
           responses_result(text="not json"), responses_result(text="still not json"),
           responses_result(text=json.dumps(HYPOTHESES)),
           responses_result(text=json.dumps(AGENT)))
    result = analyze(harness.client, CLIP, tmp_path / "run", "run-nocheck", RunOptions(pendulum_length_m=0.6))
    assert [t["step"] for t in result["trail"]][:3] == ["scene", "tracking", "hypotheses"]


def test_schema_words_echoed_next_to_a_valid_answer_are_dropped(harness):
    echoed = {**SCENE, "type": "object", "required": list(SCENE), "additionalProperties": False}
    harness.sdk("meta_api").script.append(responses_result(text=json.dumps(echoed)))
    scene, _ = ask_json(harness.client, task="scene", system="s", text="t", schema_model=Scene, run_id="r")
    assert scene.moving_object == SCENE["moving_object"] and len(harness.sdk("meta_api").calls) == 1  # no retry


@pytest.mark.parametrize(("window", "used"), [((0.2, 1.9), True), ((0.0, 0.8), False)])
def test_agent_fit_window_is_guarded_and_the_whole_clip_is_kept(harness, tmp_path, window, used):
    """The agent may leave out a throw or a catch, but never most of the clip, and the whole-clip verdict stays."""
    hyp = {**HYPOTHESES, "fit_window": {"start_s": window[0], "end_s": window[1], "reason": "a hand holds it at first"}}
    script(harness,
           responses_result(text=json.dumps(SCENE)),
           sam_result(frames=60, text=sam_text_as_object_1()),
           responses_result(text=json.dumps(TRACK_OK)),
           responses_result(text=json.dumps(hyp)),
           responses_result(text=json.dumps(AGENT)))
    result = analyze(harness.client, CLIP, tmp_path / "run", f"run-window-{used}", RunOptions(pendulum_length_m=0.6))
    notes = [t["text"] for t in result["trail"] if t["step"] == "plausibility"]
    if used:
        assert any(n.startswith("Fits 0.20 to 1.90 s") for n in notes)
        assert result["full_clip_verdict"]["status"] in ("accepted", "ambiguous", "no_fit")
        assert len(result["real_track_px"]) < 60
    else:
        assert any("Refused the agent's fit window" in n and "70%" in n for n in notes)
        assert result["full_clip_verdict"] is None and len(result["real_track_px"]) == 60


def test_window_guard():
    from physicslens.physics.verify import window_indices
    t = np.arange(300) / 30
    assert window_indices(t, 1.0, 9.0) == (30, 271)
    assert "below the 70%" in window_indices(t, 3.0, 7.0)
    assert "ends before" in window_indices(t, 5.0, 4.0)
