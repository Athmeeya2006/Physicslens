"""Setup guides: Glimmer plans, code writes the prompt, Muse Image draws, math checks, code writes the text. All mocked."""

import json
import random
import re

import cv2
import numpy as np
import pytest
import yaml
from physicslens.agent import setup_guide as sg
from physicslens.agent.schemas import SetupGuidePlan
from physicslens.costguard import InvalidRequest
from physicslens.costguard.mock import chat_result, image_result
from physicslens.perception.guide_check import check_edit
from pydantic import ValidationError

from .conftest import TEST_BUDGET, Harness

W, H = 640, 360
INSTRUCTION_LABEL = "release from about 20 degrees and film 7 s"
PLAN = {"instruction": "Release it from about 20 degrees and film for 7 s.",
        "marks": [{"kind": "release_point", "label": "let go here", "box": {"x1": 400, "y1": 300, "x2": 600, "y2": 700}}]}
RESULT = {
    "scene": {"moving_object": "metal ball", "observations": "A ball swings on a string."},
    "experiment_design": {"recommended": {"label": INSTRUCTION_LABEL}},
    "trail": [{"step": "scene"}, {"step": "design", "items": []},
              {"step": "agent", "action": "request_experiment", "text": "Please release it from about 20 degrees."}],
}
DESIGN_REF = 1


def scene_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 1)
    cv2.rectangle(img, (60, 40), (220, 300), (40, 160, 90), -1)
    cv2.line(img, (320, 0), (330, 200), (230, 230, 230), 3)
    return img


def jpeg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


KEYFRAME = jpeg(scene_image())


def marked(src: bytes) -> bytes:
    """A careful edit: a ring drawn inside the planned box, the rest left alone."""
    img = cv2.imdecode(np.frombuffer(src, np.uint8), cv2.IMREAD_COLOR)
    cv2.circle(img, (W // 2, H // 2), 45, (0, 0, 255), 6)
    return jpeg(img)


@pytest.fixture
def h(settings, monkeypatch, tmp_path) -> Harness:
    monkeypatch.setenv("REASONING_PROVIDER", "glimmer")
    monkeypatch.setenv("GLIMMER_BASE_URL", "http://127.0.0.1:8080/v1")
    (settings.config_dir / "budget.yaml").write_text(yaml.safe_dump(TEST_BUDGET | {"setup_guide_daily_cap_usd": 0.05}))
    monkeypatch.setattr(sg, "clearest_keyframe", lambda clip, cfg: KEYFRAME)
    harness = Harness(settings)
    harness.sdk("glimmer").script.extend([chat_result(text=json.dumps(PLAN))] * 4)
    return harness


def run(h: Harness, tmp_path, audience: str = "dev") -> dict:
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    return sg.make_setup_guide(h.client, run_dir, json.loads(json.dumps(RESULT)), DESIGN_REF, "guide-test", audience)


def image_posts(h: Harness) -> list[dict]:
    return [c for c in h.sdk("meta_api").calls if c["kind"] == "post"]


def test_scene_preserving_edit_is_accepted_and_stored(h, tmp_path):
    h.sdk("meta_api").script.append(image_result(marked(KEYFRAME)))
    guide = run(h, tmp_path)
    assert guide["status"] == "picture", guide.get("reason")
    assert guide["instruction"] == "Release from about 20 degrees and film 7 s."  # the math's words, not the model's
    assert (tmp_path / "run" / sg.GUIDE_JPG).is_file()
    stored = json.loads((tmp_path / "run" / sg.GUIDE_JSON).read_text())
    assert stored["plan"] == PLAN and stored["prompt"] == guide["prompt"] and stored["model"] == "muse-image-1.0"
    assert stored["checks"][0]["accepted"] and stored["checks"][0]["ssim_outside"] >= 0.8
    assert stored["images_billed"] == 1 and "cost_usd" not in stored  # money stays in the private ledger
    assert (tmp_path / "run" / sg.GUIDE_FRAME).read_bytes() == KEYFRAME
    assert guide["cost_usd"] == pytest.approx(0.01)
    item = guide["trail_item"]
    assert item["step"] == "setup_guide" and item["provider"] == "meta_api" and item["model"] == "muse-image-1.0"
    assert item["planned_by"] == {"provider": "glimmer", "model": "muse-glimmer-30b"}
    # The request is the documented JSON body for POST /images/edits.
    post = image_posts(h)[0]
    assert post["path"] == "/images/edits" and post["n"] == 1 and post["model"] == "muse-image-1.0"
    assert post["images"][0]["image_url"].startswith("data:image/jpeg;base64,")
    assert set(post) - {"kind", "path"} == {"model", "prompt", "images", "n", "reasoning_strength", "tool_enablement"}


def test_whole_scene_changed_is_rejected_retried_once_then_text_only(h, tmp_path):
    other = jpeg(scene_image(seed=7))
    h.sdk("meta_api").script.extend([image_result(other), image_result(other)])
    guide = run(h, tmp_path)
    assert guide["status"] == "text_only" and "changed the scene" in guide["reason"]
    assert len(image_posts(h)) == 2
    assert [c["accepted"] for c in guide["checks"]] == [False, False]
    assert not (tmp_path / "run" / sg.GUIDE_JPG).exists()
    assert guide["instruction"]  # the written guide always remains


def test_retry_can_succeed(h, tmp_path):
    h.sdk("meta_api").script.extend([image_result(jpeg(scene_image(seed=7))), image_result(marked(KEYFRAME))])
    guide = run(h, tmp_path)
    assert guide["status"] == "picture" and len(image_posts(h)) == 2


def test_identical_image_is_nothing_added(h, tmp_path):
    h.sdk("meta_api").script.extend([image_result(KEYFRAME), image_result(KEYFRAME)])
    guide = run(h, tmp_path)
    assert guide["status"] == "text_only" and guide["reason"] == "Nothing was added where the marks should be."


def test_budget_cap_reached_means_no_call_and_text_only(h, settings, tmp_path):
    (settings.config_dir / "budget.yaml").write_text(yaml.safe_dump(TEST_BUDGET | {"setup_guide_daily_cap_usd": 0.0}))
    h2 = Harness(settings)
    h2.sdk("glimmer").script.append(chat_result(text=json.dumps(PLAN)))
    guide = run(h2, tmp_path)
    assert guide["status"] == "text_only" and guide["reason"] == sg.PAUSED
    assert image_posts(h2) == []


def test_missing_cap_means_no_call(settings, monkeypatch, tmp_path):
    monkeypatch.setenv("REASONING_PROVIDER", "glimmer")
    monkeypatch.setenv("GLIMMER_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setattr(sg, "clearest_keyframe", lambda clip, cfg: KEYFRAME)
    h2 = Harness(settings)  # the shared test budget has no setup_guide_daily_cap_usd
    h2.sdk("glimmer").script.append(chat_result(text=json.dumps(PLAN)))
    guide = run(h2, tmp_path)
    assert guide["reason"] == sg.PAUSED and image_posts(h2) == []


def test_public_is_refused_by_costguard(h):
    with pytest.raises(InvalidRequest):
        h.client.meta_image_edit(image_jpeg=KEYFRAME, prompt="draw a ring", run_id="r", audience="public")
    assert image_posts(h) == []


def test_second_identical_request_makes_no_paid_call(h, tmp_path):
    h.sdk("meta_api").script.append(image_result(marked(KEYFRAME)))
    first = run(h, tmp_path)
    again = run(h, tmp_path)
    assert again["status"] == "picture" and again["cost_usd"] == 0.0 and len(image_posts(h)) == 1
    # Even without the stored file, the response cache (keyframe bytes + plan-derived prompt) answers for free.
    (tmp_path / "run" / sg.GUIDE_JSON).unlink()
    third = run(h, tmp_path)
    assert third["status"] == "picture" and third["cost_usd"] == 0.0 and len(image_posts(h)) == 1
    assert third["prompt"] == first["prompt"]
    assert "cache_hit" in [r["status"] for r in h.client.ledger.rows()]


def random_plan(rng: random.Random) -> SetupGuidePlan:
    words = ["here", "ball", "phone", "let", "go", "keep", "clear", "push", "the", "edge", "string", "table", "it's"]
    marks = []
    for _ in range(rng.randint(1, 4)):
        x1, y1 = rng.randint(0, 900), rng.randint(0, 900)
        marks.append({"kind": rng.choice(list(sg.DRAWING)), "label": " ".join(rng.sample(words, rng.randint(1, 4))),
                      "box": {"x1": x1, "y1": y1, "x2": rng.randint(x1 + 1, 1000), "y2": rng.randint(y1 + 1, 1000)}})
    return SetupGuidePlan.model_validate({"instruction": "Drop it from 2 m, then 4 m.", "marks": marks})


def test_prompt_never_has_digits_and_only_template_text_plus_labels():
    rng = random.Random(1234)
    vocab = sg.template_vocabulary()
    for _ in range(300):
        plan = random_plan(rng)
        prompt = sg.build_edit_prompt(plan)
        assert not re.search(r"\d", prompt)
        labels = {w for m in plan.marks for w in re.findall(r"[a-z']+", sg.clean_label(m.label))}
        assert set(re.findall(r"[a-z']+", prompt.lower())) <= vocab | labels
        assert "add no text, no numbers and no letters" in prompt.lower() and "keep the scene exactly" in prompt.lower()


def test_labels_with_digits_or_too_many_words_are_rejected():
    for label in ("release at 20", "one two three four five"):
        with pytest.raises(ValidationError):
            SetupGuidePlan.model_validate({"instruction": "x", "marks": [{"kind": "push_here", "label": label,
                                                                          "box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}]})


def test_overlay_numbers_appear_in_overlay_never_in_prompt(h, tmp_path):
    h.sdk("meta_api").script.append(image_result(marked(KEYFRAME)))
    guide = run(h, tmp_path)
    overlay = " ".join(guide["overlay_text"])
    assert "20" in overlay and "7 s" in overlay and "let go here" in overlay
    assert not re.search(r"\d", guide["prompt"])
    img = cv2.imread(str(tmp_path / "run" / sg.GUIDE_JPG))
    assert img.shape[1] == W and img.shape[0] > H  # caption band below the frame


def test_check_resizes_a_differently_sized_edit():
    original = scene_image()
    edited = cv2.imdecode(np.frombuffer(marked(jpeg(original)), np.uint8), cv2.IMREAD_COLOR)
    big = cv2.resize(edited, (W * 2, H * 2), interpolation=cv2.INTER_CUBIC)
    box = (256, 108, 384, 252)
    check, out = check_edit(original, big, [box], ssim_min=0.8, change_min=6.0, pad_fraction=0.05)
    assert check.resized and check.accepted and out.shape == original.shape


def test_only_experiment_requests_get_guides():
    with pytest.raises(sg.SetupGuideError):
        sg.source_instruction(RESULT, 0)
    assert sg.source_instruction(RESULT, 2).startswith("Please release")


def test_endpoint_refuses_public_visitors(monkeypatch):
    from fastapi.testclient import TestClient
    from physicslens import app as app_module
    monkeypatch.setattr(app_module, "audience_for", lambda code: "public")
    r = TestClient(app_module.app).post("/api/runs/abc123/setup-guide", json={"step_ref": 1})
    assert r.status_code == 403


def test_clearest_keyframe_picks_the_sharpest(monkeypatch, tmp_path):
    sharp = jpeg(scene_image())
    blurry = jpeg(cv2.GaussianBlur(scene_image(), (0, 0), 6))
    monkeypatch.setattr(sg.media, "keyframes", lambda path, n, side: [blurry, sharp, blurry])
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    assert sg.clearest_keyframe(clip, {}) == sharp
