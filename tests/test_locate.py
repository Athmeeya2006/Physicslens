"""The single-frame locate step: tighter boxes than the four-frame scene call, and a later start for objects not yet in view."""

import json

import pytest
import yaml
from physicslens.agent.schemas import Scene
from physicslens.costguard.mock import responses_result, sam_result
from physicslens.perception.media import VideoInfo, keyframe_indices
from physicslens.pipeline import AnalysisError, locate_precisely, track_objects

from .conftest import Harness
from .test_pipeline import CLIP, SCENE

INFO = VideoInfo(width=640, height=360, fps=30.0, duration_s=2.0, frames=60, codec="h264")
TIGHT = {"x1": 600, "y1": 600, "x2": 700, "y2": 760}


@pytest.fixture
def locating(settings):
    path = settings.config_dir / "limits.yaml"
    limits = yaml.safe_load(path.read_text())
    limits["scene"]["locate_single_image"] = True
    path.write_text(yaml.safe_dump(limits))
    return Harness(settings)


def located(box=None, second=None, reference=None):
    return responses_result(text=json.dumps({"moving_object_box": box, "second_object_box": second, "reference_box": reference}))


def run(h, frames=4):
    trail: list[dict] = []

    def record(step, call, text, **extra):
        trail.append({"step": step, "model": call.model, "text": text, **extra})

    scene, frame = locate_precisely(h.client, Scene(**SCENE), [f"jpeg{i}".encode() for i in range(frames)], INFO, "r", "dev", record,
                                    h.client.limits, lambda _m: None)
    return scene, frame, trail


def test_single_frame_box_replaces_the_scene_box(locating):
    locating.sdk().script.append(located(TIGHT))
    scene, frame, trail = run(locating)
    assert frame == 0 and scene.moving_object_box.model_dump() == TIGHT
    assert [(t["step"], t["status"]) for t in trail] == [("locate", "accepted")]
    sent = locating.sdk().calls[0]
    assert len([p for p in sent["input"][0]["content"] if p["type"] == "input_image"]) == 1  # one frame only


def test_object_not_yet_in_view_starts_sam_on_a_later_keyframe(locating):
    locating.sdk().script.extend([located(None), located(TIGHT)])
    scene, frame, trail = run(locating)
    assert frame == keyframe_indices(INFO.frames, 4)[1] > 0
    assert [t["status"] for t in trail] == ["rejected", "accepted"] and "not in view" in trail[0]["text"]
    # SAM gets the box on that frame, not on frame 0.
    locating.sdk().script.append(sam_result(frames=60, text=""))
    with pytest.raises(AnalysisError):  # empty SAM output: nothing tracked; only the prompt matters here
        track_objects(locating.client, CLIP, INFO, scene, "r", "dev", frame)
    prompt = locating.sdk().calls[-1]["input"][0]["content"][0]["text"]
    assert prompt.startswith(f"<{frame}f>")


def test_never_in_view_keeps_the_scene_box_on_frame_0(locating):
    locating.sdk().script.extend([located(None)] * 3)
    scene, frame, trail = run(locating)
    assert frame == 0 and scene.moving_object_box.model_dump() == SCENE["moving_object_box"]
    assert len(trail) == 3  # locate_max_frames


def test_a_failed_locate_call_keeps_the_scene_box(locating):
    locating.sdk().script.extend([responses_result(text="not json")] * 2)
    scene, frame, trail = run(locating)
    assert frame == 0 and scene.moving_object_box.model_dump() == SCENE["moving_object_box"] and trail == []


def test_locate_can_be_switched_off(harness):
    _scene, frame, trail = run(harness)
    assert frame == 0 and trail == [] and harness.sdk().calls == []


def test_small_smooth_motion_counts_but_jitter_does_not():
    """Numbers from real tracks: a truck bouncing 6 px on its springs, and a stopwatch rim SAM locked onto by mistake."""
    import numpy as np
    from physicslens.pipeline import smooth_motion
    t = np.arange(59) / 30
    truck = np.column_stack([300 + 0 * t, 200 + 3.2 * np.sin(2 * np.pi * 1.2 * t) * np.exp(-t)])
    rng = np.random.default_rng(1)
    rim = np.column_stack([500 + rng.normal(0, 0.5, 164), 300 + rng.normal(0, 0.5, 164)])
    rim[80:, 0] += 2.0  # a small jump, as a mask outline flickers
    assert smooth_motion(truck, float(np.ptp(truck, axis=0).max()))
    assert not smooth_motion(rim, float(np.ptp(rim, axis=0).max()))
    assert not smooth_motion(truck * 0.2 + 100, 1.0)  # too little travel to measure at all


def test_keyframes_follow_brief_motion_and_stay_even_otherwise(tmp_path):
    """A ball crossing for one second of an eight-second clip is what the keyframes show; steady motion keeps even spacing."""
    import cv2
    import numpy as np
    from physicslens.perception.media import clip_keyframe_indices

    def write(path, x_of):
        out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
        for i in range(240):
            frame = np.full((120, 160, 3), 200, np.uint8)
            cv2.circle(frame, (int(x_of(i)), 60), 8, (0, 0, 255), -1)
            out.write(frame)
        out.release()

    brief = tmp_path / "brief.mp4"
    write(brief, lambda i: 20 + min(max(i - 90, 0), 30) * 4)  # still, rolls across between frames 90 and 120, still
    idx = clip_keyframe_indices(brief, 4)
    assert idx[0] == 0 and all(80 <= i <= 130 for i in idx[1:])
    steady = tmp_path / "steady.mp4"
    write(steady, lambda i: 80 + 60 * np.sin(i / 10))
    assert clip_keyframe_indices(steady, 4) == keyframe_indices(240, 4)
