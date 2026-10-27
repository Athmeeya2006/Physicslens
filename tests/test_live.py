"""Live mode driven synchronously with a looped synthetic clip and the recorded SAM response (no network)."""

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest
from physicslens.costguard import BudgetExceeded
from physicslens.costguard.mock import responses_result
from physicslens.live import PREDICT_S, LiveConfig, LiveSession

from .test_pipeline import HYPOTHESES, SCENE

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sam_smoke_response.json"
CLIP_FRAMES = 60  # the smoke clip is exactly two periods, so it loops seamlessly
FPS = 30


def fixture_lines() -> dict[int, str]:
    raw = json.loads(FIXTURE.read_text())
    text = "".join(c.get("text", "") for item in raw["output"] for c in item.get("content") or [])
    return {int(re.match(r"<(\d+)f>", ln).group(1)): ln for ln in text.splitlines()}


def clip_jpegs() -> list[bytes]:
    cap = cv2.VideoCapture(str(ROOT / "samples" / "smoke_ball.mp4"))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(cv2.imencode(".jpg", frame)[1].tobytes())
    return out


class FakeSam:
    """Serves the recorded SAM lines for whichever global frames a chunk covers, as object 1."""

    def __init__(self, fail_on_call: int | None = None, shift_x: int = 0):
        self.lines = fixture_lines()
        self.prompts: list[str] = []
        self.indices: list[list[int]] = []
        self.fail_on_call = fail_on_call
        self.shift_x = shift_x

    def __call__(self, path: Path, prompt: str, indices: list[int]) -> str:
        self.prompts.append(prompt)
        self.indices.append(indices)
        if self.fail_on_call is not None and len(self.prompts) == self.fail_on_call:
            raise BudgetExceeded("per-run budget would be exceeded")
        out = []
        for k, g in enumerate(indices):
            line = re.sub(r"^<\d+f>0<", f"<{k}f>1<", self.lines[g % CLIP_FRAMES])
            if self.shift_x:
                line = re.sub(r"x([12])=(\d+)", lambda m: f"x{m.group(1)}={int(m.group(2)) + self.shift_x}", line)
            out.append(line)
        return "\n".join(out)


def make_session(harness, tmp_path, sam):
    harness.sdk("meta_api").script.extend([responses_result(text=json.dumps(SCENE)), responses_result(text=json.dumps(HYPOTHESES))])
    sent: list[dict] = []
    session = LiveSession(harness.client, tmp_path / "live", "live-test",
                          LiveConfig(fps=FPS, pendulum_length_m=0.6), sent.append, sam_fn=sam, synchronous=True)
    return session, sent


def points(sent, source):
    return [p for m in sent if m["type"] == "track" and m.get("source") == source for p in m["points"]]


def test_live_stream_tracks_fits_and_predicts(harness, tmp_path):
    sam = FakeSam()
    session, sent = make_session(harness, tmp_path, sam)
    frames = clip_jpegs()
    for i in range(150):
        assert session.add_frame(frames[i % CLIP_FRAMES]) == i

    assert sent[0]["type"] == "scene"
    # The local tracker follows every frame right away...
    local = points(sent, "local")
    assert [p[0] for p in local] == list(range(1, 1 + len(local))) and local[-1][0] >= 145
    # ...and SAM refines every second frame of the first 100 (one chunk of 50 frames).
    assert sam.indices == [list(range(0, 100, 2))]
    assert [p[0] for p in points(sent, "sam")] == list(range(0, 100, 2))
    # Local and SAM agree on the synthetic ball.
    sam_pos = {p[0]: p[1:] for p in points(sent, "sam")}
    err = [np.hypot(p[1] - sam_pos[p[0]][0], p[2] - sam_pos[p[0]][1]) for p in local if p[0] in sam_pos]
    assert np.median(err) < 1.5

    fits = [m for m in sent if m["type"] == "fit"]
    assert fits and fits[-1]["equation"]["model_id"].startswith("pendulum")
    assert 0 < fits[-1]["sam_share"] < 1
    ghost = {f: (x, y) for f, x, y in fits[-1]["ghost"]}
    newest_fitted = min(ghost) + fits[-1]["samples"] - 1
    assert newest_fitted >= 140 and max(ghost) - newest_fitted == int(PREDICT_S * FPS)
    real = {p[0]: p[1:] for p in local}
    err = [np.hypot(ghost[f][0] - real[f][0], ghost[f][1] - real[f][1]) for f in real if f in ghost]
    assert np.median(err) < 3


def test_budget_pause_keeps_local_tracking(harness, tmp_path, monkeypatch):
    import physicslens.live as live_module
    monkeypatch.setattr(live_module, "REFIT_EVERY", 10**6)  # this test is about tracking and SAM, not fitting
    sam = FakeSam(fail_on_call=2)
    session, sent = make_session(harness, tmp_path, sam)
    frames = clip_jpegs()
    for i in range(320):
        session.add_frame(frames[i % CLIP_FRAMES])
    assert session.paused_reason is not None
    assert len(sam.prompts) == 2  # chunk at frame 200 is never sent
    assert any("paused" in m["message"] for m in sent if m["type"] == "notice")
    assert points(sent, "local")[-1][0] >= 315


def test_drift_reanchors_local_tracker(harness, tmp_path):
    session, sent = make_session(harness, tmp_path, FakeSam(shift_x=40))
    frames = clip_jpegs()
    for i in range(110):
        session.add_frame(frames[i % CLIP_FRAMES])
    assert any("Re-anchored" in m["message"] for m in sent if m["type"] == "notice")
    # Re-tracking replayed from SAM's anchor frame (98) up to the newest frame, with nothing left pending.
    assert session.reanchor is None and session.local_next == 110
    assert all(f in session.local for f in range(99, 110))


def test_rejects_non_jpeg_first_frame(harness, tmp_path):
    session, _ = make_session(harness, tmp_path, FakeSam())
    with pytest.raises(ValueError):
        session.add_frame(b"not an image")


def test_websocket_plumbing_in_mock_mode(monkeypatch):
    from fastapi.testclient import TestClient
    from physicslens import app as app_module

    client = TestClient(app_module.app)
    frames = clip_jpegs()
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"fps": 30, "pendulum_length_m": 0.6})
        assert ws.receive_json()["type"] == "ready"
        for i in range(30):
            ws.send_bytes(frames[i % CLIP_FRAMES])
        # Mock models return non-JSON, so scene reading fails with a notice and the socket stays up.
        msg = ws.receive_json()
        assert msg["type"] == "notice" and "failed" in msg["message"]


def test_public_visitors_get_no_live_mode(monkeypatch):
    from fastapi.testclient import TestClient
    from physicslens import app as app_module

    monkeypatch.setattr(app_module, "audience_for", lambda code: "public")
    client = TestClient(app_module.app)
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"fps": 25})
        msg = ws.receive_json()
        assert msg["type"] == "notice" and "access code" in msg["message"]


def test_live_session_uses_its_own_cap(harness, tmp_path, monkeypatch):
    seen = []
    real = harness.client.sam_video
    monkeypatch.setattr(harness.client, "sam_video", lambda **kw: (seen.append(kw["run_cap_usd"]), real(**kw))[1])
    harness.sdk("meta_api").script.extend([responses_result(text=json.dumps(SCENE)), responses_result(text=json.dumps(HYPOTHESES))])
    session = LiveSession(harness.client, tmp_path / "live", "live-cap", LiveConfig(fps=FPS, pendulum_length_m=0.6, run_cap_usd=2.0),
                          lambda m: None, synchronous=True)
    frames = clip_jpegs()
    for i in range(100):
        session.add_frame(frames[i % CLIP_FRAMES])
    assert seen == [2.0]
    # The guard honours it: a session may exceed the per-run cap (0.20 in tests) up to its own cap.
    g = harness.client.guard
    g.check_and_reserve("meta_api", "sam-3.1", 0.15, "live-cap2", run_cap_usd=2.0).commit(0.15)
    g.check_and_reserve("meta_api", "sam-3.1", 0.15, "live-cap2", run_cap_usd=2.0).commit(0.15)


def test_tap_box_fits_the_ball():
    from physicslens.perception.local_track import box_around_tap, to_gray
    gray = to_gray(clip_jpegs()[0])
    x1, y1, x2, y2 = box_around_tap(gray, 439, 259)  # ball centre in frame 0 (radius 22)
    assert 38 <= x2 - x1 <= 52 and 38 <= y2 - y1 <= 52
    assert x1 < 439 < x2 and y1 < 259 < y2


def test_tap_starts_tracking_before_the_scene_model(harness, tmp_path):
    sam = FakeSam()
    session, sent = make_session(harness, tmp_path, sam)
    frames = clip_jpegs()
    for i in range(11):
        session.add_frame(frames[i])
    x, y = fixture_ball_center(10)
    session.select(x, y)  # frame 10, before the scene call (made at frame 25)
    for i in range(11, 130):
        session.add_frame(frames[i % CLIP_FRAMES])
    kinds = [m["type"] for m in sent]
    assert kinds.index("selected") < kinds.index("scene")
    local = points(sent, "local")
    assert local[0][0] == 11  # tracking started right after the tap frame
    # The scene model's box did not replace the user's choice: tracking stayed on the ball throughout.
    truth = {f: fixture_ball_center(f % CLIP_FRAMES) for f in range(11, 130)}
    err = [np.hypot(p[1] - truth[p[0]][0], p[2] - truth[p[0]][1]) for p in local]
    assert max(err) < 3
    assert sam.indices[0][0] == 10  # first SAM chunk begins at the tap frame


def fixture_ball_center(frame: int) -> tuple[float, float]:
    line = fixture_lines()[frame]
    x1, y1, x2, y2 = map(int, re.search(r"x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+)", line).groups())
    return (x1 + x2 + 1) / 2, (y1 + y2 + 1) / 2
