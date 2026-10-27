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

    def __init__(self, fail_on_call: int | None = None):
        self.lines = fixture_lines()
        self.prompts: list[str] = []
        self.start = 0
        self.fail_on_call = fail_on_call

    def __call__(self, path: Path, prompt: str, n: int) -> str:
        self.prompts.append(prompt)
        if self.fail_on_call is not None and len(self.prompts) == self.fail_on_call:
            raise BudgetExceeded("per-run budget would be exceeded")
        out = []
        for i in range(n):
            src = self.lines[(self.start + i) % CLIP_FRAMES]
            out.append(re.sub(r"^<\d+f>0<", f"<{i}f>1<", src))
        self.start += n
        return "\n".join(out)


def make_session(harness, tmp_path, sam):
    harness.sdk("meta_api").script.extend([responses_result(text=json.dumps(SCENE)), responses_result(text=json.dumps(HYPOTHESES))])
    sent: list[dict] = []
    session = LiveSession(harness.client, tmp_path / "live", "live-test",
                          LiveConfig(fps=FPS, pendulum_length_m=0.6), sent.append, sam_fn=sam, synchronous=True)
    return session, sent


def test_live_stream_tracks_fits_and_predicts(harness, tmp_path):
    sam = FakeSam()
    session, sent = make_session(harness, tmp_path, sam)
    frames = clip_jpegs()
    for i in range(150):
        assert session.add_frame(frames[i % CLIP_FRAMES]) == i

    kinds = [m["type"] for m in sent]
    assert kinds[0] == "scene" and kinds.count("track") == 3 and "hypotheses" in kinds
    tracked = [p[0] for m in sent if m["type"] == "track" for p in m["points"]]
    assert tracked == list(range(150))

    # Identity carries over: chunk 2 is prompted with a box where chunk 1 left the ball.
    last = [p for m in sent if m["type"] == "track" for p in m["points"]][49]
    x1, y1, x2, y2 = map(int, re.search(r"<0f>1<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+)", sam.prompts[1]).groups())
    assert x1 < last[1] < x2 and y1 < last[2] < y2
    assert "2<|box" not in sam.prompts[1]  # no reference object in this scene

    fit = [m for m in sent if m["type"] == "fit"][-1]
    assert fit["status"] in ("accepted", "ambiguous") and fit["equation"]["model_id"].startswith("pendulum")
    ghost = {f: (x, y) for f, x, y in fit["ghost"]}
    assert max(ghost) >= 149 + int(PREDICT_S * FPS) - 1  # extends past the newest tracked frame
    real = {p[0]: (p[1], p[2]) for m in sent if m["type"] == "track" for p in m["points"]}
    err = [np.hypot(ghost[f][0] - real[f][0], ghost[f][1] - real[f][1]) for f in real if f in ghost]
    assert np.median(err) < 3


def test_budget_pause_keeps_session_alive(harness, tmp_path):
    sam = FakeSam(fail_on_call=2)
    session, sent = make_session(harness, tmp_path, sam)
    frames = clip_jpegs()
    for i in range(200):
        session.add_frame(frames[i % CLIP_FRAMES])
    assert session.paused_reason is not None
    assert len(sam.prompts) == 2  # no SAM calls after the pause
    notices = [m for m in sent if m["type"] == "notice"]
    assert notices and "paused" in notices[0]["message"]


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
        for i in range(50):
            ws.send_bytes(frames[i % CLIP_FRAMES])
        # Mock models return non-JSON, so the first chunk is skipped with a notice and the socket stays up.
        msg = ws.receive_json()
        assert msg["type"] == "notice" and "Skipped" in msg["message"]


def test_public_visitors_get_no_live_mode(monkeypatch):
    from fastapi.testclient import TestClient
    from physicslens import app as app_module

    monkeypatch.setattr(app_module, "audience_for", lambda code: "public")
    client = TestClient(app_module.app)
    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"fps": 25})
        msg = ws.receive_json()
        assert msg["type"] == "notice" and "access code" in msg["message"]
