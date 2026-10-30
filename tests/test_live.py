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
        assert msg["type"] == "notice" and "paused" in msg["message"]


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


def test_capture_timestamps_set_the_time_base(harness, tmp_path):
    """A phone that can't hold the nominal rate: frames stamped with real capture times still give the right g."""
    import struct

    from physicslens.live import FRAME_HEADER

    def final_g(fps_nominal: float, stamped: bool, name: str) -> float:
        harness.sdk("meta_api").script.extend([responses_result(text=json.dumps(SCENE)),
                                               responses_result(text=json.dumps(HYPOTHESES))])
        sent: list[dict] = []
        session = LiveSession(harness.client, tmp_path / name, f"live-{name}",
                              LiveConfig(fps=fps_nominal, pendulum_length_m=0.6), sent.append, sam_fn=FakeSam(),
                              synchronous=True)
        frames = clip_jpegs()
        for i in range(150):
            jpeg = frames[i % CLIP_FRAMES]
            session.add_frame(FRAME_HEADER + struct.pack("<d", 100.0 + i / FPS) + jpeg if stamped else jpeg)
        fit = [m for m in sent if m["type"] == "fit" and m.get("parameters")][-1]
        return next(p["value"] for p in fit["parameters"] if p["name"] == "g"), fit

    g_true, _ = final_g(FPS, False, "nominal")
    g_stamped, fit = final_g(25, True, "stamped")  # the server's nominal rate is wrong; the stamps are right
    assert g_stamped == pytest.approx(g_true, rel=0.02)
    assert fit["measured_fps"] == pytest.approx(FPS, rel=0.01)


def test_tapped_family_fits_at_once_and_the_agent_still_proposes(harness, tmp_path):
    sam = FakeSam()
    session, sent = make_session(harness, tmp_path, sam)
    session.choose_family_now("pendulum")
    assert session.family == "pendulum" and "pendulum" in session.candidates
    frames = clip_jpegs()
    for i in range(150):
        session.add_frame(frames[i % CLIP_FRAMES])
    fits = [m for m in sent if m["type"] == "fit" and m.get("equation")]
    assert fits and fits[-1]["equation"]["model_id"].startswith("pendulum")
    hyps = [m for m in sent if m["type"] == "hypotheses"]
    assert len(hyps) == 1  # the agent was still consulted, once, after the scene was read
    session.choose_family_now("projectile")  # a second tap changes nothing mid-session
    assert session.family == "pendulum"


def test_live_prediction_is_stored_before_it_happens_and_then_judged(harness, tmp_path):
    from physicslens.physics.coords import Coordinates, to_coordinates
    from physicslens.physics.fit import fit_model
    from physicslens.physics.models import MODELS
    from physicslens.sigproc.trajectory import Trajectory

    from .test_physics import pendulum_xy

    session, sent = make_session(harness, tmp_path, lambda *a: "")
    session.family, session.t0 = "pendulum", 0.0
    session.history = [{}] * 3  # the law has been stable for a few refits
    xy = pendulum_xy(0.5, 0.03, 40)
    rng = np.random.default_rng(0)
    full = to_coordinates(Trajectory(np.arange(len(xy)) / FPS, xy + rng.normal(0, 0.002, xy.shape),
                                     np.arange(len(xy)), FPS), "pendulum")

    def window(a: float, b: float) -> Coordinates:
        i, j = int(a * FPS), int(b * FPS)
        return Coordinates("pendulum", full.t[i:j] - full.t[i], full.q[i:j], full.names, dict(full.ctx))

    spec = MODELS["pendulum_linear_damping"]
    early = window(0, 10)
    session._forecast_step(spec, fit_model(spec, early), early, 0.0, accepted=True)
    made = [m for m in sent if m["type"] == "prediction"]
    assert made and made[0]["status"] == "pending" and made[0]["target_s"] >= 10 + made[0]["horizon_s"] - 0.1
    stored = json.loads((tmp_path / "live" / "predictions.json").read_text())
    assert stored[0]["status"] == "pending" and stored[0]["made_wall"] > 0
    late = window(26, 36)  # a later refit window that contains the predicted crossing
    session._forecast_step(spec, fit_model(spec, late), late, 26.0, accepted=True)
    judged = [m for m in sent if m["type"] == "prediction" and m["status"] != "pending"]
    assert judged and judged[0]["status"] == "hit", judged


def test_control_cues_a_push_train_then_waits_for_it(harness, tmp_path):
    from physicslens.physics.coords import Coordinates
    from physicslens.physics.fit import fit_model
    from physicslens.physics.models import MODELS

    from .test_physics import pendulum_xy

    session, sent = make_session(harness, tmp_path, lambda *a: "")
    session.family, session.t0 = "pendulum", 0.0
    session.set_control("pump", 30)
    xy = pendulum_xy(0.1, 0.05, 8)
    th = np.arctan2(xy[:, 0] - 0.1, 1.0 - xy[:, 1])
    c = Coordinates("pendulum", np.arange(len(th)) / FPS, th[:, None], ("theta",),
                    {"length_m": 0.6, "pivot_x": 0.1, "pivot_y": 1.0})
    spec = MODELS["pendulum_linear_damping"]
    session._control_step(spec, fit_model(spec, c), c, 0.0)
    cues = [m for m in sent if m["type"] == "cue"]
    assert cues and cues[0]["action"] == "push" and len(cues[0]["pushes_s"]) == 3
    assert all(p > c.t[-1] for p in cues[0]["pushes_s"]) and cues[0]["direction"] in ("left", "right")
    session._control_step(spec, fit_model(spec, c), c, 0.0)  # the train hasn't happened yet: nothing new
    assert len([m for m in sent if m["type"] == "cue"]) == 1
    session.set_control("off", 0)
    assert session.control is None and sent[-1] == {"type": "cue", "action": "off"}


def test_object_size_sets_the_scale_for_anything_that_is_not_a_pendulum(harness, tmp_path):
    session = LiveSession(harness.client, tmp_path / "live", "live-size", LiveConfig(fps=FPS, object_size_m=0.05),
                          lambda m: None, sam_fn=FakeSam(), synchronous=True)
    session.object_size = 40.0  # the ball's size in SAM's masks, in pixels
    assert session._scale(np.zeros((10, 2))) == 40.0 / 0.05


def test_no_hypotheses_and_no_scene_family_asks_instead_of_assuming_a_pendulum(harness, tmp_path):
    from physicslens.agent.schemas import Scene
    sent: list[dict] = []
    session = LiveSession(harness.client, tmp_path / "live", "live-ask", LiveConfig(fps=FPS), sent.append,
                          sam_fn=FakeSam(), synchronous=True)
    session.scene = Scene(**{**SCENE, "motion_family": "other"})
    harness.sdk("meta_api").script.extend([responses_result(text="not json")] * 2)  # hypotheses fail twice
    t = np.arange(60) / FPS
    from physicslens.sigproc.trajectory import Trajectory
    session._choose_family(Trajectory(t=t, xy=np.column_stack([t, 0 * t]), frames=np.arange(60), fps=FPS))
    assert session.family is None and session.awaiting_family
    assert any(m["type"] == "choose_family" for m in sent)
    session.choose_family_now("line")  # the user's tap starts fitting, and the model is not asked again
    assert session.family == "line" and not session.awaiting_family and session._agent_consulted
