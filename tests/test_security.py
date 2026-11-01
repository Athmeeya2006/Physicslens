"""Request limits, upload sniffing and CORS: a flood or a wrong file is refused before any work starts."""
import pytest
from fastapi.testclient import TestClient
from physicslens.costguard.limiter import RequestWindows
from physicslens.perception.media import looks_like_video
from physicslens.settings import ConfigError, cors_origins


def test_windows_per_address_and_global():
    now = [0.0]
    w = RequestWindows({"g": {"per_ip_per_minute": 2, "global_per_minute": 3}}, clock=lambda: now[0])
    assert w.check("g", "a") is None and w.check("g", "a") is None
    assert w.check("g", "a") == pytest.approx(60.0)  # this address waits for its oldest request to age out
    assert w.check("g", "b") is None
    assert w.check("g", "c") is not None  # everyone together: three in the minute
    now[0] = 30.0
    assert w.check("g", "a") == pytest.approx(30.0)
    now[0] = 60.0
    assert w.check("g", "a") is None
    assert w.check("other", "a") is None  # a group with no limits is not limited


def test_upload_route_answers_429_with_retry_after(monkeypatch):
    from physicslens import app as app_module
    monkeypatch.setattr(app_module, "request_windows", RequestWindows({"uploads": {"per_ip_per_minute": 1}}))
    c = TestClient(app_module.app)
    first = c.post("/api/runs", files={"video": ("a.mp4", b"not a video at all", "video/mp4")})
    assert first.status_code == 400  # counted, then refused for not being a video
    second = c.post("/api/runs", files={"video": ("a.mp4", b"not a video at all", "video/mp4")})
    assert second.status_code == 429 and 1 <= int(second.headers["retry-after"]) <= 60


def test_live_socket_is_closed_with_try_again_later(monkeypatch):
    from physicslens import app as app_module
    windows = RequestWindows({"live": {"global_per_minute": 1}})
    assert windows.check("live", "someone else") is None  # the minute's one session is taken
    monkeypatch.setattr(app_module, "request_windows", windows)
    with TestClient(app_module.app).websocket_connect("/ws/live") as ws:
        msg = ws.receive_json()
        assert msg["level"] == "error" and "Too many sessions" in msg["message"]


@pytest.mark.parametrize("head, ok", [
    (b"\x00\x00\x00\x20ftypisom\x00\x00", True),
    (b"\x00\x00\x00\x14ftypqt  \x00\x00", True),
    (b"\x00\x00\x00\x08wide\x00\x00\x00\x00", True),
    (b"\x1a\x45\xdf\xa3\x01\x00\x00\x00\x00", True),
    (b"<html><script>", False),
    (b"\x89PNG\r\n\x1a\n\x00\x00", False),
    (b"PK\x03\x04zipfile", False),
    (b"ftyp", False),
])
def test_uploads_must_start_like_a_video(head, ok):
    assert looks_like_video(head) is ok


def test_a_renamed_non_video_is_refused_before_it_is_written(tmp_path):
    from physicslens import app as app_module
    r = TestClient(app_module.app).post("/api/runs", files={"video": ("clip.mp4", b"\x89PNG\r\n\x1a\n" * 100, "video/mp4")})
    assert r.status_code == 400 and "does not start like" in r.json()["detail"]


def test_cors_refuses_wildcards_outside_development():
    assert cors_origins("dev", "*") == ["*"]
    assert cors_origins("prod", "https://physicslens.vercel.app/, https://example.org") == [
        "https://physicslens.vercel.app", "https://example.org"]
    for env in ("demo", "prod"):
        with pytest.raises(ConfigError, match="wildcard"):
            cors_origins(env, "https://physicslens.vercel.app,*")
        with pytest.raises(ConfigError, match="https"):
            cors_origins(env, "http://evil.example")


def test_an_oversized_upload_is_cut_off_and_its_folder_removed(monkeypatch):
    from pathlib import Path

    from physicslens import app as app_module
    clip = Path(__file__).resolve().parents[1] / "samples" / "smoke_ball.mp4"
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 8 * 1024)  # the sample clip is 27 KB
    before = set(app_module.runs.root.iterdir()) if app_module.runs.root.exists() else set()
    r = TestClient(app_module.app).post("/api/runs", files={"video": ("a.mp4", clip.read_bytes(), "video/mp4")})
    assert r.status_code == 413 and "too large" in r.json()["detail"]
    after = set(app_module.runs.root.iterdir()) if app_module.runs.root.exists() else set()
    assert after == before  # nothing half-written is left behind
