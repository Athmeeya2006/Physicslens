"""Uploads are read before they are queued: an unreadable file or a too-short clip is refused with the reason."""
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "samples" / "smoke_ball.mp4"


def post(data: bytes, name: str = "a.mp4"):
    from physicslens import app as app_module
    return TestClient(app_module.app).post("/api/runs", files={"video": (name, data, "video/mp4")})


def test_a_file_that_is_not_a_video_is_refused_before_it_is_queued():
    r = post(b"this is not a video")
    assert r.status_code == 400 and "could not be read as a video" in r.json()["detail"]


def test_a_clip_under_two_seconds_is_refused_with_its_length(tmp_path):
    short = tmp_path / "short.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(CLIP), "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(short)], check=True)
    r = post(short.read_bytes())
    assert r.status_code == 400 and "too short" in r.json()["detail"] and "frames" in r.json()["detail"]


def test_a_real_clip_is_accepted():
    assert post(CLIP.read_bytes()).status_code == 200


def test_a_gravity_reading_must_be_three_finite_numbers():
    from physicslens import app as app_module
    c = TestClient(app_module.app)
    for bad in ("1,2", "a,b,c", "0,0,0", "nan,1,0"):
        r = c.post("/api/runs", files={"video": ("a.mp4", CLIP.read_bytes(), "video/mp4")}, data={"gravity_up": bad})
        assert r.status_code == 400 and "gravity reading" in r.json()["detail"], bad
    assert app_module.parse_gravity_up("0.1,9.8,-0.3") == (0.1, 9.8, -0.3)
