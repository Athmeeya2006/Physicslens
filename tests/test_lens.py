"""Lens calibration: a known lens is recovered, and undistorting the track removes the bias distortion puts into g."""
import cv2
import numpy as np
import pytest
from physicslens.perception.lens import (
    PATTERN,
    SQUARE_MM,
    Lens,
    LensError,
    calibrate_video,
    calibrate_views,
    checkerboard_svg,
    focal_px_at,
    undistort_px,
)

W, H = 1280, 720
K = np.array([[1000.0, 0, 642.0], [0, 1000.0, 357.0], [0, 0, 1]])
DIST = np.array([-0.12, 0.03, 0.0, 0.0, 0.0])  # a mild barrel, like a phone's main camera before its own correction
BOARD = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
BOARD[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM / 1000


def views(n: int, seed: int = 0, noise_px: float = 0.2) -> list[np.ndarray]:
    """The board's corners seen from n poses spread over the frame, as findChessboardCorners would return them."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        rvec = rng.uniform([-0.5, -0.5, -0.3], [0.5, 0.5, 0.3])
        tvec = np.array([rng.uniform(-0.3, 0.15), rng.uniform(-0.2, 0.08), rng.uniform(0.35, 0.6)])
        px, _ = cv2.projectPoints(BOARD, rvec, tvec, K, DIST)
        out.append((px + rng.normal(0, noise_px, px.shape)).astype(np.float32))
    return out


def lens() -> Lens:
    return calibrate_views(views(20), (W, H))


def test_a_known_lens_is_recovered():
    cal = lens()
    assert cal.fx == pytest.approx(1000, rel=0.01) and cal.fy == pytest.approx(1000, rel=0.01)
    assert cal.dist[0] == pytest.approx(-0.12, abs=0.02)
    assert cal.rms_px < 0.5 and cal.views == 20


def test_too_few_views_is_refused_with_advice():
    with pytest.raises(LensError, match="at least 8"):
        calibrate_views(views(4), (W, H))


def test_points_undistort_and_scale_with_the_clip():
    cal = lens()
    ideal = np.array([[100.0, 80.0], [640.0, 360.0], [1200.0, 650.0]])
    norm = (ideal - K[:2, 2]) / K[0, 0]
    seen, _ = cv2.projectPoints(np.column_stack([norm, np.ones(3)]), np.zeros(3), np.zeros(3), K, DIST)
    assert np.abs(undistort_px(seen.reshape(-1, 2), cal, W, H) - ideal).max() < 1.0
    # Half the size, same shape: the same correction at half scale. Another crop: left alone.
    assert np.abs(undistort_px(seen.reshape(-1, 2) / 2, cal, W // 2, H // 2) - ideal / 2).max() < 0.5
    assert undistort_px(seen.reshape(-1, 2), cal, 1440, 1080) is None
    assert focal_px_at(cal, W // 2, H // 2) == pytest.approx(500, rel=0.01)


def test_round_trip_and_bad_values():
    cal = lens()
    assert Lens.from_dict(cal.as_dict()) == cal
    with pytest.raises(LensError):
        Lens.from_dict({**cal.as_dict(), "fx": -3})
    with pytest.raises(LensError):
        Lens.from_dict({"width": 10})


def board_video(path):
    """The printable board, rendered through a lens-free camera from several angles, as a video."""
    sq = 40
    board = np.full(((PATTERN[1] + 3) * sq, (PATTERN[0] + 3) * sq), 255, np.uint8)
    for r in range(PATTERN[1] + 1):
        for c in range(PATTERN[0] + 1):
            if (r + c) % 2 == 0:
                board[(r + 1) * sq:(r + 2) * sq, (c + 1) * sq:(c + 2) * sq] = 0
    rng = np.random.default_rng(1)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    bh, bw = board.shape
    for _ in range(24):
        rvec = rng.uniform([-0.4, -0.4, -0.2], [0.4, 0.4, 0.2])
        tvec = np.array([rng.uniform(-0.12, 0.0), rng.uniform(-0.1, -0.02), rng.uniform(0.45, 0.6)])
        corners = np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]], np.float32) * (SQUARE_MM / 1000 / sq)
        corners[:, :2] -= SQUARE_MM / 1000  # board corner squares start one square in
        px, _ = cv2.projectPoints(corners, rvec, tvec, K, None)
        h = cv2.getPerspectiveTransform(np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]]), px.reshape(4, 2).astype(np.float32))
        frame = cv2.warpPerspective(board, h, (W, H), borderValue=200)
        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    out.release()
    return path


def test_a_video_of_the_printed_board_calibrates(tmp_path):
    cal = calibrate_video(board_video(tmp_path / "board.mp4"), samples=24)
    assert cal.fx == pytest.approx(1000, rel=0.03)
    assert abs(cal.dist[0]) < 0.05  # no distortion was put in, so none should come out


def test_undistorting_removes_the_bias_in_a_dropped_ball():
    """A ball dropped near the side of the frame, scale measured at the centre: barrel distortion shrinks the fall.

    With this lens: g 3.1% low as filmed, 0.15% after undistorting (0.3% before and 0.1% after for a drop through the
    centre, where distortion is smallest).
    """
    g, z = 9.81, 1.2
    t = np.arange(0, 0.38, 1 / 120)  # top of the frame to near the bottom
    # Camera axes (OpenCV): y points down the image, so a fall is +y. x = 0.6 m puts it near the right edge.
    world = np.column_stack([np.full_like(t, 0.6), -0.33 + 0.5 * g * t**2, np.full_like(t, z)])
    seen, _ = cv2.projectPoints(world, np.zeros(3), np.zeros(3), K, DIST)
    seen = seen.reshape(-1, 2) + np.random.default_rng(2).normal(0, 0.3, (len(t), 2))
    px_per_m = K[0, 0] / z  # a reference object at the centre of the frame

    def g_from(px: np.ndarray) -> float:
        y = -(px[:, 1] - K[1, 2]) / px_per_m
        return -2 * np.polyfit(t, y, 2)[0]

    before = abs(g_from(seen) - g) / g
    after = abs(g_from(undistort_px(seen, lens(), W, H)) - g) / g
    assert before > 0.025  # several percent wrong as filmed: far more than the bootstrap spread of a good clip
    assert after < 0.003 and after < before / 5


def test_the_printable_board_has_the_right_squares():
    svg = checkerboard_svg()
    assert svg.count("<rect") == 1 + ((PATTERN[0] + 1) * (PATTERN[1] + 1) + 1) // 2
    assert 'width="297mm"' in svg


def test_calibration_endpoints_and_a_bad_lens_on_upload(tmp_path):
    from fastapi.testclient import TestClient
    from physicslens import app as app_module
    c = TestClient(app_module.app)
    assert c.get("/api/lens/checkerboard.svg").headers["content-type"].startswith("image/svg+xml")
    r = c.post("/api/lens/calibrate", files={"video": ("b.mp4", board_video(tmp_path / "b.mp4").read_bytes(), "video/mp4")})
    assert r.status_code == 200 and r.json()["focal_px"] == pytest.approx(1000, rel=0.03)
    blank = tmp_path / "blank.mp4"
    out = cv2.VideoWriter(str(blank), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(10):
        out.write(np.full((240, 320, 3), 128, np.uint8))
    out.release()
    r = c.post("/api/lens/calibrate", files={"video": ("b.mp4", blank.read_bytes(), "video/mp4")})
    assert r.status_code == 400 and "found in 0 frames" in r.json()["detail"]
    r = c.post("/api/runs", files={"video": ("a.mp4", b"x", "video/mp4")}, data={"lens": '{"width": 3}'})
    assert r.status_code == 400 and r.json()["detail"].startswith("Lens calibration")


def test_the_pipeline_undistorts_every_track_and_says_so():
    from physicslens.perception.media import VideoInfo
    from physicslens.perception.sam_track import ObjectTrack
    from physicslens.pipeline import undistort_tracks
    cal = lens()
    pts = np.array([[1100.0, 100.0], [640.0, 360.0]])
    tr = ObjectTrack("1", np.arange(2), pts, np.zeros(2), np.ones(2), np.ones(2), np.ones(2))
    info = VideoInfo(width=W, height=H, fps=30.0, duration_s=2.0, frames=60, codec="h264")
    trail: list = []
    fixed = undistort_tracks({"1": tr}, cal.as_dict(), info, trail)
    assert np.abs(fixed["1"].centers_px[0] - pts[0]).max() > 5 and "Undistorted" in trail[0]["text"]
    other: list = []
    same = undistort_tracks({"1": tr}, cal.as_dict(), VideoInfo(width=1440, height=1080, fps=30.0, duration_s=2.0, frames=60, codec="h264"), other)
    assert same["1"] is tr and "different shape" in other[0]["text"]
