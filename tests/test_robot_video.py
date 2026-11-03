"""Video check of the robot arm: camera projection and comparison on synthetic data (no video, no model calls)."""
import numpy as np
import pytest
from physicslens.robot import video_check as V

SIZE = (640, 480)
TRUE_CAM = V.Camera(np.array([2.9, 0.1, -0.15]), np.array([0.02, -0.05, 0.55]), 520.0, (320.0, 240.0))
TRUE_OFFSET = np.array([0.01, -0.06, 0.02])


def synthetic_path(n: int = 300, seed: int = 0):
    """A gripper wandering over a 30 cm workspace and turning, like the arm in the pick-and-place episodes."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    pos = np.c_[0.15 * np.sin(2 * np.pi * t), 0.1 + 0.12 * np.cos(3 * np.pi * t), 0.05 + 0.08 * np.sin(5 * t) ** 2]
    pos += rng.normal(0, 0.005, pos.shape)
    rot = np.stack([V.rodrigues(np.array([0.8 * np.sin(4 * a), 0.6 * np.cos(3 * a), 1.2 * a])) for a in t])
    return pos, rot


def test_rodrigues_round_trip():
    r = np.array([0.3, -1.1, 0.7])
    assert np.allclose(V.rotation_to_rvec(V.rodrigues(r)), r)
    assert np.allclose(V.rodrigues(r) @ V.rodrigues(r).T, np.eye(3))


def test_dlt_recovers_an_exact_projection():
    pos, rot = synthetic_path()
    x = V.tracked_points(pos, rot, TRUE_OFFSET)
    uv, _ = TRUE_CAM.project(x)
    cam = V.camera_from_projection(V.dlt(x, uv), TRUE_CAM.c, x)
    assert cam.f == pytest.approx(TRUE_CAM.f, rel=1e-3)
    assert np.allclose(cam.project(x)[0], uv, atol=1e-3)


def test_known_projection_and_offset_are_recovered():
    pos, rot = synthetic_path()
    uv, _ = TRUE_CAM.project(V.tracked_points(pos, rot, TRUE_OFFSET))
    uv = uv + np.random.default_rng(1).normal(0, 0.5, uv.shape)
    fit = V.fit_camera(pos, rot, uv, SIZE)
    assert fit.camera.f == pytest.approx(TRUE_CAM.f, rel=0.02)
    assert np.allclose(fit.offset, TRUE_OFFSET, atol=0.003)
    assert np.allclose(fit.camera.tvec, TRUE_CAM.tvec, atol=0.01)
    assert fit.rms_px < 1.0
    # Checked on a path it never saw.
    pos2, rot2 = synthetic_path(seed=7)
    held = V.pixel_errors(fit.camera.project(V.tracked_points(pos2, rot2, fit.offset))[0],
                          TRUE_CAM.project(V.tracked_points(pos2, rot2, TRUE_OFFSET))[0])
    assert np.median(held) < 0.5


def test_fit_resists_a_few_mistracked_frames():
    pos, rot = synthetic_path()
    uv, _ = TRUE_CAM.project(V.tracked_points(pos, rot, TRUE_OFFSET))
    uv[::25] += 60.0  # every 25th frame on something else entirely
    fit = V.fit_camera(pos, rot, uv, SIZE)
    assert fit.median_px < 0.5
    assert fit.camera.f == pytest.approx(TRUE_CAM.f, rel=0.02)


def test_perturbed_path_shows_the_right_pixel_and_mm_error():
    pos, rot = synthetic_path()
    x = V.tracked_points(pos, rot, TRUE_OFFSET)
    uv, depth = TRUE_CAM.project(x)
    shifted = uv + np.array([3.0, 4.0])
    err = V.pixel_errors(uv, shifted)
    assert np.allclose(err, 5.0)
    mm = V.mm_errors(err, depth, TRUE_CAM.f)
    assert np.allclose(mm, 5000.0 * depth / TRUE_CAM.f)
    # A 1 cm sideways move of the 3D point at depth Z shows as f * 0.01 / Z pixels.
    xc = TRUE_CAM.to_camera(x)
    moved = (xc + np.array([0.01, 0.0, 0.0])) @ V.rodrigues(TRUE_CAM.rvec)  # back to world coordinates
    moved -= TRUE_CAM.tvec @ V.rodrigues(TRUE_CAM.rvec)
    e2 = V.pixel_errors(TRUE_CAM.project(moved)[0], uv)
    assert np.allclose(e2, TRUE_CAM.f * 0.01 / depth, rtol=1e-6)
    assert np.allclose(V.mm_errors(e2, depth, TRUE_CAM.f), 10.0, rtol=1e-6)


def test_shift_frames_and_summaries():
    q = np.c_[np.arange(10.0), 2 * np.arange(10.0)]
    assert np.allclose(V.shift_frames(q, 1.5)[2], [3.5, 7.0])
    assert np.allclose(V.shift_frames(q, -3)[0], [0.0, 0.0])
    s = V.summary(np.array([1.0, 2.0, 3.0, 4.0]))
    assert s["median"] == 2.5 and s["max"] == 4.0
    ti = V.t_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    assert ti["mean"] == 3.0 and ti["lo"] == pytest.approx(3.0 - 2.776 * np.sqrt(2.5) / np.sqrt(5), rel=1e-3)
    bins = V.binned(np.array([1.0, 1.0, 9.0]), np.array([0.0, 0.0, 1.0]), [0.0, 0.5, 1.0])
    assert [b["median_err"] for b in bins] == [1.0, 9.0]
    assert V.rank_correlation(np.ones(3), np.arange(3.0)) is None


def test_focal_length_can_be_fixed_and_the_mm_scale_survives():
    pos, rot = synthetic_path()
    uv, depth = TRUE_CAM.project(V.tracked_points(pos, rot, TRUE_OFFSET))
    fixed = V.fit_camera(pos, rot, uv, SIZE, focal_px=(800.0, 800.0))
    assert fixed.camera.f == pytest.approx(800.0, rel=1e-6)
    # A wrong focal length costs pixels, but the millimetres per pixel at the gripper stay close to the truth.
    _, z = fixed.camera.project(V.tracked_points(pos, rot, fixed.offset))
    assert np.median(z / fixed.camera.f) == pytest.approx(np.median(depth / TRUE_CAM.f), rel=0.15)
