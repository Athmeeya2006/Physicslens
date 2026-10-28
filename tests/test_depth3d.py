"""3D from one phone: synthetic throws rendered through a pinhole camera must come back as the same physics."""

import json

import numpy as np
import pytest
from physicslens.costguard.mock import responses_result, sam_result
from physicslens.perception.sam_track import ObjectTrack
from physicslens.physics.depth3d import (
    analyze,
    check_frames,
    equivalent_diameter,
    focal_from_reference,
    for_run,
)
from physicslens.pipeline import RunOptions, fit_and_verify, full_track
from physicslens.pipeline import analyze as analyze_run

from tests.test_pipeline import (
    AGENT,
    CLIP,
    HYPOTHESES,
    SCENE,
    TRACK_OK,
    sam_text_as_object_1,
)

G = 9.81
F_PX = 1400.0  # a typical phone main camera at 1920 px wide
W, H = 1920, 1080
BALL_M = 0.068  # a tennis ball
FPS = 60.0


def render(p0: np.ndarray, v0: np.ndarray, seconds: float, seed: int = 0, noise_px: float = 1.0) -> ObjectTrack:
    """A drag-free throw seen by a pinhole camera, with 1 px noise on the ball's centre and on its diameter."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / FPS)
    xyz = p0 + np.outer(t, v0) + np.outer(0.5 * t**2, [0.0, -G, 0.0])
    z = xyz[:, 2]
    u = (W - 1) / 2 + xyz[:, 0] * F_PX / z + rng.normal(0, noise_px, len(t))
    v = (H - 1) / 2 - xyz[:, 1] * F_PX / z + rng.normal(0, noise_px, len(t))
    d = F_PX * BALL_M / z + rng.normal(0, noise_px, len(t))
    n = len(t)
    boxes = np.column_stack([u - d / 2, v - d / 2, u + d / 2, v + d / 2])
    return ObjectTrack("1", np.arange(n), np.column_stack([u, v]), np.zeros(n), d.copy(), d.copy(), np.ones(n),
                       boxes, np.pi * d**2 / 4)


def run3d(track: ObjectTrack, focal: float | None = F_PX, reps: int = 12) -> dict:
    return analyze(track.frames, track.centers_px, track.long_px, track.short_px, track.area_px, fps=FPS, width=W,
                   height=H, focal_px=focal, size_m=BALL_M, boxes=track.boxes, bootstrap_reps=reps)


@pytest.fixture(scope="module")
def toward_and_across():
    # Thrown across and away from the camera: depth goes from 2.0 m to about 3.3 m, the ball from 48 px to 29 px.
    return run3d(render(np.array([-0.9, 0.3, 2.0]), np.array([1.6, 3.5, 1.5]), 0.9))


def test_throw_in_depth_recovers_g(toward_and_across):
    r = toward_and_across
    assert r["status"] == "accepted" and r["equation"]["model_id"] == "free_fall_3d"
    assert r["g"]["value"] == pytest.approx(G, rel=0.03)
    assert r["g"]["ci95"][0] < r["g"]["value"] < r["g"]["ci95"][1]
    # Single frames scatter by a few percent in depth; the fitted path through them lands on the true one.
    z_true_end = 2.0 + 1.5 * (len(r["track"]) - 1) / FPS
    assert r["fit_track"][0][2] == pytest.approx(2.0, rel=0.03) and r["fit_track"][-1][2] == pytest.approx(z_true_end, rel=0.03)


def test_depth_error_follows_relative_diameter_error(toward_and_across):
    rows = np.array([row[:7] for row in toward_and_across["track"]], dtype=float)
    z, z_err = rows[:, 4], rows[:, 5]
    d = F_PX * BALL_M / z
    # 1 px of diameter noise: relative depth error is about 1 / d.
    assert z_err / z == pytest.approx(toward_and_across["diameter_noise_px"] / d, rel=0.05)
    assert toward_and_across["diameter_noise_px"] >= 1.0


def test_blurred_and_occluded_frames_are_rejected():
    tr = render(np.array([-0.9, 0.3, 2.0]), np.array([1.6, 3.5, 1.5]), 0.9, seed=3)
    d = equivalent_diameter(tr.area_px, tr.long_px, tr.short_px)
    long_px, short_px, area = tr.long_px.copy(), tr.short_px.copy(), tr.area_px.copy()
    blur, hidden = [10, 11, 30], [20, 40]
    long_px[blur] *= 1.5  # streaked along the motion
    area[hidden] *= 0.5  # a hand covers half the ball
    check = check_frames(equivalent_diameter(area, long_px, short_px), long_px, short_px)
    assert check.blurred == 3 and check.occluded == 2
    assert not check.keep[blur + hidden].any() and check.keep.sum() == len(d) - 5
    # A ball cut by the frame edge is partly hidden too.
    boxes = tr.boxes.copy()
    boxes[5, 0] = 0.0
    assert not check_frames(d, tr.long_px, tr.short_px, boxes, (W, H)).keep[5]

    hurt = ObjectTrack("1", tr.frames, tr.centers_px, tr.angles, long_px, short_px, tr.confidence, tr.boxes, area)
    r = run3d(hurt, reps=4)
    assert r["frames_rejected"] == {"blurred": 3, "occluded": 2}
    assert r["frames_interpolated"] == 5  # single-frame gaps are bridged, as in the 2D track
    assert r["g"]["value"] == pytest.approx(G, rel=0.03)


def test_too_few_sharp_frames_skips_with_a_note():
    tr = render(np.array([-0.9, 0.3, 2.0]), np.array([1.6, 3.5, 1.5]), 0.9)
    long_px = tr.long_px * 1.6  # every frame blurred
    blurred = ObjectTrack("1", tr.frames, tr.centers_px, tr.angles, long_px, tr.short_px, tr.confidence, tr.boxes, tr.area_px)
    out = for_run(blurred, tr.frames, fps=FPS, width=W, height=H, focal_px=F_PX, size_m=BALL_M,
                  moving_object="tennis ball", family="projectile")
    assert out["status"] == "skipped" and "blurred" in out["reason"]


def test_without_focal_length_3d_is_skipped():
    tr = render(np.array([-0.9, 0.3, 2.0]), np.array([1.6, 3.5, 1.5]), 0.9)
    out = for_run(tr, tr.frames, fps=FPS, width=W, height=H, focal_px=None, size_m=BALL_M,
                  moving_object="tennis ball", family="projectile")
    assert out == {"status": "skipped", "reason": out["reason"]} and "focal length" in out["reason"]
    assert "2D analysis" in out["reason"]
    not_ball = for_run(tr, tr.frames, fps=FPS, width=W, height=H, focal_px=F_PX, size_m=BALL_M,
                       moving_object="toy car", family="projectile")
    assert not_ball["status"] == "skipped" and "ball" in not_ball["reason"]
    no_size = for_run(tr, tr.frames, fps=FPS, width=W, height=H, focal_px=F_PX, size_m=None,
                      moving_object="ball", family="projectile")
    assert no_size["status"] == "skipped" and "diameter" in no_size["reason"]


def test_throw_across_the_image_plane_matches_2d():
    tr = render(np.array([-0.9, 0.0, 2.5]), np.array([2.0, 3.5, 0.0]), 0.9, seed=5)
    r3 = run3d(tr, reps=4)
    # The 2D path with the same ball as its scale, as the app does when the user gives the object's size.
    ppm = float(np.median(tr.long_px)) / BALL_M
    traj, _ = full_track(tr, FPS, ppm)
    _, v2 = fit_and_verify(traj, "projectile", ["free_fall", "linear_drag", "quadratic_drag"])
    assert v2.status == "accepted" and r3["status"] == "accepted"
    g2, g3 = v2.winner.params["g"], r3["g"]["value"]
    assert g2 == pytest.approx(G, rel=0.03) and g3 == pytest.approx(G, rel=0.03)
    assert g3 == pytest.approx(g2, rel=0.02)


def test_focal_calibration_from_a4_sheet():
    # An A4 sheet 1.5 m away whose long side spans 277.2 px.
    assert focal_from_reference(277.2, 1.5, 0.297) == pytest.approx(1400.0)
    with pytest.raises(ValueError):
        focal_from_reference(0, 1.5, 0.297)


def test_pipeline_runs_2d_and_explains_why_3d_was_skipped(harness, tmp_path):
    harness.sdk("meta_api").script.extend([
        responses_result(text=json.dumps(SCENE)), sam_result(frames=60, text=sam_text_as_object_1()),
        responses_result(text=json.dumps(TRACK_OK)), responses_result(text=json.dumps(HYPOTHESES)),
        responses_result(text=json.dumps(AGENT))])
    options = RunOptions(pendulum_length_m=0.6, object_size_m=0.05, depth_3d=True, focal_px=1400.0)
    result = analyze_run(harness.client, CLIP, tmp_path / "run", "run-depth", options)
    assert result["status"] in ("accepted", "ambiguous") and result["family"] == "pendulum"
    assert result["depth3d"]["status"] == "skipped" and "fitted as pendulum" in result["depth3d"]["reason"]
