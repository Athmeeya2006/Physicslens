"""Live mode for every family the page offers, not just the pendulum: a spring, a slide to a stop and a throw, each
streamed as synthetic frames with SAM answering from the true positions (the recorded mask of the same 43 px ball)."""
import json
import re

import cv2
import numpy as np
import pytest
from physicslens.costguard.mock import responses_result
from physicslens.live import LiveConfig, LiveSession

from .test_live import FPS, fixture_lines
from .test_pipeline import HYPOTHESES, SCENE

W, H, R = 640, 360, 21


def motion(kind: str, n: int) -> np.ndarray:
    t = np.arange(n) / FPS
    if kind == "oscillator":  # a ball bobbing on a spring at 1 Hz, slowly damped
        return np.column_stack([np.full(n, 320.0), 180 + 70 * np.exp(-0.15 * t) * np.cos(2 * np.pi * t)])
    if kind == "line":  # a block sliding to a stop, constant deceleration
        v0, a = 260.0, 90.0
        ts = np.minimum(t, v0 / a)
        return np.column_stack([60 + v0 * ts - 0.5 * a * ts**2, np.full(n, 200.0)])
    s = 25.0  # a throw, 25 px per metre, real gravity
    return np.column_stack([60 + 3.0 * s * t, 300 - (7.0 * t - 4.905 * t**2) * s])


class TrueSam:
    def __init__(self, pos: np.ndarray):
        self.pos, self.mask = pos, fixture_lines()[0]

    def __call__(self, path, prompt, indices):
        out = []
        for k, g in enumerate(indices):
            x1, y1 = (round(float(v) - R - 0.5) for v in self.pos[g])
            line = re.sub(r"^<\d+f>0<", f"<{k}f>1<", self.mask)
            out.append(re.sub(r"x1=\d+;y1=\d+;x2=\d+;y2=\d+", f"x1={x1};y1={y1};x2={x1 + 43};y2={y1 + 43}", line))
        return "\n".join(out)


def stream(harness, tmp_path, kind: str, n: int, size_m: float | None, pos: np.ndarray | None = None) -> dict:
    pos = motion(kind, n) if pos is None else pos
    harness.sdk("meta_api").script.extend([responses_result(text=json.dumps(SCENE)),
                                           responses_result(text=json.dumps(HYPOTHESES))])
    sent: list[dict] = []
    session = LiveSession(harness.client, tmp_path / kind, f"live-{kind}", LiveConfig(fps=FPS, object_size_m=size_m),
                          sent.append, sam_fn=TrueSam(pos), synchronous=True)
    jpegs = []
    for x, y in pos:
        img = np.full((H, W, 3), (40, 30, 25), np.uint8)
        cv2.circle(img, (round(float(x)), round(float(y))), R, (235, 235, 235), -1, cv2.LINE_AA)
        jpegs.append(cv2.imencode(".jpg", img)[1].tobytes())
    session.add_frame(jpegs[0])
    session.select(*pos[0])
    session.choose_family_now(kind)  # the user's tap on "On a spring", "Rolling or sliding", "Thrown or falling"
    for j in jpegs[1:]:
        session.add_frame(j)
    fits = [m for m in sent if m["type"] == "fit"]
    assert fits, f"live mode never fitted the {kind}"
    stream.session = session  # for the save test
    return fits[-1]


@pytest.mark.slow
def test_live_fits_a_spring(harness, tmp_path):
    fit = stream(harness, tmp_path, "oscillator", 210, 0.1)
    assert fit["status"] == "accepted" and fit["equation"]["model_id"] == "damped_harmonic"
    p = {q["name"]: q["value"] for q in fit["parameters"]}
    assert p["omega"] == pytest.approx(2 * np.pi, rel=0.01) and p["zeta"] * p["omega"] == pytest.approx(0.15, rel=0.2)


@pytest.mark.slow
def test_live_fits_a_slide_to_a_stop(harness, tmp_path):
    fit = stream(harness, tmp_path, "line", 120, 0.1)
    assert fit["status"] == "accepted" and fit["equation"]["model_id"] == "line_constant_accel"


@pytest.mark.slow
def test_live_fits_a_throw_within_its_one_second_flight(harness, tmp_path):
    # Before: live mode waited for 2.5 s of track, longer than a throw is in the air, so a throw never got a fit.
    fit = stream(harness, tmp_path, "projectile", 45, 1.72)
    assert fit["status"] == "accepted" and fit["equation"]["model_id"] == "free_fall"
    assert {q["name"]: q["value"] for q in fit["parameters"]}["g"] == pytest.approx(9.81, rel=0.05)


@pytest.mark.slow
def test_an_unmeasured_slow_throw_takes_gravity_as_its_ruler(harness, tmp_path):
    # A slow-motion throw with nothing of known size in view fell at 0.04 "m/s²" on the nominal scale, below any
    # allowed g, and live mode showed no law (and a forecast error of 1506880239378%). Gravity now sets the scale.
    t = np.arange(150) / FPS
    pos = np.column_stack([60 + 90 * t, 300 - (110 * t - 18.3 * t**2)])
    fit = stream(harness, tmp_path, "projectile", 150, None, pos)
    assert fit["status"] == "accepted" and fit.get("scale_basis") == "gravity"
    g = next(q for q in fit["parameters"] if q["name"] == "g")
    assert g["value"] == 9.81 and "ruler" in g["assumed"]


@pytest.mark.slow
def test_a_live_session_is_saved_as_a_normal_result(harness, tmp_path):
    import json as _json

    from physicslens.export.script import python_script
    stream(harness, tmp_path, "oscillator", 120, 0.1)
    session = stream.session
    run_dir = tmp_path / "saved"
    run_dir.mkdir()
    result = session.save(run_dir, keep_video=False)
    stored = _json.loads((run_dir / "result.json").read_text())
    assert stored["source"] == "live session" and stored["video_kept"] is False and not (run_dir / "clip.mp4").exists()
    assert stored["equation"]["model_id"] == result["fit_state"]["model_id"] and stored["real_track_px"]
    assert "def rhs" in python_script(stored, 2.0)  # exports work from the saved fit
    session.save(run_dir, keep_video=True)
    assert (run_dir / "clip.mp4").stat().st_size > 0


def test_a_slide_is_fitted_without_the_rest_after_it():
    """Pushed, slides, stops, lies still: the law is fitted on the slide (constant deceleration), not on the rest."""
    import numpy as np
    from physicslens.live import moving_stretch
    from physicslens.sigproc.trajectory import Trajectory
    t = np.arange(250) / 25  # 3.3 s of sliding then 6.7 s at rest
    ts = np.minimum(t, 200 / 60)
    x = (200 * ts - 0.5 * 60 * ts * ts) / 200
    traj = Trajectory(t, np.column_stack([x, np.zeros_like(x)]), np.arange(250), 25.0)
    kept = moving_stretch(traj)
    assert 2.8 < kept.t[-1] - kept.t[0] < 3.5 and kept.t[0] < 0.2
    glide = Trajectory(t, np.column_stack([0.3 * t, np.zeros_like(t)]), np.arange(250), 25.0)
    assert len(moving_stretch(glide)) == len(glide)  # never at rest: all of it
