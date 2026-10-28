"""Physics audit: synthetic clips that break one law each are caught; clean clips with realistic noise pass."""

import json

import numpy as np
import pytest
from physicslens.costguard.mock import responses_result, sam_result
from physicslens.physics.audit import AuditInput, LawEvidence, audit, pm
from physicslens.pipeline import RunOptions, analyze
from physicslens.sigproc.trajectory import Trajectory
from scipy.integrate import solve_ivp

from .test_pipeline import (
    AGENT,
    CLIP,
    HYPOTHESES,
    SCENE,
    TRACK_OK,
    sam_text_as_object_1,
)

PPM = 500.0  # pixels per meter, as with an A4 sheet filling a fifth of a 1080p frame


def traj_of(t: np.ndarray, xy: np.ndarray, noise: float, seed: int, fps: float) -> Trajectory:
    rng = np.random.default_rng(seed)
    return Trajectory(t, xy + rng.normal(0, noise, xy.shape), np.arange(len(t)), fps)


def bouncing(g: float = 9.81, e: float = 0.75, noise: float = 0.002, fps: float = 60, seed: int = 0,
             seconds: float = 3.0, height: float = 1.0, vx: float = 0.4) -> Trajectory:
    """Exact parabolas between floor impacts; any e, including the impossible e > 1."""
    t = np.arange(0, seconds, 1 / fps)
    xy = np.zeros((len(t), 2))
    tc, y0, vy = 0.0, height, 0.0
    hits = []
    while tc < seconds:
        disc = vy * vy + 2 * g * y0
        hit = tc + (vy + np.sqrt(disc)) / g
        if vy < 0.05 and y0 == 0.0 and tc > 0:  # hops too small to matter: the ball rests
            hits.append((tc, 0.0, 0.0, np.inf))
            break
        hits.append((tc, y0, vy, hit))
        vy = e * np.sqrt(disc)
        tc, y0 = hit, 0.0
    for i, ti in enumerate(t):
        start, yy, v, end = next(h for h in hits if h[0] <= ti < h[3])
        dt = ti - start
        xy[i] = (vx * ti, yy if end == np.inf else yy + v * dt - 0.5 * g * dt * dt)
    return traj_of(t, xy, noise, seed, fps)


def pendulum(theta0: float = 0.5, length: float = 0.6, gamma: float = 0.05, noise: float = 0.002, fps: float = 30,
             seed: int = 0, seconds: float = 10.0, g: float = 9.81) -> Trajectory:
    t = np.arange(0, seconds, 1 / fps)
    sol = solve_ivp(lambda _t, y: [y[1], -g / length * np.sin(y[0]) - gamma * y[1]], (0, seconds), [theta0, 0.0],
                    t_eval=t, rtol=1e-10, atol=1e-12)
    th = sol.y[0]
    return traj_of(t, np.column_stack([length * np.sin(th), 1.0 - length * np.cos(th)]), noise, seed, fps)


def spring(omega: float = 4.0, zeta: float = 0.02, amp: float = 0.1, noise: float = 0.002, fps: float = 30, seed: int = 0,
           seconds: float = 10.0, drift: float = 0.0, growth: float = 0.0) -> Trajectory:
    """Vertical oscillation; drift: relative frequency change over the clip, growth: amplitude rate (1/s)."""
    t = np.arange(0, seconds, 1 / fps)
    w = omega * (1 + drift * t / seconds)
    phase = np.cumsum(w) / fps
    s = amp * np.exp((growth - zeta * omega) * t) * np.cos(phase)
    return traj_of(t, np.column_stack([0.002 * np.sin(t), s]), noise, seed, fps)


def run(traj: Trajectory, family: str, source: str = "a4_sheet", **kw) -> dict:
    return audit(AuditInput(traj=traj, family=family, calibration_source=source, px_per_m=PPM, **kw))


def violated(result: dict) -> set[str]:
    return {v["title"] for v in result["violations"]}


def test_wrong_gravity_with_a_real_scale():
    r = run(bouncing(g=14.2), "projectile")
    assert r["verdict"] == "violates_physics"
    g = next(c for c in r["checks"] if c["title"] == "Gravity against Earth's")
    assert g["status"] == "violation" and g["value"] == pytest.approx(14.2, rel=0.03) and g["z"] > 5
    assert "m/s²" in r["headline"]


def test_wrong_gravity_from_the_verified_law():
    law = LawEvidence("accepted", "projectile_no_drag", 0.02, 1.0, g=14.2, g_std=0.6)
    r = run(pendulum(), "pendulum", law=law)
    assert "Gravity against Earth's" in violated(r)
    assert "14.2 ± 0.6 m/s²" in r["headline"]


def test_wrong_gravity_is_not_judged_without_a_scale():
    r = run(bouncing(g=14.2), "projectile", source="none")
    assert r["verdict"] == "consistent"  # one g, just not 9.81: in pixels nothing says it is wrong


def test_gravity_changing_between_flights_without_a_scale():
    a, b = bouncing(g=9.81, seconds=3.0, seed=3), bouncing(g=16.0, seconds=3.0, seed=4, height=0.6)
    # Second half of the clip follows a world with a stronger pull.
    cut = int(np.searchsorted(a.t, 1.4))
    second = b.xy[np.searchsorted(b.t, 0.0):] + np.array([a.xy[cut, 0], 0.0])
    xy = np.vstack([a.xy[:cut], second[: len(a.t) - cut]])
    r = run(Trajectory(a.t, xy, a.frames, a.fps), "projectile", source="none")
    assert r["verdict"] == "violates_physics"
    assert "One gravity for every flight" in violated(r)


def test_bounce_gaining_energy():
    r = run(bouncing(e=1.1, height=0.5, seconds=2.5), "projectile")
    assert r["verdict"] == "violates_physics"
    e = next(c for c in r["violations"] if c["title"] == "Bounces lose energy")
    assert e["value"] == pytest.approx(1.1, abs=0.04) and e["z"] > 5
    assert "energy" in e["text"]


def test_period_drift():
    r = run(spring(drift=0.15), "oscillator")
    assert r["verdict"] == "violates_physics"
    assert "Steady period" in violated(r)


def test_rising_amplitude():
    r = run(spring(growth=0.12, zeta=0.0), "oscillator")
    assert r["verdict"] == "violates_physics"
    assert "Swing doesn't grow" in violated(r)


def test_regime_change_excuses_a_period_change():
    r = run(spring(drift=0.15), "oscillator", regime_change_s=5.0)
    assert "Steady period" not in violated(r)


def test_teleport():
    tr = pendulum(seed=5)
    xy = tr.xy.copy()
    xy[150:, 0] += 0.15  # the bob reappears 15 cm to the side and swings on from there
    r = run(Trajectory(tr.t, xy, tr.frames, tr.fps), "pendulum", object_size_px=0.05 * PPM)
    assert r["verdict"] == "violates_physics"
    jump = next(c for c in r["violations"] if c["title"] == "No teleporting")
    assert jump["value"] == pytest.approx(0.15, abs=0.02)


def test_single_frame_glitch_is_not_a_teleport():
    tr = pendulum(seed=6)
    xy = tr.xy.copy()
    xy[150, 0] += 0.15  # one wrong mask, then back on the object
    r = run(Trajectory(tr.t, xy, tr.frames, tr.fps), "pendulum")
    assert "No teleporting" not in violated(r)


def test_morphing_mask():
    tr = pendulum(seed=7)
    rng = np.random.default_rng(7)
    areas = 900 * (1 + rng.normal(0, 0.05, len(tr)))
    areas[160:] *= 2.2
    r = run(tr, "pendulum", areas_px=areas)
    assert "No morphing" in violated(r)
    steady = run(tr, "pendulum", areas_px=900 * (1 + rng.normal(0, 0.05, len(tr))))
    assert "No morphing" not in violated(steady)


def test_no_law_for_a_named_system():
    law = LawEvidence("no_fit", None, 0.9, 40.0)
    assert "A law forecasts the motion" in violated(run(pendulum(), "pendulum", law=law))
    # Free-form motion without a law is not an accusation: a push or a walk has none either.
    r = run(spring(), "line", law=law)
    assert r["verdict"] == "cant_judge" and not r["violations"]


def test_too_short_clip():
    r = run(bouncing(seconds=1.0), "projectile")
    assert r["verdict"] == "cant_judge" and "lasts" in r["headline"]


@pytest.mark.parametrize("kw,phrase", [({"frame_rate_varies": True}, "variable frame rate"),
                                       ({"coverage": 0.4}, "tracked in only"),
                                       ({"confidence": np.full(10, 0.2)}, "unsure")])
def test_preconditions_give_cant_judge(kw, phrase):
    r = run(bouncing(g=14.2), "projectile", **kw)
    assert r["verdict"] == "cant_judge" and phrase in r["headline"] and not r["violations"]


def test_moving_camera_gives_cant_judge():
    tr = pendulum()
    ref = np.column_stack([np.linspace(100, 160, len(tr)), np.full(len(tr), 300.0)])
    r = run(tr, "pendulum", reference_px=ref, reference_size_px=120.0)
    assert r["verdict"] == "cant_judge" and "camera" in r["headline"]
    still = run(tr, "pendulum", reference_px=np.full((len(tr), 2), 200.0) + np.random.default_rng(1).normal(0, 1, (len(tr), 2)),
                reference_size_px=120.0)
    assert still["verdict"] == "consistent"


def clean_clip(i: int) -> tuple[Trajectory, str]:
    """A realistic, physical clip: random system, frame rate, noise and setup."""
    rng = np.random.default_rng(1000 + i)
    fps = float(rng.choice([30, 60]))
    noise = float(rng.uniform(0.001, 0.004))
    kind = i % 3
    if kind == 0:
        return bouncing(e=rng.uniform(0.6, 0.9), noise=noise, fps=fps, seed=i, height=rng.uniform(0.7, 1.3),
                        vx=rng.uniform(-0.6, 0.6)), "projectile"
    if kind == 1:
        return pendulum(theta0=rng.uniform(0.25, 0.9), length=rng.uniform(0.3, 1.0), gamma=rng.uniform(0.0, 0.15),
                        noise=noise, fps=fps, seed=i), "pendulum"
    return spring(omega=rng.uniform(3, 9), zeta=rng.uniform(0.0, 0.05), amp=rng.uniform(0.05, 0.15), noise=noise,
                  fps=fps, seed=i), "oscillator"


def test_clean_clips_pass():
    verdicts = [run(*clean_clip(i))["verdict"] for i in range(40)]
    assert verdicts.count("violates_physics") == 0
    assert verdicts.count("consistent") >= 38, verdicts


def test_value_formatting():
    assert pm(14.234, 0.61, "m/s²") == "14.2 ± 0.6 m/s²"
    assert pm(1.2, None) == "1.2"


def test_every_analysis_carries_an_audit(harness, tmp_path):
    harness.sdk("meta_api").script.extend([
        responses_result(text=json.dumps(SCENE)), sam_result(frames=60, text=sam_text_as_object_1()),
        responses_result(text=json.dumps(TRACK_OK)), responses_result(text=json.dumps(HYPOTHESES)),
        responses_result(text=json.dumps(AGENT))])
    # No scale given: the smoke clip is drawn in pixels, so any length typed in would be invented.
    result = analyze(harness.client, CLIP, tmp_path / "run", "run-audit", RunOptions())
    a = result["audit"]
    assert a["verdict"] in ("consistent", "cant_judge"), a["headline"]
    assert {c["title"] for c in a["checks"]} >= {"No teleporting", "No morphing"} or a["reasons"]


def test_audit_endpoint_is_the_runs_endpoint():
    from physicslens import app as app_module

    routes = {r.path: r for r in app_module.app.routes if hasattr(r, "methods")}
    assert routes["/api/audit"].endpoint is routes["/api/runs"].endpoint
