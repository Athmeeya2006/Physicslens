"""Scale-free audit checks: free fall compared with itself, on synthetic tracks in arbitrary units (no scale)."""

import numpy as np
from physicslens.physics.audit import AuditInput, audit, check_scale_free, noise_sigma
from physicslens.sigproc.trajectory import Trajectory

from .test_audit import bouncing, pendulum, traj_of

FPS = 60.0


def inp(traj: Trajectory, family: str = "projectile") -> AuditInput:
    # "none": no real scale, so only checks that need none can speak.
    return AuditInput(traj=traj, family=family, calibration_source="none", px_per_m=1.0)


def by_title(traj: Trajectory, family: str = "projectile") -> dict[str, dict]:
    checks = check_scale_free(inp(traj, family), noise_sigma(traj.xy))
    return {c.title: {"status": c.status, "text": c.text} for c in checks}


def flight(seconds: float = 1.8, g: float = 9.81, vx: float = 1.5, vy: float = 4.0, drag: float = 0.0,
           noise: float = 0.002, seed: int = 1) -> Trajectory:
    """A thrown ball, with optional linear drag (rate per second): real motion."""
    t = np.arange(0, seconds, 1 / FPS)
    k = drag
    if k > 0:
        x = vx / k * (1 - np.exp(-k * t))
        vt = g / k
        y = (vy + vt) / k * (1 - np.exp(-k * t)) - vt * t
    else:
        x, y = vx * t, vy * t - 0.5 * g * t * t
    return traj_of(t, np.column_stack([x, 10 + y]), noise, seed, FPS)


def speeds_up_then_slows(seconds: float = 1.8, g: float = 9.81) -> Trajectory:
    """Falls freely for half the clip, then slows in mid-air with nothing touching it: a generated-video failure."""
    t = np.arange(0, seconds, 1 / FPS)
    half = seconds / 2
    ay = np.where(t < half, -g, 0.6 * g)
    vy = np.concatenate([[0.0], np.cumsum(ay[:-1]) / FPS])
    y = np.concatenate([[0.0], np.cumsum(vy[:-1]) / FPS])
    return traj_of(t, np.column_stack([1.5 * t, 20 + y]), 0.002, 2, FPS)


def test_a_real_parabola_passes():
    c = by_title(flight())
    assert c["Free fall keeps one acceleration"]["status"] == "pass"
    assert c["Free-flight path is a parabola"]["status"] == "pass"
    assert c["Falling objects don't gain energy"]["status"] == "pass"
    assert c["Each bounce no higher than the last"]["status"] == "skipped"  # no bounce: can't judge, not a pass
    assert audit(inp(flight()), scale_free=True)["verdict"] == "consistent"


def test_a_ball_with_some_air_drag_passes():
    c = by_title(flight(drag=0.1))
    assert c["Free fall keeps one acceleration"]["status"] == "pass"
    assert c["Free-flight path is a parabola"]["status"] == "pass"


def test_speeding_up_then_slowing_in_mid_air_fails():
    traj = speeds_up_then_slows()
    c = by_title(traj)
    assert c["Free fall keeps one acceleration"]["status"] == "violation"
    assert audit(inp(traj), scale_free=True)["verdict"] == "violates_physics"


def test_a_bounce_gaining_height_fails():
    c = by_title(bouncing(e=1.1, seconds=2.5, height=0.6))
    assert c["Each bounce no higher than the last"]["status"] == "violation"


def test_a_bounce_losing_height_passes():
    c = by_title(bouncing(e=0.75))
    assert c["Each bounce no higher than the last"]["status"] == "pass"


def test_rolling_off_an_edge_is_not_a_change_of_acceleration():
    """Losing a support starts a fall with no impact: the acceleration grows, which is allowed."""
    t = np.arange(0, 2.0, 1 / FPS)
    edge = 0.8
    y = np.where(t < edge, 0.0, -0.5 * 9.81 * (t - edge) ** 2)
    traj = traj_of(t, np.column_stack([1.0 * t, 20 + y]), 0.002, 3, FPS)
    c = by_title(traj)
    assert c["Free fall keeps one acceleration"]["status"] != "violation"
    assert c["Falling objects don't gain energy"]["status"] != "violation"


def test_a_too_short_clip_cannot_be_judged():
    short = flight(seconds=0.15)
    c = by_title(short)
    assert all(v["status"] == "skipped" and v["text"].startswith("Can't judge") for v in c.values())
    assert audit(inp(flight(seconds=0.5)), scale_free=True)["verdict"] == "cant_judge"


def test_a_swing_is_not_judged_as_free_fall():
    c = by_title(pendulum(), family="pendulum")
    assert all(v["status"] == "skipped" and v["text"].startswith("Can't judge") for v in c.values())


def test_a_partly_hidden_object_cannot_be_judged():
    """A mask shrinking behind an obstacle moves its centre with no force: not evidence either way."""
    traj = speeds_up_then_slows()
    areas = np.where(traj.t < 0.9, 400.0, 200.0)
    a = AuditInput(traj=traj, family="projectile", calibration_source="none", px_per_m=1.0, areas_px=areas)
    checks = {c.title: c for c in check_scale_free(a, noise_sigma(traj.xy))}
    assert checks["Free fall keeps one acceleration"].status == "skipped"
    assert checks["Free fall keeps one acceleration"].text.startswith("Can't judge")
