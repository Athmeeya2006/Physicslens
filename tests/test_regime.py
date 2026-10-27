"""Regime changes: a shortened string or a jump in damping mid-recording, plus no-change controls."""

import numpy as np
import pytest
from physicslens.agent.schemas import Scene
from physicslens.live import LiveConfig, LiveSession
from physicslens.perception.calibration import Calibration
from physicslens.physics.regime import detect_change
from physicslens.pipeline import find_regimes
from physicslens.sigproc.trajectory import Trajectory
from scipy.integrate import solve_ivp

from .test_pipeline import SCENE

FPS = 30
G = 9.81
PIVOT = (0.0, 1.0)


def pendulum(segments, theta0=0.6, seed=2) -> Trajectory:
    """segments: (seconds, length, damping); the swing continues through each change, pivot fixed."""
    rng = np.random.default_rng(seed)
    parts, state = [], [theta0, 0.0]
    for secs, length, gamma in segments:
        t = np.arange(0, secs, 1 / FPS)
        sol = solve_ivp(lambda _t, y, L=length, g=gamma: [y[1], -G / L * np.sin(y[0]) - g * y[1]], (0, secs), state,
                        t_eval=t, rtol=1e-10, atol=1e-12, dense_output=True)
        th = sol.y[0]
        parts.append(np.column_stack([PIVOT[0] + length * np.sin(th), PIVOT[1] - length * np.cos(th)]))
        state = list(sol.sol(secs))
    xy = np.vstack(parts) + rng.normal(0, 0.002, (sum(len(p) for p in parts), 2))
    return Trajectory(np.arange(len(xy)) / FPS, xy, np.arange(len(xy)), FPS)


@pytest.mark.parametrize("segments,theta0", [([(10, 0.6, 0.08)], 0.6), ([(10, 0.6, 0.0)], 0.9)])
def test_no_false_alarm(segments, theta0):
    assert detect_change(pendulum(segments, theta0), "pendulum", "pendulum_linear_damping") is None


def test_shortened_string_detected():
    ch = detect_change(pendulum([(6, 0.6, 0.05), (5, 0.35, 0.05)]), "pendulum", "pendulum_linear_damping")
    assert ch is not None and abs(ch.t_estimate - 6.0) < 0.6 and ch.t >= 6.0


def test_each_regime_fitted_separately():
    traj = pendulum([(6, 0.6, 0.05), (5, 0.35, 0.05)])
    change, regimes = find_regimes(traj, "pendulum", ["pendulum", "pendulum_linear_damping"], np.array([0.0, 0.0]), 1000.0)
    assert change is not None and len(regimes) == 2
    first, second = regimes
    # g is a property of the world, the length a property of the setup: only the length should change.
    assert first["geometry"]["length_m"] == pytest.approx(0.6, rel=0.03)
    assert second["geometry"]["length_m"] == pytest.approx(0.35, rel=0.03)
    assert first["params"]["g"] == pytest.approx(G, rel=0.03) and second["params"]["g"] == pytest.approx(G, rel=0.03)
    assert first["ghost"] and second["ghost"][0][0] > first["ghost"][-1][0]


def test_live_session_switches_to_the_new_regime(harness, tmp_path):
    sent: list[dict] = []
    session = LiveSession(harness.client, tmp_path / "live", "live-regime", LiveConfig(fps=FPS), sent.append, synchronous=True)
    session.scene = Scene.model_validate(SCENE)
    session.family = "pendulum"
    session.candidates = ["pendulum", "pendulum_linear_damping"]
    session.calibration = Calibration(px_per_m=500.0, rel_std=0.0, source="test")
    traj = pendulum([(7, 0.6, 0.05), (7, 0.35, 0.05)])
    px = np.column_stack([320 + traj.xy[:, 0] * 500, 600 - traj.xy[:, 1] * 500])  # image rows point down
    regime_msgs = []
    for f in range(len(px)):
        session.local[f] = (float(px[f, 0]), float(px[f, 1]))
        if f % 15 == 0 and f > 0:
            session.refit()
            regime_msgs = [m for m in sent if m["type"] == "regime_change"]
    assert len(regime_msgs) == 1 and abs(regime_msgs[0]["at_s"] - 7.0) < 1.0
    last_fit = [m for m in sent if m["type"] == "fit"][-1]
    assert last_fit["window_s"] < 7.0  # fitted only on motion after the change
    assert last_fit["parameters"][0]["value"] == pytest.approx(G, rel=0.05)
