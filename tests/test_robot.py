"""Servo identification on synthetic joints with known parameters (no data download, no model calls)."""

from __future__ import annotations

import numpy as np
import pytest
from physicslens.robot import servo as S

DT = 1 / 30


def commands(n_eps: int, n: int, seed: int) -> list[np.ndarray]:
    """Smooth random reach-and-hold commands, like a teleoperated arm: ramps between random targets."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_eps):
        u = np.zeros(n)
        k, x = 0, 0.0
        while k < n:
            target = rng.uniform(-40, 40)
            ramp = int(rng.integers(8, 25))
            hold = int(rng.integers(5, 30))
            seg = np.concatenate([np.linspace(x, target, ramp), np.full(hold, target)])
            u[k:k + len(seg)] = seg[: n - k]
            k += len(seg)
            x = target
        out.append(u)
    return out


def synthetic(law: str, p: dict[str, float], n_eps: int, seed: int, noise: float = 0.05) -> S.Batch:
    us = commands(n_eps, 240, seed)
    clean = S.Batch.from_lists(us, [np.zeros(len(u)) for u in us], DT)
    y = S.simulate(law, p, clean)
    y = y + np.random.default_rng(seed + 1).normal(0, noise, y.shape)
    return S.Batch.from_lists(us, list(y), DT)


TRUE = {"wn": 30.0, "zeta": 0.7, "d": 0.05, "b": 0.8}


def test_recovers_second_order_parameters() -> None:
    f = S.fit("so2", synthetic("so2", TRUE, 6, 0))
    assert f.params["wn"] == pytest.approx(30.0, rel=0.05)
    assert f.params["zeta"] == pytest.approx(0.7, rel=0.08)
    assert f.params["d"] == pytest.approx(0.05, abs=0.005)
    assert f.params["b"] == pytest.approx(0.8, abs=0.05)


def test_wrong_law_loses_on_heldout_episodes() -> None:
    train, test = synthetic("so2", TRUE, 5, 2), synthetic("so2", TRUE, 6, 3)
    fits = {law: S.fit(law, train) for law in ("lag1", "so2")}
    held = {law: S.rms_per_episode(S.simulate(law, f.params, test), test) for law, f in fits.items()}
    assert held["so2"].mean() < 0.5 * held["lag1"].mean()
    winner, steps = S.select(fits, held)
    assert winner == "so2"
    assert any(s.child == "so2" and s.accepted for s in steps)


def test_extra_term_without_support_is_not_chosen() -> None:
    """A deadband the data do not contain must not be accepted just because it adds a parameter."""
    train, test = synthetic("lag1", {"tau": 0.08, "d": 0.03, "b": 0.0}, 5, 4), synthetic(
        "lag1", {"tau": 0.08, "d": 0.03, "b": 0.0}, 6, 5)
    fits = {law: S.fit(law, train) for law in ("lag1", "lag1_deadband")}
    held = {law: S.rms_per_episode(S.simulate(law, f.params, test), test) for law, f in fits.items()}
    winner, _ = S.select(fits, held)
    assert winner == "lag1"


def test_recovers_deadband() -> None:
    p = TRUE | {"db": 0.6}
    f = S.fit("so2_deadband", synthetic("so2_deadband", p, 6, 6, noise=0.02))
    assert f.params["db"] == pytest.approx(0.6, abs=0.1)
    assert f.params["d"] == pytest.approx(0.05, abs=0.01)


def test_delay_before_the_log_holds_the_first_command() -> None:
    u = np.array([[1.0, 2.0, 3.0, 4.0]])
    ud = S.delayed_input(u, d=DT, dt=DT, substeps=1)
    assert np.allclose(ud, [[1.0, 1.0, 2.0, 3.0]])


def test_interval_covers_the_mean() -> None:
    x = np.random.default_rng(0).normal(2.0, 0.5, 40)
    mu, lo, hi = S.mean_interval(x)
    assert lo < mu < hi and lo < 2.0 < hi


def test_servo_gains_match_the_law() -> None:
    from physicslens.robot import arm_sim as A
    g = A.servo_gains("so2_deadband", TRUE | {"db": 0.5}, inertia=0.1)
    assert g["kp"] == pytest.approx(0.1 * 30.0**2)
    assert g["kv"] == pytest.approx(2 * 0.7 * 30.0 * 0.1)
    assert g["frictionloss"] == pytest.approx(g["kp"] * np.radians(0.5))
    assert g["delay"] == 0.05
