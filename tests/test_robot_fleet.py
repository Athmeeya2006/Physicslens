"""The cross-arm comparison (scripts/robot_fleet.py): an arm's own servo laws track it best, a borrowed calibration
does worse, and a pure difference in command offset (leader and follower calibration) is not counted against it."""

import importlib.util
from pathlib import Path

import numpy as np
from physicslens.robot import servo as S

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("robot_fleet", ROOT / "scripts" / "robot_fleet.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
DT = 1 / 30


def arm(tau: float, d: float, b: float, seed: int) -> tuple[dict, dict, dict]:
    """Ten episodes of smooth commands on five joints, the measured angle simulated with one lag1 law per joint."""
    rng = np.random.default_rng(seed)
    law = {n: ("lag1", {"tau": tau, "d": d, "b": b}) for n in F.NAMES}
    us, ys = {}, {}
    for e in range(10):
        t = np.arange(240) * DT
        u = np.column_stack([20 * np.sin(2 * np.pi * (0.2 + 0.1 * j) * t + rng.uniform(0, 6)) for j in range(6)])
        y = np.zeros_like(u)
        for j in F.ARM:
            batch = S.Batch.from_lists([u[:, j]], [np.zeros(len(t))], DT, [e])
            y[:, j] = S.simulate("lag1", law[F.NAMES[j]][1], batch)[0]
        us[e], ys[e] = u, y
    return law, us, ys


def test_own_laws_beat_borrowed_ones_and_offsets_are_recentred():
    law_a, us_a, ys_a = arm(tau=0.05, d=0.04, b=0.0, seed=1)
    law_b, _, _ = arm(tau=0.20, d=0.0, b=0.0, seed=2)
    train, test = list(range(6)), list(range(6, 10))
    own = F.arm_error(law_a, us_a, ys_a, train, test, DT)
    borrowed = F.arm_error(law_b, us_a, ys_a, train, test, DT)
    assert own.mean() < 1e-6 and borrowed.mean() > 0.5
    # The same dynamics with a different command offset is the same arm calibrated differently: no extra error.
    shifted = {n: ("lag1", {**p, "b": 5.0}) for n, (_, p) in law_a.items()}
    assert F.arm_error(shifted, us_a, ys_a, train, test, DT).mean() < 0.01
