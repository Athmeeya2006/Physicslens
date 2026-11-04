"""The compensator of scripts/robot_impact.py inverts the servo laws (a compensated plan is followed far more closely
than the plan sent as is), and the pick and place moments are found where a pick-and-place gripper command has them."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from physicslens.robot import servo as S

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("robot_impact", ROOT / "scripts" / "robot_impact.py")
I = importlib.util.module_from_spec(spec)
spec.loader.exec_module(I)
DT = 1 / 30


@pytest.mark.parametrize(("law", "p"), [
    ("lag1_deadband", {"tau": 0.08, "d": 0.05, "b": 1.5, "db": 1.0}),
    ("lag1_deadband_sag", {"tau": 0.07, "d": 0.055, "b": -2.0, "db": 0.5, "sc": 1.0, "ss": -0.5}),
    ("so2_deadband", {"wn": 15.0, "zeta": 1.2, "d": 0.02, "b": 0.5, "db": 0.6}),
])
def test_compensated_plan_is_followed_closely(law, p):
    t = np.arange(300) * DT
    plan = 25 * np.sin(2 * np.pi * 0.3 * t) + 8 * np.sin(2 * np.pi * 0.7 * t + 1)

    def track(cmd):
        sim = S.simulate(law, p, S.Batch.from_lists([cmd], [plan], DT))[0]
        return float(np.sqrt(np.mean((sim - plan)[15:] ** 2)))  # after the start-up transient

    naive = track(plan - p["b"])
    compensated = track(I.compensate(law, p, plan, DT))
    assert compensated < 0.25 * naive


def test_pick_and_place_moments():
    u = np.r_[np.zeros(60), np.linspace(0, 28, 30), np.full(30, 28), np.linspace(28, 9, 10), np.full(90, 9),
              np.linspace(9, 20, 10), np.full(10, 20), np.zeros(60)]
    pick, place = I.moments(u)
    assert 120 <= pick <= 130  # closing past half of the opening (14)
    assert 225 <= place <= 245  # widest opening after the pick (reached at frame 229)
    assert I.moments(np.zeros(200)) == (None, None)
