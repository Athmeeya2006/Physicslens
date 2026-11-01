"""URDF and SDF exports, parsed back and checked against the fitted numbers."""
import xml.etree.ElementTree as ET

import pytest
from physicslens.export.robot import RobotExportError, gravity_of, sdf, urdf

PENDULUM = {"model_id": "pendulum_linear_damping", "params": {"g": 9.79, "gamma": 0.08}, "ctx": {"length_m": 0.6}, "y0": [0.5, 0]}
SPRING = {"model_id": "oscillator_mixed", "params": {"omega": 6.0, "s0": 0.01, "zeta": 0.04, "f": 0.06}, "ctx": {}, "y0": [0.05, 0]}
THROW = {"model_id": "linear_drag", "params": {"g": 9.8, "k": 0.12}, "ctx": {}, "y0": [0, 0, 1, 2]}
SPIN = {"model_id": "spin_dry", "params": {"c": 2.0}, "ctx": {}, "y0": [0, 10]}


def test_pendulum_urdf_and_sdf_carry_the_fitted_physics():
    u = ET.fromstring(urdf(PENDULUM))
    assert u.find("joint").get("type") == "continuous"
    assert float(u.find("joint/dynamics").get("damping")) == pytest.approx(0.08 * 0.6**2)
    assert "9.79" in urdf(PENDULUM)  # URDF has no world: the measured g is in a comment for the simulator
    s = sdf(PENDULUM)
    assert gravity_of(s) == pytest.approx(9.79)
    assert float(ET.fromstring(s).find("world/model/joint/axis/dynamics/damping").text) == pytest.approx(0.08 * 0.36)


def test_spring_sdf_has_the_spring_and_urdf_says_it_cannot():
    dyn = ET.fromstring(sdf(SPRING)).find("world/model/joint/axis/dynamics")
    assert float(dyn.find("spring_stiffness").text) == pytest.approx(36.0)
    assert float(dyn.find("damping").text) == pytest.approx(2 * 0.04 * 6.0)
    assert float(dyn.find("spring_reference").text) == pytest.approx(0.01)
    assert "no joint spring" in urdf(SPRING)


def test_projectile_exports_note_the_drag_they_cannot_hold():
    assert "linear drag k = 0.12" in urdf(THROW) and "linear drag k = 0.12" in sdf(THROW)
    assert gravity_of(sdf(THROW)) == pytest.approx(9.8)


def test_other_families_are_refused_with_a_reason():
    with pytest.raises(RobotExportError, match="pendulum, oscillator, projectile"):
        urdf(SPIN)
