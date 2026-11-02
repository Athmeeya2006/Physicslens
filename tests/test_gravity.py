"""Local gravity: the truth a measured g is compared with."""

import pytest
from physicslens.physics.gravity import AACHEN, local_g, normal_gravity


def test_normal_gravity_matches_the_published_ellipsoid_values():
    # WGS 84 normal gravity at the equator and the poles (NIMA TR8350.2, table 3.4).
    assert normal_gravity(0.0) == pytest.approx(9.7803253359, abs=1e-9)
    assert normal_gravity(90.0) == pytest.approx(9.8321849379, abs=1e-8)


def test_height_lowers_gravity_by_the_free_air_gradient():
    assert normal_gravity(45.0, 1000.0) - normal_gravity(45.0) == pytest.approx(-3.086e-3, rel=1e-9)


def test_aachen_is_not_standard_gravity_but_close_to_it():
    g = local_g(AACHEN)
    assert 9.8105 < g < 9.8112
    assert abs(g - 9.80665) > 0.004  # standard gravity is a convention, 0.04% off here
