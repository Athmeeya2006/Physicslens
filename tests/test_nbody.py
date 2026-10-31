"""The heliocentric N-body integrator against cases with known answers."""
import numpy as np
import pytest
from physicslens.physics.nbody import Body, acceleration, gm_au3_d2, propagate

GM_SUN = 2.9591220828559115e-4  # AU^3/day^2


def test_circular_orbit_returns_after_one_period():
    r = 1.0
    v = np.sqrt(GM_SUN / r)
    period = 2 * np.pi * r / v
    at = propagate([r, 0, 0, 0, v, 0], 0.0, period, GM_SUN, [])
    np.testing.assert_allclose(at(period)[:3], [r, 0, 0], atol=1e-9)


def test_energy_is_kept_on_an_eccentric_orbit():
    s0 = np.array([0.6, 0, 0, 0, 0.03, 0.002])
    at = propagate(s0, 0.0, 3000.0, GM_SUN, [])

    def energy(s):
        return 0.5 * np.dot(s[3:], s[3:]) - GM_SUN / np.linalg.norm(s[:3])
    assert energy(at(3000.0)) == pytest.approx(energy(s0), rel=1e-9)


def test_a_planet_pulls_toward_itself_minus_its_pull_on_the_sun():
    planet = Body("p", 1e-6, lambda _jd: np.array([2.0, 0.0, 0.0]))
    r = np.array([1.0, 0.0, 0.0])
    a = acceleration(r, 0.0, 0.0, [planet])
    # direct 1e-6/1^2 toward the planet (+x), minus the Sun's own acceleration 1e-6/2^2 toward it
    np.testing.assert_allclose(a, [1e-6 - 1e-6 / 4, 0, 0])


def test_units():
    assert gm_au3_d2(132712440041.27942) == pytest.approx(GM_SUN, rel=1e-9)
