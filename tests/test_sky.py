"""The sky engine against NASA/JPL and against physics with known answers (offline, from data/sky/)."""
import math

import numpy as np
import pytest
from physicslens.sky import validate
from physicslens.sky.elements import Elements, angle_diff, to_elements, to_state
from physicslens.sky.ephem import J2000, catalogue_system, sun_gm
from physicslens.sky.nbody import (
    AU_KM,
    Body,
    J2Field,
    Options,
    km_per_s_to_au_per_day,
    passing_star,
    simulate,
)

GM = sun_gm()


def test_planets_stay_on_jpls_track_for_ten_years():
    # Newtonian, every catalogue body: Earth's leftover is about relativity's share over ten years (next test).
    err = validate.planets_ten_years()
    assert err["Earth"] < 1e-5 and err["Jupiter"] < 1e-6 and err["Saturn"] < 1e-6


def test_with_relativity_earth_is_within_a_few_kilometres_of_jpl():
    truth = validate.validation_positions()
    run = simulate(GM, catalogue_system(validate.MASSIVE), J2000, truth["Earth"][:, 0], Options(gr=True))
    worst = max(float(np.max(np.linalg.norm(run.of(n)[0] - rows[:, 1:4], axis=1))) for n, rows in truth.items())
    assert worst < 1e-7  # AU: 15 km after ten years, for Earth, Jupiter and Saturn


def test_heliocentric_and_barycentric_frames_give_the_same_motion():
    assert validate.frames_agree() < 1e-9  # AU over ten years (measured: 2e-10, the integrator's own tolerance)


@pytest.mark.slow
def test_relativity_advances_mercurys_perihelion_by_43_arcseconds_a_century():
    assert validate.mercury_relativity(years=100) == pytest.approx(42.98, abs=0.2)


def test_exaggerated_relativity_scales_the_advance():
    # The lab's fictional c_scale multiplies 1/c^2, so the advance grows in proportion (over a short span).
    assert validate.mercury_relativity(years=10, c_scale=100) == pytest.approx(100 * 42.98, rel=0.02)


@pytest.mark.slow
def test_apophis_from_j2000_meets_earth_in_2029_where_jpl_says():
    r = validate.apophis_2029()
    assert abs(r["hours"]) < 0.1 and abs(r["km"] - r["jpl_km"]) < 2000


@pytest.mark.slow
def test_halley_from_j2000_returns_in_2061_on_jpls_date():
    with_planets = validate.halley_2061(planets=True)
    assert abs(with_planets["days"]) < 2
    sun_only = validate.halley_2061(planets=False)
    assert abs(sun_only["days"]) > 50  # the planets carry the rest; the number is in the README, not assumed here


def test_elements_round_trip():
    for el in (Elements(1.3, 0.2, 0.3, 1.0, 2.0, 0.5, GM), Elements(17.8, 0.967, 2.83, 1.03, 1.95, 3.0, GM),
               Elements(0.9, 0.0, 0.0, 0.0, 0.0, 1.0, GM)):
        back = to_elements(to_state(el), GM)
        np.testing.assert_allclose(to_state(back), to_state(el), atol=1e-12)
        assert back.a == pytest.approx(el.a, rel=1e-12)


def test_halleys_elements_from_its_jpl_state():
    halley = to_elements(catalogue_system(["Halley"])[0].state, GM)
    assert halley.e == pytest.approx(0.967, abs=0.002) and halley.period / 365.25 == pytest.approx(76, abs=2)
    assert math.degrees(halley.i) == pytest.approx(162, abs=1)


def test_j2_precesses_an_inclined_moon_orbit_and_spares_an_equatorial_one():
    gm_j, radius = 1.0, 1.0
    field = J2Field(gm_j, 0.0147, radius, (0.0, 0.0, 1.0))
    days = np.array([0.0, 2000.0])
    tilted = to_state(Elements(6.0, 0.0, math.radians(20), 0.0, 0.0, 0.0, gm_j))
    run = simulate(gm_j, [Body("moon", 0.0, tilted, massive=False)], 0.0, days, Options(j2=field, rtol=1e-11, atol=1e-13))
    pos, vel = run.of("moon")
    node = to_elements(np.concatenate([pos[-1], vel[-1]]), gm_j).node
    # Nodal regression rate -3/2 n J2 (R/a)^2 cos i, for a circular orbit.
    n = math.sqrt(gm_j / 6.0**3)
    expected = -1.5 * n * 0.0147 * (radius / 6.0) ** 2 * math.cos(math.radians(20)) * days[-1]
    assert angle_diff(node, 0.0) == pytest.approx(expected, rel=0.02)
    flat = to_state(Elements(6.0, 0.0, 0.0, 0.0, 0.0, 0.0, gm_j))
    run = simulate(gm_j, [Body("moon", 0.0, flat, massive=False)], 0.0, days, Options(j2=field))
    assert abs(run.of("moon")[0][-1][2]) < 1e-12


def test_a_passing_star_kicks_a_distant_body_and_leaves_the_sun_frame_consistent():
    star = passing_star(0.5 * GM, 200.0, km_per_s_to_au_per_day(30.0), J2000 + 3650.0)
    far = to_state(Elements(45.0, 0.0, 0.0, 0.0, 0.0, 0.0, GM))
    days = J2000 + np.array([0.0, 7300.0])
    calm = simulate(GM, [Body("kbo", 0.0, far.copy(), massive=False)], J2000, days)
    kicked = simulate(GM, [Body("kbo", 0.0, far.copy(), massive=False)], J2000, days, Options(prescribed=(star,)))
    moved = float(np.linalg.norm(kicked.of("kbo")[0][-1] - calm.of("kbo")[0][-1]))
    assert 1e-4 < moved < 10  # AU: a real kick, not a scattering
    with pytest.raises(ValueError):
        passing_star(GM, 200.0, 0.0, J2000)


def test_mass_scaling_changes_a_planets_pull():
    days = J2000 + np.array([0.0, 3650.0])
    normal = simulate(GM, catalogue_system(["Jupiter", "Earth"]), J2000, days)
    heavy = simulate(GM, catalogue_system(["Jupiter", "Earth"], mass_scale={"Jupiter": 10.0}), J2000, days)
    shift = np.linalg.norm(heavy.of("Earth")[0][-1] - normal.of("Earth")[0][-1]) * AU_KM
    assert shift > 1e5  # km after ten years with Jupiter ten times heavier


def test_energy_is_kept_and_a_loose_run_is_flagged():
    days = J2000 + np.array([0.0, 3652.5])
    run = simulate(GM, catalogue_system(["Jupiter", "Saturn"]), J2000, days)
    assert run.energy_drift < 1e-9 and not run.warnings
    loose = simulate(GM, catalogue_system(["Mercury", "Jupiter"]), J2000, days, Options(rtol=1e-5, atol=1e-7))
    assert loose.warnings and "energy" in loose.warnings[0].lower()


def test_test_particles_pull_on_nothing():
    days = J2000 + np.array([0.0, 365.25])
    alone = simulate(GM, catalogue_system(["Earth"]), J2000, days)
    crowd = [Body(f"p{i}", 0.0, catalogue_system(["Ceres"])[0].state * (1 + 0.01 * i), massive=False) for i in range(5)]
    with_particles = simulate(GM, catalogue_system(["Earth"]) + crowd, J2000, days)
    # Equal to the integrator's tolerance: the particles only change the adaptive step sizes, never a force.
    np.testing.assert_allclose(alone.of("Earth")[0], with_particles.of("Earth")[0], atol=1e-10)


@pytest.mark.slow
def test_performance_is_measured():
    # Eight planets and ten test particles for a century, daily output. The budget in the plan was 20 s on one
    # core; the measured figure goes in the README whatever it is. This guards against a large regression only.
    assert validate.performance()["seconds"] < 90


def test_rebound_is_optional():
    from physicslens.sky.nbody import rebound_available
    bodies = catalogue_system(["Jupiter", "Saturn"])
    days = J2000 + np.array([0.0, 3650.0])
    if not rebound_available():
        with pytest.raises(ValueError, match="pip install rebound"):
            simulate(GM, bodies, J2000, days, Options(backend="rebound"))
        return
    ours = simulate(GM, bodies, J2000, days)
    theirs = simulate(GM, catalogue_system(["Jupiter", "Saturn"]), J2000, days, Options(backend="rebound"))
    np.testing.assert_allclose(theirs.pos[-1], ours.pos[-1], atol=1e-6)
