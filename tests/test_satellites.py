"""Satellites: the IOD report format (checked against the specification's own examples), trajectory geometry,
SGP4 against a NASA ephemeris excerpt, impulses, and tumble periods with the half-period alias."""
from pathlib import Path

import numpy as np
import pytest
from physicslens.astro import satellites as T

FIX = Path(__file__).parent / "fixtures"
ISS = {"OBJECT_NAME": "ISS (ZARYA)", "OBJECT_ID": "1998-067A", "EPOCH": "2026-11-01T05:00:23.160096", "MEAN_MOTION": 15.49138562,
       "ECCENTRICITY": 0.0007056, "INCLINATION": 51.6332, "RA_OF_ASC_NODE": 342.0408, "ARG_OF_PERICENTER": 334.5041,
       "MEAN_ANOMALY": 25.5599, "EPHEMERIS_TYPE": 0, "CLASSIFICATION_TYPE": "U", "NORAD_CAT_ID": 25544, "ELEMENT_SET_NO": 999,
       "REV_AT_EPOCH": 59299, "BSTAR": 0.00013267359, "MEAN_MOTION_DOT": 6.903e-05, "MEAN_MOTION_DDOT": 0}


@pytest.mark.parametrize(("value", "code"), [(0.001, "15"), (0.05, "56"), (0.1, "17"), (0.9, "97"), (1, "18"), (2, "28"),
                                             (5, "58"), (10, "19"), (90, "99"), (0.0003, "34")])
def test_iod_uncertainty_codes_match_the_specification(value, code):
    assert T._mx(value) == code


def test_iod_line_columns():
    line = T.iod_line(25544, "1998-067A", T.Site(51.5, 0.0, 0, "2007"), 2461346.25, 15 * 23.712, -13.7458, 0.1, 0.5)
    assert line[0:15] == "25544 98 067A  " and line[16:20] == "2007" and line[21] == "G"
    assert line[23:40] == "20261101180000000" and line[41:43] == "17" and line[44:46] == "25"
    assert line[47:54] == "2342720" and line[54] == "-" and line[55:61] == "134475" and line[62:64] == "57"


def test_iod_rounding_carries():
    line = T.iod_line(1, "2000-001A", T.Site(0, 0), 2461346.25, 15 * (1 + 59.99999 / 60), 10.999999, 1, 1)
    assert line[47:54] == "0200000" and line[55:61] == "110000"


def test_oem_excerpt_and_sgp4_agree_within_kilometres():
    oem = T.read_oem((FIX / "iss_oem_head.txt").read_text())
    tt = oem.t_jd[5:30]
    d = T.ric(oem, T.propagate_gp(ISS, oem.t_jd))
    assert np.max(np.abs(d["intrack"])) < 30 and np.max(np.abs(d["crosstrack"])) < 5
    mid = T.interpolate(oem, (tt[:-1] + tt[1:]) / 2)
    assert np.all(np.abs(np.linalg.norm(mid.r, axis=1) - 6780) < 100)


def test_ric_directions():
    ref = T.Trajectory(np.array([0.0]), np.array([[7000.0, 0, 0]]), np.array([[0, 7.5, 0.0]]))
    other = T.Trajectory(np.array([0.0]), np.array([[7001.0, 2.0, 3.0]]), np.array([[0, 7.5, 0.0]]))
    d = T.ric(ref, other)
    assert (d["radial"][0], d["intrack"][0], d["crosstrack"][0]) == pytest.approx((1.0, 2.0, 3.0))
    assert d["intrack_s"][0] == pytest.approx(2 / 7.5)


def test_a_prograde_impulse_makes_it_fall_behind():
    s0 = np.array([6778.0, 0, 0, 0, 7.6686, 0])  # an equatorial circular orbit, so the phase is atan2(y, x)
    t = np.array([0.5, 1.0])
    a = T.integrate(s0, 0.0, t, 0.0, (0.25, np.array([0, 1e-3, 0])))
    b = T.integrate(s0, 0.0, t, 0.0, None)
    lag = (np.arctan2(a[:, 1], a[:, 0]) - np.arctan2(b[:, 1], b[:, 0]) + np.pi) % (2 * np.pi) - np.pi
    # A higher orbit has a longer period: after the push the object drifts behind, more as time goes on.
    assert lag[1] < lag[0] < 0 and np.linalg.norm(a[1]) != pytest.approx(np.linalg.norm(b[1]), abs=0.1)


def test_tumble_period_and_its_alias():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 300, 1500))
    # Two flashes per turn (the strongest peak is half the period) that differ slightly: the turn is 37.2 s.
    mag = 5 + 0.8 * np.abs(np.cos(2 * np.pi * t / 37.2)) ** 4 + 0.15 * np.cos(2 * np.pi * t / 37.2) + rng.normal(0, 0.05, len(t))
    r = T.tumble_period(t, mag)
    assert r["period_s"] == pytest.approx(37.2, rel=0.02) and r["doubled"]


def test_identical_orbits_are_one_answer():
    site = T.Site(51.5, 0.0)
    grid = np.arange(2461346.0, 2461347.0, 20 / 86400)
    _, _, el = T.radec(T.propagate_gp(ISS, grid).r, T.site_gcrs(site, grid))
    obs_t = grid[T.passes(grid, el)[0]][:10]
    tr = T.propagate_gp(ISS, obs_t)
    ra, dec, _ = T.radec(tr.r, T.site_gcrs(site, obs_t))
    twin = {**ISS, "OBJECT_NAME": "ISS (NAUKA)", "NORAD_CAT_ID": 49044, "OBJECT_ID": "2021-066A"}
    got = T.identify(obs_t, ra, dec, site, [ISS, twin], max_clock_s=1.0)
    assert got[0]["name"] == "ISS (ZARYA)" and got[0]["rms_arcsec"] < 1 and "ISS (NAUKA)" in got[0].get("same_orbit", [])
