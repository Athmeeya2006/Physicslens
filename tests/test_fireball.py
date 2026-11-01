"""Fireballs: the alpha-beta law and its triage, the drag curve, dark flight against the published Winchcombe
stones, the orbit before entry against the published one, and the radiosonde parser."""
import csv
import math
from pathlib import Path

import numpy as np
import pytest
from physicslens.astro import fireball as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "astro" / "fireball"


def test_alpha_beta_recovers_a_synthetic_flight():
    v = np.linspace(0.98, 0.35, 60)
    h = F.alpha_beta_height(40.0, 1.5, v)
    rng = np.random.default_rng(0)
    ab = F.fit_alpha_beta(h, v * 15000 + rng.normal(0, 30, len(v)), 45.0, v0=15000, n_boot=50)
    assert ab.alpha == pytest.approx(40.0, rel=0.1) and ab.beta == pytest.approx(1.5, rel=0.1)
    assert ab.heldout_rms_kms < 0.2


def test_speed_and_height_are_inverse():
    v = np.array([0.9, 0.6, 0.3])
    assert F.alpha_beta_speed(30.0, 2.0, F.alpha_beta_height(30.0, 2.0, v)) == pytest.approx(v, rel=1e-6)


def test_dfn_example_reaches_the_objective_minimum():
    with open(DATA / "dfn_DN150417.csv") as fh:
        rows = list(csv.DictReader(fh))
    ab = F.fit_alpha_beta([float(r["height"]) for r in rows], [float(r["D_DT_geo"]) for r in rows], 15.17, n_boot=20)
    assert ab.alpha == pytest.approx(249.6, rel=0.01) and ab.beta == pytest.approx(1.806, rel=0.01)
    assert ab.triage()["verdict"] == "unlikely"


@pytest.mark.parametrize(("x", "beta", "verdict"), [(1.0, 0.5, "likely"), (2.0, 4.0, "possible"), (4.0, 3.0, "unlikely")])
def test_triage_against_the_two_lines(x, beta, verdict):
    ab = F.AlphaBeta(math.exp(x), beta, 15000.0, 90.0)
    assert ab.triage()["verdict"] == verdict


def test_drag_curve_is_continuous_and_shape_ordered():
    for a in (1.21, 2.18):
        knee = 0.722 if a == 1.21 else 1.15
        # The published cube fit steps by 0.024 at its join; the sphere's is continuous.
        assert F.drag_coefficient(knee - 1e-9, a) == pytest.approx(F.drag_coefficient(knee + 1e-9, a), abs=0.03)
    assert F.drag_coefficient(0.3, 2.7) > F.drag_coefficient(0.3, 1.21)


def _winchcombe_end() -> F.EndState:
    return F.EndState(51.940114, -2.096335, 27554.2, 3500.0, 84.142584, 90 - 48.704165, ground_m=300.0)


@pytest.mark.parametrize(("mass", "shape", "speed", "minutes"), [(0.32, 1.45, (44, 51), 4.5), (0.16, 1.6, (34, 41), 6.0)])
def test_dark_flight_matches_the_published_stones(mass, shape, speed, minutes):
    r = F.dark_flight(_winchcombe_end(), mass, F.standard_atmosphere(), shape_a=shape, rho_m=2090.0)
    assert speed[0] <= r["impact_speed_ms"] <= speed[1]
    assert r["time_s"] / 60 == pytest.approx(minutes, rel=0.1)


def test_main_mass_lands_near_the_driveway():
    r = F.dark_flight(_winchcombe_end(), 0.32, F.standard_atmosphere(), shape_a=1.45, rho_m=2090.0)
    assert F.distance_m(r["lat"], r["lon"], 51.951, -1.976) < 800


def test_orbit_before_entry_matches_the_published_one():
    o = F.pre_entry_orbit("2021-02-28T21:54:15.882159", 51.871055, -3.109318, 90622.59, 66.66236, 27.69771, 13764.53)
    for key, (val, sig) in {"a_au": (2.5855, 0.0077), "e": (0.6183, 0.0011), "i_deg": (0.460, 0.014),
                            "q_au": (0.986839, 0.000012), "v_g_ms": (8123.0, 13.0)}.items():
        assert abs(o[key] - val) <= 2 * sig, key


def test_slower_than_escape_is_refused():
    with pytest.raises(ValueError):
        F.pre_entry_orbit("2021-02-28T21:54:15", 51.87, -3.1, 90000.0, 66.7, 27.7, 9000.0)


def test_wyoming_sounding_in_both_formats():
    rows = ["   PRES   HGHT   TEMP   DWPT   RELH   MIXR   DRCT   {}   THTA   THTE   THTV",
            "    hPa      m      C      C      %   g/kg    deg    m/s      K      K      K ",
            "-" * 77]
    for col, factor in (("SPED", 1.0), ("SKNT", 0.514444)):
        body = [f" 1000.0    {h:3d}    5.5    0.6     71   3.98     90   10.0  278.6  289.8  279.3" for h in (110, 290, 500)]
        h, e, n = F.parse_wyoming_sounding("\n".join([rows[0].format(col), *rows[1:], *body]))
        assert list(h) == [110, 290, 500] and e[0] == pytest.approx(-10 * factor) and n[0] == pytest.approx(0, abs=1e-9)


def test_strewn_field_exports():
    fld = F.strewn_field(_winchcombe_end(), F.standard_atmosphere(), [0.1], n_per_mass=8, rho_m=2090.0)
    gj = F.to_geojson(fld, [{"lat": 51.95, "lon": -2.0, "mass_g": 100}])
    kinds = {f["properties"]["kind"] for f in gj["features"]}
    assert {"simulated", "2-sigma area", "recovered"} <= kinds
    assert F.to_kml(fld, "t").count("<Placemark>") == 2


def test_standard_atmosphere_above_86_km():
    # US Standard Atmosphere 1976 table: 6.958e-6 kg/m^3 at 86 km, 3.416e-6 at 90 km.
    rho = F.us76_density([86000.0, 90000.0])
    assert rho[0] == pytest.approx(6.958e-6, rel=0.01) and rho[1] == pytest.approx(3.416e-6, rel=0.03)


def test_strewn_field_spots_follow_where_the_stones_started():
    end = _winchcombe_end()
    fld = F.strewn_field(end, F.standard_atmosphere(), [0.1], n_per_mass=12, rho_m=2090.0,
                         release=(end.h_m, 34900.0, lambda h: 3500.0 + 0.0 * h))
    b = fld["by_mass"][0]
    en = np.array([F._east_north(end, p["lat"], p["lon"]) for p in fld["points"]])
    assert (b["east_m"], b["north_m"]) == pytest.approx(tuple(np.median(en, axis=0)), abs=1.0)
