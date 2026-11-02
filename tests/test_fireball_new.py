"""The new-fireball pipeline: it must reproduce Winchcombe's blind result, its geometry must give Winchcombe's published
orbit from the begin and end points alone, and its regions must agree with its own scoring rule."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fireball_new", ROOT / "scripts" / "fireball_new.py")
fn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fn)

COMMITTED = json.loads((ROOT / "data" / "astro" / "fireball" / "results.json").read_text())["summary"]


def test_winchcombe_blind_counts_are_reproduced():
    # Same release rule, bins, seed and stone count as the committed run; one find either way allows for the
    # atmosphere (NRLMSIS when its indices are reachable, else the standard atmosphere).
    res = fn.winchcombe_check(n_per_mass=fn.N_PER_MASS)
    assert res["finds"] == COMMITTED["finds"] == 9
    assert res["release_heights_m"] == []  # BLIND_RULE: no fragmentation detected, so the luminous end only
    assert abs(res["inside_95"] - COMMITTED["blind_inside_95"]) <= 1
    assert abs(res["inside_68"] - COMMITTED["blind_inside_68"]) <= 1


def test_orbit_from_begin_and_end_points_matches_the_published_one():
    ref = fn.W.read_report_reference()
    start, end = (51.871055, -3.109318, 90622.59), (51.933744, -2.188428, 33360.05)  # the report's begin and end
    rad = fn.apparent_radiant(ref["when_utc"], start, end, ref["v_inf_ms"], inertial_speed=True)
    assert abs(rad["ra_deg"] - ref["ra_deg"]) < 0.1 and abs(rad["dec_deg"] - ref["dec_deg"]) < 0.1
    orbit = fn.F.pre_entry_orbit(ref["when_utc"], *start, rad["ra_deg"], rad["dec_deg"], rad["v_inf_ms"])
    for k, (val, sig) in fn.W.PUBLISHED_ORBIT.items():
        assert abs(orbit[k] - val) <= 2 * sig, k


def test_track_direction_of_a_northward_dive():
    geo = fn.track_direction((47.0, -119.4, 80000.0), (47.3, -119.4, 40000.0))
    assert abs(geo["azimuth_deg"] % 360) < 0.5 or abs(geo["azimuth_deg"] - 360) < 0.5
    assert abs(geo["slope_deg"] - math.degrees(math.atan2(40000, 33360))) < 1.0


def test_parsers_read_the_published_formats():
    html = ("<td>Chicken Little Start Lat/Lon</td><td>+47.412, \t -119.412</td><td>Chicken Little End Lat/Lon</td>"
            "<td>+47.850, -119.408</td><td>Chicken Little Altitude</td><td>77.9 km &rarr; 45.5 km (48.4 miles)</td>"
            "<td>Chicken Little Speed</td><td>14.6 km/s</td><p>Doppler weather radar indicates that some small meteorites</p>")
    m = fn.parse_nasa_skyfalls(html)
    assert (m["start_lat"], m["end_lon"], m["end_h_m"], m["speed_ms"], m["radar_signature"]) == (47.412, -119.408, 45500.0, 14600.0, True)
    doc = {"fields": ["date", "energy", "impact-e", "lat", "lat-dir", "lon", "lon-dir", "alt", "vel", "vx", "vy", "vz"],
           "data": [["2026-08-14 07:48:36", "3.8", "0.13", "47.7", "N", "119.4", "W", "30.0", "12.2", "5.2", "22.3", "0.2"]]}
    c = fn.parse_cneos(doc, "2026-08-14T07:48:33")
    assert c["lon"] == -119.4 and c["alt_m"] == 30000.0 and c["components_consistent"] is False
    head = "#USM00072786 2026 08 14 06 0533  395 ncdc-nws ncdc-gts  476806 -1176266"
    def line(gph, wdir, wspd):  # IGRA2 fixed columns: GPH 17-21, WDIR 41-45, WSPD 47-51 (tenths of m/s)
        return f"20 {0:>5} {-9999:>6} {gph:>5} {-9999:>5} {-9999:>5} {-9999:>5} {wdir:>5} {wspd:>5}"
    rows = [line(729, 270, 100), line(1073, 0, 50), line(24900, 180, 20), line(25200, -9999, -9999)]
    s = fn.parse_igra2("\n".join([head, *rows, "#next"]), 2026, 8, 14, 6)
    assert list(s["height_m"]) == [729.0, 1073.0, 24900.0]
    assert np.allclose(s["east"], [10.0, 0.0, 0.0], atol=1e-9) and np.allclose(s["north"], [0.0, -5.0, 2.0], atol=1e-9)
    with pytest.raises(ValueError):
        fn.parse_igra2(head, 2026, 8, 15, 6)


def test_regions_agree_with_the_scoring_rule():
    rng = np.random.default_rng(1)
    end = fn.F.EndState(47.85, -119.41, 30000.0, 3000.0, 0.0, 40.0)
    pts = []
    for m in fn.MASSES_KG:
        for e, n in rng.normal(0, [300.0, 1000.0], (200, 2)):
            lat, lon = fn.F.offset_latlon(end.lat, end.lon, e, n + 5000 * math.log10(m / 0.005))
            pts.append({"lat": lat, "lon": lon, "mass_kg": m})
    regs = fn.regions(end, {"points": pts}, grid=60)
    r = regs[3]
    assert r["hdr68"]["area_km2"] < r["hdr95"]["area_km2"]
    mid = fn.score_find(end, r, *r["median_lat_lon"])
    far = fn.score_find(end, r, r["median_lat_lon"][0] + 0.2, r["median_lat_lon"][1])
    assert mid["inside_68"] and not far["inside_95"]
    inside = [fn.score_find(end, r, a, b)["inside_95"] for a, b in r["points_lat_lon"][:60]]
    assert 0.85 <= np.mean(inside) <= 1.0
    assert fn.mass_bin(0.0172) == 0.02 and fn.mass_bin(0.34) == 0.32
