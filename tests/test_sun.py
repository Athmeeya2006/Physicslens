"""The Sun's rotation: laws recovered from synthetic spots, the published profiles, Earth's motion, forecasts, and
NOAA's region summaries (a real one, saved as a fixture)."""
import math
from pathlib import Path

import numpy as np
import pytest
from physicslens.astro import sun as S

FIX = Path(__file__).parent / "fixtures"


def _tracks(A=14.4, B=-2.8, n=40, noise=0.05, seed=0, span=10.0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        lat = rng.uniform(-30, 30)
        t = np.sort(rng.uniform(0, span, 30))
        rate = A + B * math.sin(math.radians(lat)) ** 2 - S.CARRINGTON_DEG_PER_DAY
        lon = rng.uniform(0, 360) + rate * t + rng.normal(0, noise, len(t))
        out.append(S.Track([S.Spot(float(tt), 0, 0, 10.0, lat + rng.normal(0, 0.05), float(L % 360), 0.0, 0.9)
                            for tt, L in zip(t, lon, strict=True)]))
    return out


def test_differential_rotation_is_found_and_rigid_rejected():
    r = S.assess([_tracks(seed=1), _tracks(seed=2)])
    assert r["status"] in ("accepted", "ambiguous")
    w = r.get("winner") or r["contenders"][0]
    assert w.law.id in ("sin2", "sin2sin4", "abs")
    rigid = r["fits"]["rigid"]
    assert rigid.heldout_chi2 > S.HELDOUT_CHI2 or rigid.bic - w.bic > S.DECISIVE_DELTA_BIC
    sin2 = r["fits"]["sin2"]
    assert sin2.coef[0] == pytest.approx(14.4, abs=0.05) and sin2.coef[1] == pytest.approx(-2.8, abs=0.3)


def test_a_rigid_sun_comes_out_rigid():
    r = S.assess([_tracks(B=0.0, seed=3), _tracks(B=0.0, seed=4)])
    assert (r.get("winner") or r["contenders"][0]).law.id == "rigid" or r["status"] == "ambiguous"


def test_track_rates_add_the_carrington_rate():
    tr = _tracks(n=1, noise=0.0)[0]
    rate = S.track_rates(tr)
    assert rate["sidereal"] == pytest.approx(14.4 - 2.8 * math.sin(math.radians(tr.lat)) ** 2, abs=0.01)


def test_published_profiles_at_the_equator():
    assert S.published_omega("howard", 0) == pytest.approx(2.894e-6 * 86400 * 180 / math.pi)
    assert S.published_omega("allen", 0) == pytest.approx(14.44)


def test_earth_rate_changes_over_the_year():
    jan, jul = S.earth_rate_deg_per_day("2024-01-03T00:00:00"), S.earth_rate_deg_per_day("2024-07-05T00:00:00")
    assert 0.94 < jul < jan < 1.03  # faster near perihelion (early January)


def test_return_forecast_arithmetic():
    law = S.LawFit(S.LAWS["rigid"], np.array([14.18]), np.zeros((1, 1)), 0.0, 1, 1)
    f = S.forecast_region(0.0, -10.0, "2024-05-01T00:00:00", law, 0.98, 0.0)
    syn = 14.18 - 0.98
    assert f["central_meridian"]["days"] == pytest.approx(10 / syn)
    assert f["west_limb"]["days"] == pytest.approx(100 / syn) and f["east_limb_return"]["days"] == pytest.approx(280 / syn)


def test_noaa_region_summary():
    regions, returns = S.parse_srs((FIX / "noaa_srs_20240508.txt").read_text(), "2024-05-08")
    r3664 = next(r for r in regions if r.number == 3664)
    assert (r3664.lat, r3664.lon_stony, r3664.lon_carr, r3664.area) == (-20.0, 9.0, 349.0, 630)
    assert any(x["number"] == 3645 and x["lat"] == -8 and x["lon_carr"] == 247 for x in returns)


def test_helioviewer_header():
    h = S.parse_header("<meta><fits><NAXIS1>4096</NAXIS1><CDELT1>0.6</CDELT1><DATE-OBS>2024-05-08T11:59:50.2</DATE-OBS>"
                       "<history>x</history></fits></meta>")
    assert h == {"naxis1": 4096, "cdelt1": 0.6, "date-obs": "2024-05-08T11:59:50.2"}
