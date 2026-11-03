"""Asteroid follow-up: the MPC observatory list, ADES PSV as its description specifies, and an orbit fit that
recovers a known orbit (and gets more certain with another night)."""
from pathlib import Path

import numpy as np
import pytest
from physicslens.astro import asteroids as A

FIX = Path(__file__).parent / "fixtures"


def test_obscodes():
    codes = A.parse_obscodes((FIX / "mpc_obscodes_excerpt.txt").read_text())
    assert codes["568"] == (204.5278, 0.94171, 0.33725) and codes["W68"][0] == pytest.approx(289.23502)
    assert "C51" not in codes  # a spacecraft code has no fixed site


def test_ades_psv_follows_the_description():
    rows = [{"provid": "2024 YR4", "trksub": "abc", "mode": "CCD", "obstime": "2024-12-27T05:42:49.52Z", "ra": "134.1707",
             "dec": "-0.26998", "rmsra": "0.2", "rmsdec": "0.16", "astcat": "Gaia3", "mag": "16.5", "band": "G"}]
    psv = A.ades_psv(None, rows, mpc_code="Z99", submitter="A. Observer")
    lines = psv.splitlines()
    assert lines[0] == "# version=2022" and lines[1] == "# observatory" and lines[2] == "! mpcCode Z99"
    keyword = next(ln for ln in lines if not ln.startswith(("#", "!")))
    tokens = [x.strip() for x in keyword.split("|")]
    # "A Keyword Record is identified as any record ... for which all pipe-separated tokens start with a lower-case
    # letter", and the identification group comes first.
    assert all(t[0].islower() for t in tokens) and tokens[:2] == ["provID", "trkSub"]
    data = [x.strip() for x in lines[-1].split("|")]
    assert len(data) == len(tokens) and data[3] == "Z99" and "|" not in "".join(data)


def _synthetic(state, epoch, t_utc, bodies, seed=0, sigma=0.3):
    # Two real observatories, as follow-up usually is: their parallax across the Earth helps fix the distance.
    codes = A.parse_obscodes((FIX / "mpc_obscodes_excerpt.txt").read_text())
    stn = np.array(["568" if i % 2 else "W68" for i in range(len(t_utc))])
    obs = A.Observations(t_utc, np.zeros(len(t_utc)), np.zeros(len(t_utc)), np.full(len(t_utc), sigma), np.full(len(t_utc), sigma),
                         stn, A.observer_positions(stn, t_utc, codes))
    ra, dec = A.predict(state, epoch, obs, bodies)
    rng = np.random.default_rng(seed)
    obs.ra = ra + rng.normal(0, sigma / 3600, len(ra)) / np.cos(np.radians(dec))
    obs.dec = dec + rng.normal(0, sigma / 3600, len(dec))
    return obs


def test_orbit_fit_recovers_a_known_orbit_and_more_nights_help():
    epoch = 2460700.5
    bodies = A.planets(epoch - 2, epoch + 40)
    earth0 = A.observer_positions(np.array(["500"]), np.array([epoch]), {"500": (0.0, 0.0, 0.0)})[0]
    # A near-Earth object 0.06 AU away, moving 8 km/s relative to the Earth (AU, AU/day).
    truth = np.concatenate([earth0 + [0.05, 0.03, 0.01], [-0.012, -0.012, 0.0005]])
    nights = [epoch + d + np.array([0.0, 0.05, 0.4, 0.45]) for d in (0.0, 1.0, 2.0, 3.0)]  # each site's night
    obs = _synthetic(truth, epoch, np.concatenate(nights), bodies)
    two = obs.subset(np.arange(len(obs.ra)) < 8)
    start = truth + np.array([1e-4, -1e-4, 5e-5, 2e-6, -2e-6, 1e-6])
    a = A.fit(two, start, epoch, bodies)
    b = A.fit(obs, start, epoch, bodies)
    earth = A.observer_positions(np.array(["500"]), np.array([epoch + 30]), {"500": (0.0, 0.0, 0.0)})[0]
    ua, ub = A.position_uncertainty(a, epoch + 30, earth, bodies), A.position_uncertainty(b, epoch + 30, earth, bodies)
    assert ub["major_arcsec"] < ua["major_arcsec"]
    # The start was 23,000 km off; the observations bring it in, to within three of the fit's own sigmas.
    err = b.state[:3] - truth[:3]
    chi2 = float(err @ np.linalg.inv(b.cov[:3, :3]) @ err)
    assert np.linalg.norm(err) < 0.5 * np.linalg.norm(start[:3] - truth[:3]) and chi2 < 3 ** 2 * 3
    assert b.rms_arcsec == pytest.approx(0.3, rel=0.4)


def test_jpl_offsets_wrap_ra_and_scale_by_cos_dec():
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("asteroid_jpl_check", root / "scripts" / "asteroid_jpl_check.py")
    import sys
    sys.path.insert(0, str(root / "scripts"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    # 1 arcsec east across RA = 0 at dec 60 is 2 arcsec of RA, and the sign survives the wrap.
    a, b = m.sky_offset([2 / 3600], [60.0], [0.0], [60.0])
    assert a[0] == pytest.approx(1.0) and b[0] == 0
    a, _ = m.sky_offset([359.9999], [0.0], [0.0001], [0.0])
    assert a[0] == pytest.approx(-0.72, abs=1e-6)
