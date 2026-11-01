"""Real astronomical data: times with uncertainties, PSF centroids, plate solutions, moving objects, the observer,
photometry and event timing, and the astro modes of the pipeline (which never call SAM)."""
import json
import math

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from physicslens.astro import centroid, io, moving, observer, photometry, platesolve
from physicslens.costguard.mock import responses_result

# ---------- times ----------

def test_fits_time_is_mid_exposure_with_its_uncertainty(tmp_path):
    hdr = fits.Header({"DATE-OBS": "2026-10-31T21:00:00.000", "EXPTIME": 30.0, "TIMEUNC": 0.05})
    fits.PrimaryHDU(np.zeros((8, 8)), hdr).writeto(tmp_path / "f.fits")
    f = io.read_fits(tmp_path / "f.fits")
    assert f.time.isot == "2026-10-31T21:00:15.000" and f.time_sigma_s == 0.05 and f.trust == "high"
    assert abs((f.tdb.jd - f.time.jd) * 86400 - 69.184) < 0.01  # TDB - UTC: 37 leap seconds + 32.184 s (+ a ms)
    fits.PrimaryHDU(np.zeros((8, 8))).writeto(tmp_path / "none.fits")
    with pytest.raises(io.AstroIOError, match="no time"):
        io.read_fits(tmp_path / "none.fits")


def test_typed_video_start_is_low_trust_and_gps_is_high():
    t, sigma, source = io.video_times(3, 25.0, "2026-11-01T00:00:00")
    assert source == "user" and sigma == 2.0 and abs((t[2] - t[0]).sec - 0.08) < 1e-9
    _, sigma_gps, source_gps = io.video_times(3, 25.0, "2026-11-01T00:00:00", gps=True)
    assert source_gps == "gps" and sigma_gps < 0.1
    frames = [io.Frame(np.zeros((2, 2)), t[k], sigma, source) for k in range(3)]
    prov = io.provenance(frames, "a phone video", "CC BY 4.0")
    assert prov["time_trust"] == "low" and prov["time_sigma_s"] == 2.0


# ---------- centroids ----------

def test_psf_centroid_beats_the_mask_and_meets_the_target_at_snr_20():
    rows = {r["snr"]: r for r in centroid.accuracy_vs_snr(snrs=(20.0, 100.0), trials=40)}
    assert rows[20.0]["psf_rms_px"] <= 0.2
    assert rows[20.0]["psf_rms_px"] < rows[20.0]["mask_rms_px"]
    assert rows[100.0]["psf_rms_px"] < 0.05 and rows[100.0]["mask_rms_px"] > 3 * rows[100.0]["psf_rms_px"]


def test_formal_uncertainty_is_honest():
    rng = np.random.default_rng(1)
    pulls = []
    for _ in range(40):
        x, y = 15 + rng.uniform(-0.5, 0.5), 15 + rng.uniform(-0.5, 0.5)
        img = centroid.synthetic_star((31, 31), x, y, 3000, 3.0) + rng.normal(0, 5, (31, 31))
        s = centroid.gaussian_centroid(img, 15, 15, noise=5.0)
        pulls.append((s.x - x) / s.sigma_x)
    assert 0.6 < np.std(pulls) < 1.5  # stated errors match the scatter


def test_a_streak_is_measured_end_to_end():
    yy, xx = np.mgrid[0:60, 0:80]
    truth = np.array([40.3, 30.7, 30.0, 0.4, 5000, 1.4, 100])
    img = centroid._trail_model(truth, xx, yy) + np.random.default_rng(0).normal(0, 3, (60, 80))
    t = centroid.fit_trail(img, 40, 31, 25, 0.3)
    assert abs(t.x0 - 40.3) < 4 * t.sigma_x0 + 0.05 and abs(t.length - 30) < 1 and abs(t.angle - 0.4) < 0.02
    assert len(t.profile) == 30 and np.all(t.profile[3:-3] > 0)


# ---------- plate solution ----------

def _field(seed=0, n=60, shape=(400, 600)):
    rng = np.random.default_rng(seed)
    truth = platesolve.tan_wcs(150.0, 20.0, 1.5, 12.0, shape)
    xy = np.column_stack([rng.uniform(20, shape[1] - 20, n), rng.uniform(20, shape[0] - 20, n)])
    sky = truth.pixel_to_world(xy[:, 0], xy[:, 1])
    img = np.full(shape, 100.0)
    for (x, y), f in zip(xy, rng.uniform(2000, 20000, n), strict=True):
        img += centroid.synthetic_star(shape, x, y, f, 3.0, sky=0.0)
    img += rng.normal(0, 5, shape)
    return truth, sky, img


def test_refinement_recovers_a_known_wcs_from_a_rough_pointing():
    truth, sky, img = _field()
    detected = np.array(centroid.detect(img, fwhm=3.0))
    rough = platesolve.tan_wcs(150.0 + 20 / 3600, 20.0 - 15 / 3600, 1.5 * 1.01, 12.5, img.shape)  # 25", 1%, 0.5 deg off
    sol = platesolve.refine_wcs(detected, sky, rough)
    assert sol.matched >= 50 and sol.rms_arcsec < 0.2
    centre = truth.pixel_to_world(300, 200)
    assert sol.wcs.pixel_to_world(300, 200).separation(centre).arcsec < 0.1
    assert abs(sol.summary()["scale_arcsec_per_px"] - 1.5) < 0.002


def test_solve_field_is_called_safely_or_explains_how_to_install(tmp_path, monkeypatch):
    monkeypatch.setattr(platesolve.shutil, "which", lambda _n: None)
    with pytest.raises(platesolve.PlateSolveError, match="apt install astrometry.net"):
        platesolve.solve_field(tmp_path / "x.fits")
    monkeypatch.setattr(platesolve.shutil, "which", lambda _n: "/usr/bin/solve-field")
    (tmp_path / "x.fits").write_bytes(b"")
    with pytest.raises(platesolve.PlateSolveError, match="between"):
        platesolve.solve_field(tmp_path / "x.fits", ra_deg=400, dec_deg=0)


def test_gaia_queries_are_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(platesolve, "CACHE", tmp_path)
    rows = {"ra": [150.0, 150.01], "dec": [20.0, 20.01], "g": [10.0, 11.0], "query": "q", "source": "test"}
    import hashlib
    key = hashlib.sha256(f"{150.0:.5f},{20.0:.5f},{0.5:.4f},{16.0:.2f}".encode()).hexdigest()[:16]
    key_file = tmp_path / f"gaia_{key}.json"
    key_file.write_text(json.dumps(rows))
    stars = platesolve.gaia_stars(150.0, 20.0, 0.5)  # from the cache: no network
    assert len(stars) == 2 and stars[1].dec.deg == pytest.approx(20.01)


# ---------- moving objects ----------

def test_a_mover_is_linked_and_stars_hot_pixels_and_cosmic_rays_are_not():
    rng = np.random.default_rng(3)
    stars = rng.uniform(0, 500, (30, 2))
    hot = np.array([[250.0, 250.0]])
    times = np.arange(6) * 60.0
    dets = []
    for k, t in enumerate(times):
        mover = np.array([[100 + 0.5 * t, 80 + 0.2 * t]])
        d = [stars + rng.normal(0, 0.2, stars.shape), hot, mover + rng.normal(0, 0.2, (1, 2))]
        if k == 2:
            d.append(np.array([[400.0, 30.0]]))  # a cosmic ray, one frame only
        dets.append(np.vstack(d))
    tracks = moving.find_moving(dets, times)
    assert len(tracks) == 1
    tr = tracks[0]
    assert len(tr.frames) == 6 and tr.rate_px_s == pytest.approx((0.5, 0.2), abs=0.01) and tr.rms_px < 0.5


# ---------- the observer ----------

def test_light_time_and_aberration_are_applied():
    site = observer.Site(51.48, 0.0, 50.0, "Greenwich")
    t = Time("2026-03-20T12:00:00", scale="utc")
    sun = observer.get_body_barycentric("sun", t).xyz.to("au").value
    fixed = sun + np.array([2.0, 0.5, 0.3])  # a body that does not move: light time is distance / c
    app, light_s = observer.apparent_from_ephemeris(lambda _tt: fixed, t, site)
    dist = np.linalg.norm(fixed - observer.observer_barycentric_au(t, site))
    assert light_s == pytest.approx(dist / observer.C_AU_PER_DAY * 86400, rel=1e-9)
    geometric = SkyCoord(*(fixed - observer.observer_barycentric_au(t, site)), unit="au", representation_type="cartesian")
    shift = SkyCoord(app.ra, app.dec).separation(SkyCoord(geometric.spherical.lon, geometric.spherical.lat)).arcsec
    assert 0.5 < shift < 21.0  # aberration from the observer's velocity, at most about 20.5 arcsec
    polaris = SkyCoord(37.95456067, 89.26410897, unit="deg")
    assert abs(observer.altaz(polaris, t, site).alt.deg - 51.48) < 1.0  # the pole star sits at the site's latitude


# ---------- photometry and timing ----------

def test_a_transit_mid_time_is_recovered_and_the_clock_counts():
    rng = np.random.default_rng(5)
    t = np.linspace(0, 0.3, 400)  # days
    true_mid = 0.1523
    flux = photometry.trapezoid(t, true_mid, 0.012, 0.09, 0.012) + rng.normal(0, 0.0015, len(t))
    ev = photometry.event_timing(t, flux, np.full(len(t), 0.0015))
    assert abs(ev.mid_time - true_mid) < 4 * ev.sigma_fit and ev.depth == pytest.approx(0.012, abs=0.002)
    slow = photometry.event_timing(t, flux, np.full(len(t), 0.0015), clock_sigma=2.0 / 86400)
    assert slow.sigma == pytest.approx(math.hypot(slow.sigma_fit, 2.0 / 86400))


def test_aperture_photometry_and_comparison_stars():
    rng = np.random.default_rng(2)
    imgs, true = [], []
    for k in range(5):
        cloud = 1 - 0.1 * k  # thinning sky affects both stars alike
        img = centroid.synthetic_star((60, 60), 20, 20, 10000 * cloud, 3.0) + centroid.synthetic_star((60, 60), 40, 40, 20000 * cloud, 3.0, sky=0)
        imgs.append(img + rng.normal(0, 3, (60, 60)))
        true.append(cloud)
    flux, err = photometry.aperture_series(imgs, [(20, 20), (40, 40)], radius=6)
    assert flux[0, 0] == pytest.approx(10000, rel=0.03) and np.all(err > 0)
    rel, _ = photometry.relative(flux, err, 0, [1])
    assert np.ptp(rel) < 0.02  # the cloud is divided out


# ---------- the astro modes ----------

def test_astro_modes_never_call_sam_and_carry_provenance_and_an_outcome(harness):
    from physicslens.astro.analysis import analyze_sky_track
    harness.sdk("meta_api").script.extend([responses_result(text=json.dumps(
        {"family": "plane", "candidates": [{"model_id": "plane_constant_velocity", "rationale": "Steady drift."}],
         "free_models": []}))])
    from astropy.time import TimeDelta
    t = Time("2026-11-01T00:00:00", scale="utc") + TimeDelta(np.arange(40) * 0.01, format="jd")  # 0.01 day apart
    ra = 150 + 0.002 * np.arange(40) + np.random.default_rng(0).normal(0, 2e-6, 40)
    dec = 20 + 0.001 * np.arange(40) + np.random.default_rng(1).normal(0, 2e-6, 40)
    prov = {"data_source": "synthetic frames", "licence": "test", "time_source": "fits", "time_sigma_s": 0.1,
            "wcs_rms_arcsec": 0.12}
    r = analyze_sky_track(harness.client, t, ra, dec, run_id="astro-test", name="test asteroid", provenance=prov)
    assert r["mode"] == "astro_frames" and r["provenance"]["wcs_rms_arcsec"] == 0.12
    # A straight line in RA and Dec bends slightly on the tangent plane: at 7 mas of noise either law may win.
    assert r["status"] == "accepted" and r["equation"]["model_id"] in ("plane_constant_velocity", "plane_constant_accel")
    assert r["outcome"]["status"] == "partial" and r["outcome"]["headline"]
    calls = [c for sdk in harness.sdks.values() for c in sdk.calls]
    assert calls and not any(c.get("model") == "sam-3.1" or c["kind"] in ("stream", "files") for c in calls)


def test_video_analysis_refuses_astro_modes(tmp_path):
    from physicslens.pipeline import AnalysisError, RunOptions, analyze
    with pytest.raises(AnalysisError, match="astro.analysis"):
        analyze(None, tmp_path / "x.mp4", tmp_path, "r", RunOptions(mode="series"))
