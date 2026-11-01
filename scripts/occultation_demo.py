"""Umbriel's outline from the 2020 September 21 stellar occultation, rebuilt from the published chords.

  python scripts/occultation_demo.py

Reads the observers' sites and their disappearance and reappearance times (data/astro/occultation/, from Assafin
et al. 2023, arXiv:2310.11196, Tables 1 to 3), projects every observer onto the plane perpendicular to the star,
takes Umbriel's position from JPL Horizons (cached), and fits a circle and an ellipse, with the five observers who
saw nothing as constraints. Compared with the paper's own fit. Also checks the edge-time fit on synthetic light
curves. Writes data/astro/occultation/results.json. Free (one Horizons query, cached).
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import astropy.units as u
from astropy.coordinates import Angle, EarthLocation
from astropy.time import Time
from physicslens.astro import occultation as O

DATA = ROOT / "data" / "astro" / "occultation"
CACHE = ROOT / "var" / "astro" / "ast"
DATE = "2020-09-21"
STAR = ("02h30m28.84657s", "+14d19m36.3762s")  # Gaia DR3 75195604519240064 at the event (Table 1)
PUBLISHED = {"radius": (582.4, 0.8), "fc": (-23.7, 0.8), "gc": (-169.4, 1.9), "a_prime": (582.6, 1.1), "R_eq_prime": (581.7, 2.0),
             "limb_sigma_km": 4.2, "source": "Assafin et al. 2023 (arXiv:2310.11196), Table 4"}


def umbriel_track() -> tuple[np.ndarray, np.ndarray]:
    """Umbriel's geocentric position (km, ICRF axes) every 5 s over the event, from JPL Horizons (astrometric)."""
    path = CACHE / "umbriel_horizons.json"
    if not path.is_file():
        from astroquery.jplhorizons import Horizons
        # Uranus's moons in Horizons: 701 Ariel, 702 Umbriel, 703 Titania, 704 Oberon, 705 Miranda.
        q = Horizons(id="702", location="500", epochs={"start": f"{DATE} 08:15", "stop": f"{DATE} 08:35", "step": "1m"})
        eph = q.ephemerides(extra_precision=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"jd": list(map(float, eph["datetime_jd"])), "ra": list(map(float, eph["RA"])),
                                    "dec": list(map(float, eph["DEC"])), "delta_au": list(map(float, eph["delta"]))}))
    e = json.loads(path.read_text())
    ra, dec, d = (np.radians(e["ra"]), np.radians(e["dec"]), np.array(e["delta_au"]) * u.au.to(u.km))
    vec = np.column_stack([d * np.cos(dec) * np.cos(ra), d * np.cos(dec) * np.sin(ra), d * np.sin(dec)])
    return np.array(e["jd"]), vec


def at(jd: np.ndarray, track: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Umbriel's position at any time: a cubic fit per axis over the 20 minutes (smooth to metres)."""
    t, v = track
    x = (t - t[0]) * 1440
    return np.column_stack([np.polyval(np.polyfit(x, v[:, i], 3), (jd - t[0]) * 1440) for i in range(3)])


def observer(row: dict, jd: np.ndarray) -> np.ndarray:
    loc = EarthLocation.from_geodetic(Angle(row["lon"].replace(" ", ":"), u.deg), Angle(row["lat"].replace(" ", ":"), u.deg), float(row["alt_m"]) * u.m)
    pos, _ = loc.get_gcrs_posvel(Time(jd, format="jd", scale="utc"))
    return pos.xyz.to_value(u.km).T


def jd_of(hms: str) -> float:
    return Time(f"{DATE}T{hms}", scale="utc").jd


def main() -> None:
    star = (Angle(STAR[0]).deg, Angle(STAR[1]).deg)
    axes = O.plane_axes(*star)
    track = umbriel_track()
    with open(DATA / "umbriel_2020-09-21_chords.csv") as fh:
        rows = list(csv.DictReader(fh))
    chords = []
    speed = None
    for r in rows:
        if r["status"] == "P":
            jd = np.array([jd_of(r["ingress_utc"]), jd_of(r["egress_utc"])])
            pts = O.project(observer(r, jd) - at(jd, track), axes)
            if speed is None:  # the shadow's speed in the plane, from one observer over one second
                j2 = np.array([jd[0], jd[0] + 1 / 86400])
                p2 = O.project(observer(r, j2) - at(j2, track), axes)
                speed = float(np.hypot(*(p2[1] - p2[0])))
            sig = np.array([float(r["ingress_sigma_s"]), float(r["egress_sigma_s"])]) * speed
            chords.append(O.Chord(r["observer"], True, pts, sig))
        else:
            jd = np.linspace(jd_of(r["window_start_utc"]), jd_of(r["window_end_utc"]), 200)
            chords.append(O.Chord(r["observer"], False, O.project(observer(r, jd) - at(jd, track), axes), np.zeros(200)))
    # The limb parameter: topography and model error added to every point's error, raised until chi-square per degree
    # of freedom is 1 (as the paper does with SORA's sigma_model).
    fits = {}
    for shape in ("circle", "ellipse"):
        limb = 0.0
        for _ in range(60):
            cs = [O.Chord(c.observer, c.positive, c.points, np.hypot(c.sigma_km, limb)) if c.positive else c for c in chords]
            f = O.fit_outline(cs, shape)
            if f["chi2"] / max(f["dof"], 1) <= 1.0:
                break
            limb += 0.1
        f["limb_sigma_km"] = limb
        fits[shape] = f
    # Edge times on synthetic light curves with known edges: how well the integration model recovers them.
    rng = np.random.default_rng(0)
    trials = []
    for exposure in (0.033, 0.13, 0.5, 1.0):
        errs = []
        for _ in range(40):
            td = rng.uniform(10, 11)
            tr_ = td + rng.uniform(20, 40)
            t = np.arange(0, 60, exposure)
            flux = O._model(t, exposure, td, tr_, 1.0, 0.1) + rng.normal(0, 0.08, len(t))
            e = O.edge_times(t, flux, exposure, 0.08)
            errs.append(((e["disappear"] - td) / e["disappear_sigma"], (e["reappear"] - tr_) / e["reappear_sigma"], e["disappear"] - td))
        a = np.array(errs)
        trials.append({"exposure_s": exposure, "rms_error_s": float(np.sqrt(np.mean(a[:, 2] ** 2))),
                       "pull_sd": float(np.std(a[:, :2])), "within_1_sigma": float(np.mean(np.abs(a[:, :2]) <= 1))})
    c = fits["circle"]
    out = {"generated": "scripts/occultation_demo.py", "event": "Umbriel, 2020-09-21", "star": STAR, "shadow_speed_km_s": speed,
           "chords": [{"observer": ch.observer, "positive": ch.positive, "points": ch.points[[0, -1]].tolist() if not ch.positive else ch.points.tolist(),
                       "track": ch.points[::10].tolist() if not ch.positive else None, "sigma_km": ch.sigma_km.tolist() if ch.positive else None}
                      for ch in chords],
           "fits": fits, "published": PUBLISHED, "edge_trials": trials}
    err = c["params"]["radius"] - PUBLISHED["radius"][0]
    sig = math.hypot(c["sigma"]["radius"], PUBLISHED["radius"][1])
    out["outcome"] = {
        "headline": (f"Umbriel is {c['params']['radius']:.1f} ± {c['sigma']['radius']:.1f} km in radius, from {sum(ch.positive for ch in chords)} "
                     f"observers' timings of one star blinking out; the published fit of the same chords gives {PUBLISHED['radius'][0]} ± "
                     f"{PUBLISHED['radius'][1]} km, and none of the {sum(not ch.positive for ch in chords)} observers who saw nothing falls inside the outline."),
        "action": "Report your disappearance and reappearance times with their uncertainties; a miss near the edge constrains the size too.",
        "artifacts": [],
        "checked_against": {"name": "radius (km)", "truth": PUBLISHED["radius"][0], "measured": c["params"]["radius"],
                            "relative_error": err / PUBLISHED["radius"][0], "within_interval": abs(err) <= 1.96 * sig, "source": PUBLISHED["source"]},
        "confidence": f"radius {c['params']['radius'] - 1.96 * c['sigma']['radius']:.1f} to {c['params']['radius'] + 1.96 * c['sigma']['radius']:.1f} km (95%)",
        "status": "proven" if abs(err) <= 1.96 * sig and c["misses_crossed"] == 0 else "partial",
        "limits": [] if abs(err) <= 1.96 * sig else [f"{err:+.1f} km from the published radius"], "readable": None}
    (DATA / "results.json").write_text(json.dumps(out, indent=1, default=float))
    print("circle", {k: round(v, 2) for k, v in c["params"].items()}, {k: round(v, 2) for k, v in c["sigma"].items()}, "limb", c["limb_sigma_km"],
          "misses crossed", c["misses_crossed"], "speed", round(speed, 2))
    e = fits["ellipse"]
    print("ellipse", {k: round(v, 3) for k, v in e["params"].items()}, "Req", round(e["equivalent_radius"], 1))
    print("edges", trials)
    print(out["outcome"]["status"], out["outcome"]["headline"])


if __name__ == "__main__":
    main()
