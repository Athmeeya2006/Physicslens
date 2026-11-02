"""Fireballs on real data: the Winchcombe meteorite fall (2021) from bright flight to the stones on the ground.

  python scripts/fireball_demo.py [--quick]

Reads data/astro/fireball/ (the published UK Fireball Alliance trajectory solution, the radiosonde winds from the
night, the find sites from the curation paper, the Desert Fireball Network's example event) and writes
data/astro/fireball/results.json, a strewn field (GeoJSON, KML) and the comparison with every recovered stone.
Free and local: no model calls.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.astro import fireball as F

DATA = ROOT / "data" / "astro" / "fireball"
EVENT_UTC = "2021-02-28T21:54:16"

# Published values compared against (each checked in the source named; see data/astro/fireball/SOURCES.md).
PUBLISHED_ORBIT = {"a_au": (2.5855, 0.0077), "e": (0.6183, 0.0011), "i_deg": (0.460, 0.014), "node_deg": (160.1955, 0.0014),
                   "peri_deg": (351.798, 0.018), "q_au": (0.986839, 0.000012), "v_g_ms": (8123.0, 13.0)}
PUBLISHED_ENTRY_MASS_KG = (13.0, 3.0)  # King et al. 2022 via Russell et al. 2024 ("13 +- 3 kg"); the paper's model: 12.5 kg
PUBLISHED_DARK_FLIGHT = [  # McMullan et al. 2023 (arXiv:2303.12126), section on cross-matching samples with simulations
    {"stone": "main mass", "mass_kg": 0.32, "shape": (1.4, 1.5), "impact_ms": (46.0, 49.0), "minutes": 4.5},
    {"stone": "160 g (site 5)", "mass_kg": 0.16, "shape": (1.5, 1.7), "impact_ms": (35.0, 40.0), "minutes": 6.0},
    {"stone": "16.5 g (site 6)", "mass_kg": 0.0165, "shape": (2.7, 2.7), "impact_ms": (17.0, 19.0), "minutes": 16.0},
]
# The last fragmentation, a flare seen from Hullavington: 34.9 km (McMullan et al. 2023, fragmentation table).
LAST_FRAGMENTATION_M = 34_900.0
DFN_EXAMPLE = {"event": "DN150417", "slope_deg": 15.17, "notebook_alpha": 294.451599359, "notebook_beta": 1.14005615987}


def read_points() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time (s), height (m) and distance along the trajectory (m) of every measured point, all stations."""
    lines = (DATA / "winchcombe_wmpl_report_all.txt").read_text().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.strip() == "Points")
    hdr = [c.strip() for c in lines[i + 2].split(",")]
    col = {k: hdr.index(k) for k in ("Ignore", "Time (s)", "Height (m)", "State vect dist (m)")}
    t, h, d = [], [], []
    for ln in lines[i + 3:]:
        c = [x.strip() for x in ln.split(",")]
        if len(c) < len(hdr) - 2:
            break
        if c[col["Ignore"]] != "0":
            continue
        t.append(float(c[col["Time (s)"]]))
        h.append(float(c[col["Height (m)"]]))
        d.append(float(c[col["State vect dist (m)"]]))
    order = np.argsort(t)
    return np.array(t)[order], np.array(h)[order], np.array(d)[order]


def speeds(t, h, d, step: float = 0.1, half: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Speed along the track every `step` seconds: a straight line through every station's points within `half`
    seconds (point-to-point speeds from one camera are far noisier)."""
    out_h, out_v = [], []
    for c in np.arange(t.min() + half, t.max() - half, step):
        m = np.abs(t - c) <= half
        if m.sum() < 6:
            continue
        out_v.append(np.polyfit(t[m], d[m], 1)[0])
        out_h.append(float(np.mean(h[m])))
    return np.array(out_h), np.array(out_v)


def read_winds(atm: F.Atmosphere, name: str) -> F.Atmosphere:
    with open(DATA / name) as fh:
        rows = list(csv.DictReader(fh))
    return atm.with_wind([float(r["height_m"]) for r in rows], [float(r["wind_east_ms"]) for r in rows],
                         [float(r["wind_north_ms"]) for r in rows], f"radiosonde {name}")


def read_finds() -> list[dict]:
    """The recovered stones. A find marked part_of another is debris of that stone's impact (the 17.2 g piece that
    bounced 10 m into the neighbour's driveway from the main mass, Russell et al. 2024), so it is counted with it."""
    with open(DATA / "winchcombe_finds.csv") as fh:
        rows = [{**r, "lat": float(r["lat"]), "lon": float(r["lon"]), "mass_kg": float(r["mass_g"]) / 1000} for r in csv.DictReader(fh)]
    stones = [r for r in rows if not r.get("part_of")]
    for r in rows:
        if r.get("part_of"):
            host = next(x for x in stones if x["site"] == r["part_of"])
            host["mass_kg"] += r["mass_kg"]
            host["mass_g"] = f"{host['mass_kg'] * 1000:.1f}"
            host.setdefault("includes", []).append(f"{r['mass_g']} g that bounced into the {r['site']} (impact debris)")
    return stones


def end_state() -> F.EndState:
    """The dark-flight start the authors used (their DFN_darkflight configuration, from the trajectory solution)."""
    cfg = {}
    for ln in (DATA / "winchcombe_darkflight_config.cfg").read_text().splitlines():
        if "=" in ln and not ln.lstrip().startswith("#"):
            k, v = (x.strip() for x in ln.split("=", 1))
            cfg[k] = v
    lat, lon = float(cfg["lat0"]), float(cfg["lon0"])
    return F.EndState(lat, lon, float(cfg["z0"]), float(cfg["vtot0"]), float(cfg["azimuth0"]), 90 - float(cfg["zenangle"]),
                      sigma_pos_m=float(cfg["dlat"]) * 111_000, sigma_h_m=float(cfg["dz"]), sigma_speed_ms=float(cfg["dvtot"]),
                      sigma_angle_deg=max(float(cfg["dzenith"]), float(cfg["dazimuth0"])), ground_m=float(cfg["gndlvl"]))


def along_cross(end: F.EndState, lat: float, lon: float) -> tuple[float, float]:
    """A place as (along, across) the fireball's ground track from the luminous end, meters."""
    dn = math.radians(lat - end.lat) * F.EARTH_R_M
    de = math.radians(lon - end.lon) * F.EARTH_R_M * math.cos(math.radians(end.lat))
    az = math.radians(end.azimuth_deg)
    return de * math.sin(az) + dn * math.cos(az), de * math.cos(az) - dn * math.sin(az)


def dfn_example() -> dict:
    with open(DATA / "dfn_DN150417.csv") as fh:
        rows = list(csv.DictReader(fh))
    h = np.array([float(r["height"]) for r in rows])
    v = np.array([float(r["D_DT_geo"]) for r in rows])
    ab = F.fit_alpha_beta(h, v, DFN_EXAMPLE["slope_deg"])
    ok = np.isfinite(v) & (v > 1)
    vv, yy = v[ok] / np.mean(v[ok][:10]), h[ok] / F.H0_M
    q4 = lambda a, b: float(np.sum(F._q4([a, b], vv, yy) ** 2))
    ref = F.AlphaBeta(DFN_EXAMPLE["notebook_alpha"], DFN_EXAMPLE["notebook_beta"], ab.v0, DFN_EXAMPLE["slope_deg"])
    return {"event": DFN_EXAMPLE["event"], "alpha": ab.alpha, "beta": ab.beta, "alpha_sigma": ab.alpha_sigma, "beta_sigma": ab.beta_sigma,
            "objective": q4(ab.alpha, ab.beta), "notebook": {"alpha": ref.alpha, "beta": ref.beta, "objective": q4(ref.alpha, ref.beta),
                                                             "triage": ref.triage()["verdict"]},
            "triage": ab.triage(), "heldout_rms_kms": ab.heldout_rms_kms}


def main() -> None:
    if "--outcome-only" in sys.argv:  # re-judge the stored result, no simulation
        out = json.loads((DATA / "results.json").read_text())
        out["outcome"] = outcome(out)
        (DATA / "results.json").write_text(json.dumps(out, indent=1, default=float))
        print(out["outcome"]["status"], out["outcome"]["confidence"])
        return
    quick = "--quick" in sys.argv
    t0 = time.time()
    out: dict = {"generated": "scripts/fireball_demo.py", "event": "Winchcombe, 2021-02-28 21:54 UTC"}
    out["dfn_example"] = dfn_example()
    print("DN150417:", round(out["dfn_example"]["alpha"], 1), round(out["dfn_example"]["beta"], 3), out["dfn_example"]["triage"]["verdict"])

    t, h, d = read_points()
    hs, vs = speeds(t, h, d)
    end = end_state()
    slope = end.slope_deg
    ab = F.fit_alpha_beta(hs, vs, slope)
    m0, mf = ab.masses_kg(cd_a=2 * 0.8 * 1.21, rho_m=2090.0)  # the paper's drag factor 0.8 and sphere shape 1.21
    out["alpha_beta"] = {"alpha": ab.alpha, "beta": ab.beta, "alpha_sigma": ab.alpha_sigma, "beta_sigma": ab.beta_sigma,
                         "v0_ms": ab.v0, "heldout_rms_kms": ab.heldout_rms_kms, "train_rms_kms": ab.train_rms_kms, "n": ab.n,
                         "triage": ab.triage(), "entry_mass_kg": m0, "final_mass_kg": mf,
                         "curve": [[float(x), float(F.alpha_beta_height(ab.alpha, ab.beta, x / ab.v0))] for x in np.linspace(vs.min(), ab.v0 * 0.999, 80)],
                         "points": [[float(a), float(b)] for a, b in zip(hs, vs, strict=True)]}
    print("alpha-beta:", round(ab.alpha, 1), round(ab.beta, 3), ab.triage()["verdict"], "entry", round(m0, 2), "kg")

    try:
        atm = F.msis_atmosphere(EVENT_UTC, end.lat, end.lon)
    except (ImportError, OSError, ValueError) as e:  # pymsis missing or its indices unreachable: say so
        print("NRLMSIS unavailable, standard atmosphere:", e)
        atm = F.standard_atmosphere()
    sb = F.fit_single_body(hs, vs, slope, atm)
    single, frag = sb["single"], sb["fragmentation"]
    out["single_body"] = {"v0_ms": single.v0, "gamma0": single.gamma0, "sigma": single.sigma, "bic": single.bic,
                          "heldout_rms_kms": sb["train_only"].heldout_rms_kms, "entry_mass_kg": single.entry_mass_kg(0.8 * 2 * 1.21, 2090.0),
                          "fragmentation": None if frag is None else {"height_m": frag.jump_h, "drag_jump": frag.jump, "bic": frag.bic,
                                                                     "delta_bic": single.bic - frag.bic},
                          "end_mass_fraction": sb["end_mass_fraction"], "atmosphere": sb["atmosphere"]}
    print("single body:", out["single_body"]["fragmentation"], "entry", round(out["single_body"]["entry_mass_kg"], 2))

    ref = read_report_reference()
    orbit = F.pre_entry_orbit(**ref)
    out["orbit"] = {"ours": orbit, "published": PUBLISHED_ORBIT,
                    "pulls": {k: (orbit[k] - v[0]) / v[1] for k, v in PUBLISHED_ORBIT.items()}, "input": ref}
    print("orbit pulls:", {k: round(v, 2) for k, v in out["orbit"]["pulls"].items()})

    windy = read_winds(atm, "winds_nottingham_03354_2021-03-01_00Z.csv")
    checks = []
    for p in PUBLISHED_DARK_FLIGHT:
        shape = sum(p["shape"]) / 2
        r = F.dark_flight(end, p["mass_kg"], windy, shape_a=shape, rho_m=2090.0)
        checks.append({**p, "ours_impact_ms": r["impact_speed_ms"], "ours_minutes": r["time_s"] / 60, "shape_used": shape})
    out["dark_flight_vs_paper"] = checks

    finds = read_finds()
    masses = sorted({0.005, 0.01, 0.02, 0.05, 0.1, 0.16, 0.32, 0.6})
    n = 40 if quick else 150
    order = np.argsort(hs)
    speed_at = lambda hh: np.interp(hh, hs[order], vs[order])
    scenarios = {
        "at_end": F.strewn_field(end, windy, masses, n_per_mass=n, shape_range=(1.21, 2.7), rho_m=2090.0, wind_sigma=0.3),
        "fragmentation_zone": F.strewn_field(end, windy, masses, n_per_mass=n, shape_range=(1.21, 2.7), rho_m=2090.0, wind_sigma=0.3,
                                             seed=1, release=(end.h_m, LAST_FRAGMENTATION_M, speed_at)),
    }
    field = scenarios["at_end"]
    line = sorted(field["by_mass"], key=lambda b: b["mass_kg"])
    for f in finds:
        al, cr = along_cross(end, f["lat"], f["lon"])
        f.update({"along_m": al, "across_m": cr})
        for name, fld in scenarios.items():
            sims = [p for p in fld["points"] if 0.5 <= p["mass_kg"] / f["mass_kg"] <= 2.0] or fld["points"]
            ac = np.array([along_cross(end, p["lat"], p["lon"]) for p in sims])
            lo, hi = np.percentile(ac[:, 0], [2.5, 97.5])
            # Across the track: the simulated stones' spread at a similar distance along it (no wider than 600 m
            # either side: the paper's corridor of finds; a stone outside both is a miss).
            near = ac[np.abs(ac[:, 0] - al) < 1000] if (np.abs(ac[:, 0] - al) < 1000).sum() >= 5 else ac
            f[name] = {"along_range_m": [float(lo), float(hi)], "along_ok": bool(lo <= al <= hi),
                       "across_offset_m": float(cr - np.median(near[:, 1])), "across_sigma_m": float(np.std(near[:, 1]))}
        f["explained_by"] = next((k for k in scenarios if f[k]["along_ok"]), None)
    blind_contours(end, scenarios["at_end"], masses, finds)
    out["finds"] = finds
    out["fall_line"] = [{k: b[k] for k in ("mass_kg", "lat", "lon", "east_m", "north_m", "n")} for b in line]
    out["strewn_points"] = {k: [[round(p["lon"], 6), round(p["lat"], 6), p["mass_kg"], round(p["shape_a"], 2)] for p in v["points"]]
                            for k, v in scenarios.items()}
    out["end"] = end.__dict__
    out["last_fragmentation_m"] = LAST_FRAGMENTATION_M
    across = [abs(f["at_end"]["across_offset_m"]) for f in finds]
    out["summary"] = {"finds": len(finds), "median_across_m": float(np.median(across)), "max_across_m": float(max(across)),
                      "along_at_end": sum(f["at_end"]["along_ok"] for f in finds),
                      "along_either": sum(f["explained_by"] is not None for f in finds),
                      "blind_inside_68": sum(f["blind"]["inside_68"] for f in finds),
                      "blind_inside_95": sum(f["blind"]["inside_95"] for f in finds),
                      "blind_rule": BLIND_RULE}
    (DATA / "winchcombe_strewn_field.geojson").write_text(json.dumps(F.to_geojson(field, [
        {"lat": f["lat"], "lon": f["lon"], "mass_g": f["mass_g"], "site": f["site"]} for f in finds])))
    (DATA / "winchcombe_strewn_field.kml").write_text(F.to_kml(field, "Winchcombe strewn field (PhysicsLens simulation)"))
    out["outcome"] = outcome(out)
    (DATA / "results.json").write_text(json.dumps(out, indent=1, default=float))
    print("finds:", out["summary"], f"({time.time() - t0:.0f} s)")


# Fixed before any find was compared: where stones are released and which mass bins they are sorted into.
BLIND_RULE = ("stones released at the luminous end, plus at every fragmentation our own deceleration fit detects with a "
              "BIC gain of at least 10 (none was detected for Winchcombe, so the luminous end only); mass bins 5, 10, 20, "
              "50, 100, 160, 320 and 600 g; each find scored in the bin nearest its reported mass, against the 68% and "
              "95% highest-density regions of that bin's simulated landing points")


def blind_contours(end, field: dict, masses: list[float], finds: list[dict]) -> None:
    """Score each find against the 68% and 95% landing regions of its fixed mass bin, computed from the simulation
    alone. A find is inside the p region when fewer than p of the simulated stones land at a higher density than it
    (highest-density regions, so a curved or lopsided field is not forced into an ellipse)."""
    from scipy.stats import gaussian_kde

    by_bin: dict[float, np.ndarray] = {}
    for m in masses:
        pts = [p for p in field["points"] if abs(p["mass_kg"] - m) < 1e-9]
        by_bin[m] = np.array([along_cross(end, p["lat"], p["lon"]) for p in pts])
    for f in finds:
        m = min(masses, key=lambda x: abs(np.log(x / f["mass_kg"])))
        ac = by_bin[m]
        kde = gaussian_kde(ac.T)
        dens = kde(ac.T)
        at = float(kde(np.array([[f["along_m"]], [f["across_m"]]]))[0])
        level = float(np.mean(dens > at))  # the smallest region that contains the find
        f["blind"] = {"mass_bin_kg": m, "hdr_level": level, "inside_68": bool(level <= 0.68), "inside_95": bool(level <= 0.95),
                      "simulated": len(ac)}


def read_report_reference() -> dict:
    """Reference point, apparent ECI radiant and initial speed from the trajectory report used for the orbit."""
    txt = (DATA / "winchcombe_wmpl_report_initial.txt").read_text().splitlines()
    i = next(k for k, ln in enumerate(txt) if ln.startswith("Reference point on the trajectory"))
    when = txt[i + 1].split("Time:")[1].strip().replace(" UTC", "").replace(" ", "T")
    val = lambda ln: float(ln.split("=")[1].split("+/-")[0])
    lat, lon, ht = val(txt[i + 2]), val(txt[i + 3]), val(txt[i + 4])
    j = next(k for k in range(i, len(txt)) if txt[k].startswith("Radiant (apparent in ECI"))
    ra, dec = val(txt[j + 1]), val(txt[j + 2])
    vinit = next(val(txt[k]) for k in range(j, j + 8) if txt[k].strip().startswith("Vinit"))
    return {"when_utc": when, "lat": lat, "lon": lon, "h_m": ht, "ra_deg": ra, "dec_deg": dec, "v_inf_ms": vinit * 1000}


def outcome(out: dict) -> dict:
    s = out["summary"]
    pulls = out["orbit"]["pulls"]
    orbit_ok = all(abs(v) <= 2 for v in pulls.values())
    # Judged only on what was fixed before comparing with the finds: the orbit, and the blind landing regions (release
    # rule and mass bins fixed in advance, 95% regions from the simulation alone). Proven needs at least two thirds of
    # the finds inside them. The fall-line corridor and release anywhere after the last flare were chosen with the
    # finds (or the paper's fragment table) in view, so they are reported as "after review", never in the status.
    blind_ok = s.get("blind_inside_95", 0) * 3 >= 2 * s["finds"]
    proven = orbit_ok and blind_ok
    late = [f"{f['mass_g']} g at {f['place']}" for f in out["finds"] if not f["at_end"]["along_ok"]]
    misses = [f"{f['mass_g']} g at {f['place']}" for f in out["finds"] if f["explained_by"] is None]
    return {
        "headline": (f"Winchcombe, from the published triangulated trajectory (not a video): the flight says it dropped meteorites "
                     f"({out['alpha_beta']['triage']['verdict']} by alpha-beta), and the orbit before it met the Earth matches "
                     f"the published one to {max(abs(v) for v in pulls.values()):.1f} sigma. Blind, with the release rule fixed in "
                     f"advance, {s.get('blind_inside_95', 0)} of {s['finds']} stones land inside their predicted 95% regions and "
                     f"{s['along_at_end']} of {s['finds']} in the predicted stretch along the track; the fall line runs within "
                     f"{s['max_across_m']:.0f} m of all of them, but the predicted spread across the track is too narrow."),
        "action": "Search along the fall line, heavier stones further along the track; download the strewn field (GeoJSON or KML).",
        "artifacts": [{"kind": "geojson"}, {"kind": "kml"}],
        "checked_against": {"name": "perihelion distance q (AU)", "truth": PUBLISHED_ORBIT["q_au"][0], "measured": out["orbit"]["ours"]["q_au"],
                            "relative_error": (out["orbit"]["ours"]["q_au"] - PUBLISHED_ORBIT["q_au"][0]) / PUBLISHED_ORBIT["q_au"][0],
                            "within_interval": abs(pulls["q_au"]) <= 1.96, "source": "McMullan et al. 2023 (orbit); Russell et al. 2024 (find sites)"},
        "confidence": (f"blind (rule fixed before comparing): {s.get('blind_inside_95', 0)} of {s['finds']} finds inside the 95% "
                       f"landing region of their mass bin ({s.get('blind_inside_68', 0)} inside the 68% one); along the track "
                       f"{s['along_at_end']} of {s['finds']}. After review (with published fragment information): every find within "
                       f"{s['max_across_m']:.0f} m of the fall line, and with release anywhere after the last flare (34.9 km) "
                       f"{s['along_either']} of {s['finds']} along it"),
        "status": "proven" if proven else "partial",
        "limits": ([] if blind_ok else [(f"blind, only {s.get('blind_inside_95', 0)} of {s['finds']} finds fall inside the 95% landing "
                    "regions: the simulated stones of one mass land in a strip tens of metres wide across the track, while "
                    "the real finds scatter over hundreds of metres, so the simulation's spread (end point within 100 m, "
                    "winds within 30%, the shape range) is too narrow across the track")])
                  + ([(f"along the track the rule fixed in advance misses {', '.join(late)}; they fit only if stones also left "
                     "earlier, after the last flare, a choice made after seeing the finds")] if late else [])
                  + ([f"not reproduced along the track at all: {', '.join(misses)}"] if misses else []),
        "readable": None}


if __name__ == "__main__":
    main()
