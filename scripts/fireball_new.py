"""Fireballs on new events: the Winchcombe method run blind on a fall whose stones have not been reported yet.

  python scripts/fireball_new.py check                # reproduce Winchcombe's blind result with this pipeline
  python scripts/fireball_new.py wa-20260814 [--register] [--quick]

For a new event the steps are the ones fixed on Winchcombe (scripts/fireball_demo.py), applied before anything is
known about where stones lie:
  1. triage: alpha-beta when a speed profile is published; otherwise the simpler rule SIMPLE_TRIAGE (stated below);
  2. the pre-atmospheric orbit when the data give a direction and a speed (with assumed errors when none are given);
  3. the blind strewn field under the same BLIND_RULE as Winchcombe (release at the luminous end plus fragmentations
     our own deceleration fit detects; fixed mass bins), as 68% and 95% highest-density landing regions per bin;
  4. with --register, the regions are pre-registered (physicslens.prereg) to be scored once finds are published.

Downloads are cached in var/fireball_new/ (gitignored); results go to data/fireball_new/<event>/. Free: public data
only, no model calls.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.astro import fireball as F

_spec = importlib.util.spec_from_file_location("fireball_demo", ROOT / "scripts" / "fireball_demo.py")
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)  # Winchcombe: BLIND_RULE, blind_contours, its data readers (not modified here)

CACHE = ROOT / "var" / "fireball_new"
OUT = ROOT / "data" / "fireball_new"
MASSES_KG = [0.005, 0.01, 0.02, 0.05, 0.1, 0.16, 0.32, 0.6]  # the Winchcombe bins, fixed by BLIND_RULE
SHAPE_RANGE = (1.21, 2.7)  # sphere to the flattest shape matched on Winchcombe (its authors' range)
WIND_SIGMA = 0.3
N_PER_MASS = 150

SIMPLE_TRIAGE = ("without a published speed profile alpha-beta cannot be fitted, so a weaker rule: a meteorite-dropping "
                 "candidate needs (a) an entry speed of 20 km/s or less, (b) a published record of luminous flight at or "
                 "below 35 km, and (c) an independent sign of surviving mass (a Doppler weather radar fall signature "
                 "reported by the analysing agency); all three must hold")


# --- downloads ---------------------------------------------------------------------------------------------------------

def fetch(url: str, path: Path) -> Path:
    """Download once; later runs read the cached copy, so a rerun is reproducible and offline."""
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "physicslens-research"})
        with urllib.request.urlopen(req, timeout=300) as r:
            path.write_bytes(r.read())
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- parsers for the published records ---------------------------------------------------------------------------------

def parse_nasa_skyfalls(html: str) -> dict[str, float]:
    """NASA Meteoroid Environment Office event page: start and end of the trajectory solution, and its speed."""
    import re

    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
    num = r"([+-]?\d+(?:\.\d+)?)"
    s = re.search(rf"Start Lat/Lon {num},? {num}", text)
    e = re.search(rf"End Lat/Lon {num},? {num}", text)
    a = re.search(rf"Altitude {num} km \S+ {num} km", text)
    v = re.search(rf"Speed {num} km/s", text)
    if not (s and e and a and v):
        raise ValueError("Not a NASA skyfalls event page with a trajectory (start, end, altitude, speed).")
    return {"start_lat": float(s[1]), "start_lon": float(s[2]), "end_lat": float(e[1]), "end_lon": float(e[2]),
            "start_h_m": float(a[1]) * 1000, "end_h_m": float(a[2]) * 1000, "speed_ms": float(v[1]) * 1000,
            "radar_signature": "radar" in text.lower() and "meteorites" in text.lower()}


def parse_cneos(doc: dict, when_utc: str) -> dict[str, Any]:
    """The CNEOS fireball record closest in time: peak-brightness place, height and speed. The velocity components
    are kept but checked against the speed, since they are sometimes inconsistent."""
    rows = [dict(zip(doc["fields"], r, strict=True)) for r in doc["data"]]
    t0 = datetime.fromisoformat(when_utc)
    r = min(rows, key=lambda x: abs((datetime.fromisoformat(x["date"]) - t0).total_seconds()))
    lat = float(r["lat"]) * (1 if r["lat-dir"] == "N" else -1)
    lon = float(r["lon"]) * (1 if r["lon-dir"] == "E" else -1)
    out = {"date": r["date"], "lat": lat, "lon": lon, "alt_m": float(r["alt"]) * 1000, "speed_ms": float(r["vel"]) * 1000,
           "impact_energy_kt": float(r["impact-e"])}
    if r.get("vx") is not None:
        comp = math.sqrt(sum(float(r[k]) ** 2 for k in ("vx", "vy", "vz")))
        out["components_norm_ms"] = comp * 1000
        out["components_consistent"] = abs(comp - float(r["vel"])) <= 0.5
    return out


def parse_igra2(text: str, year: int, month: int, day: int, hour: int) -> dict[str, Any]:
    """One sounding from an IGRA2 station file (NOAA NCEI fixed columns): heights above sea level (m, geopotential)
    with wind east and north (m/s). WDIR is where the wind blows from, WSPD is tenths of m/s; -9999 is missing."""
    lines = text.splitlines()
    tag = f"{year:04d} {month:02d} {day:02d} {hour:02d}"
    try:
        i = next(k for k, ln in enumerate(lines) if ln.startswith("#") and ln[13:26] == tag)
    except StopIteration as e:
        raise ValueError(f"No sounding at {tag} UTC in this IGRA2 file.") from e
    rows = []
    for ln in lines[i + 1:]:
        if ln.startswith("#"):
            break
        gph, wdir, wspd = int(ln[16:21]), int(ln[40:45]), int(ln[46:51])
        if gph < -8888 or wdir < -8888 or wspd < -8888:
            continue
        sp = wspd / 10
        rows.append((float(gph), -sp * math.sin(math.radians(wdir)), -sp * math.cos(math.radians(wdir))))
    if len(rows) < 3:
        raise ValueError("The sounding has fewer than three levels with winds.")
    a = np.array(sorted(rows))
    _, keep = np.unique(a[:, 0], return_index=True)
    a = a[keep]
    return {"header": lines[i], "height_m": a[:, 0], "east": a[:, 1], "north": a[:, 2]}


# --- geometry ----------------------------------------------------------------------------------------------------------

def _ecef(lat: float, lon: float, h: float) -> np.ndarray:
    import astropy.units as u
    from astropy.coordinates import EarthLocation

    loc = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, h * u.m)
    return np.array([loc.x.to_value(u.m), loc.y.to_value(u.m), loc.z.to_value(u.m)])


def _enu(lat: float, lon: float) -> np.ndarray:
    """Rows: east, north, up unit vectors at a place, in Earth-fixed axes."""
    la, lo = math.radians(lat), math.radians(lon)
    return np.array([[-math.sin(lo), math.cos(lo), 0.0],
                     [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                     [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])


def track_direction(start: tuple[float, float, float], end: tuple[float, float, float]) -> dict[str, float]:
    """Azimuth (from north) and slope below the horizontal of the straight line from start to end, at the end."""
    d = _ecef(*end) - _ecef(*start)
    e, n, up = _enu(end[0], end[1]) @ (d / np.linalg.norm(d))
    return {"azimuth_deg": math.degrees(math.atan2(e, n)) % 360, "slope_deg": math.degrees(math.asin(-up)),
            "length_m": float(np.linalg.norm(d))}


def apparent_radiant(when_utc: str, start: tuple[float, float, float], end: tuple[float, float, float],
                     speed_ms: float, inertial_speed: bool = False) -> dict[str, float]:
    """The apparent radiant in an Earth-centred inertial frame of date (what pre_entry_orbit takes): the ground-relative
    velocity along the track plus the Earth's rotation at the start point, turned from Earth-fixed axes into TETE.
    inertial_speed: the given speed is already the inertial one (as trajectory reports' Vinit is), so only the
    direction takes the rotation."""
    import astropy.units as u
    from astropy.coordinates import ITRS, TETE, CartesianRepresentation
    from astropy.time import Time

    t = Time(when_utc, scale="utc")
    r = _ecef(*start)
    d = _ecef(*end) - r
    v = speed_ms * d / np.linalg.norm(d) + np.cross([0.0, 0.0, 7.292115e-5], r)
    pts = CartesianRepresentation(np.column_stack([r, r + v * 1000.0]) * u.m)
    tete = ITRS(pts, obstime=t).transform_to(TETE(obstime=t)).cartesian.xyz.to_value(u.m)
    vel = (tete[:, 1] - tete[:, 0]) / 1000.0
    if inertial_speed:
        vel *= speed_ms / np.linalg.norm(vel)
    rad = -vel / np.linalg.norm(vel)
    return {"ra_deg": math.degrees(math.atan2(rad[1], rad[0])) % 360, "dec_deg": math.degrees(math.asin(rad[2])),
            "v_inf_ms": float(np.linalg.norm(vel))}


def orbit_with_errors(when_utc: str, start, end, speed_ms: float, sigma_speed_ms: float, sigma_pos_m: float,
                      n: int = 100, seed: int = 0) -> dict[str, Any]:
    """The orbit from the published line, and its spread when the end points and speed move within stated errors."""
    rng = np.random.default_rng(seed)
    rad = apparent_radiant(when_utc, start, end, speed_ms)
    best = F.pre_entry_orbit(when_utc, *start, rad["ra_deg"], rad["dec_deg"], rad["v_inf_ms"])
    draws = []
    for _ in range(n):
        s = _jitter(start, sigma_pos_m, rng)
        e = _jitter(end, sigma_pos_m, rng)
        r = apparent_radiant(when_utc, s, e, speed_ms + rng.normal(0, sigma_speed_ms))
        try:
            draws.append(F.pre_entry_orbit(when_utc, *s, r["ra_deg"], r["dec_deg"], r["v_inf_ms"]))
        except ValueError:
            continue
    keys = ("a_au", "e", "i_deg", "q_au", "Q_au", "tisserand_jupiter", "v_g_ms")
    ci = {k: [float(np.percentile([d[k] for d in draws], p)) for p in (2.5, 16, 50, 84, 97.5)] for k in keys}
    bound = float(np.mean([d["e"] < 1 for d in draws]))
    return {"radiant": rad, "best": best, "percentiles_2.5_16_50_84_97.5": ci, "fraction_bound": bound, "draws": len(draws)}


def _jitter(p: tuple[float, float, float], sigma_m: float, rng) -> tuple[float, float, float]:
    de, dn, dh = rng.normal(0, sigma_m, 3)
    lat, lon = F.offset_latlon(p[0], p[1], de, dn)
    return lat, lon, p[2] + dh


# --- triage, release heights, the blind field and its regions --------------------------------------------------------

def triage(profile: tuple[np.ndarray, np.ndarray] | None, slope_deg: float, atm: F.Atmosphere,
           facts: dict[str, Any]) -> dict[str, Any]:
    """Alpha-beta and the fragmentation search when heights with speeds exist; the stated simple rule otherwise."""
    if profile is not None:
        hs, vs = profile
        ab = F.fit_alpha_beta(hs, vs, slope_deg)
        sb = F.fit_single_body(hs, vs, slope_deg, atm)
        frag = sb["fragmentation"]
        return {"method": "alpha-beta (Sansom et al. 2019) and the single-body fragmentation search",
                "verdict": ab.triage()["verdict"], "alpha": ab.alpha, "beta": ab.beta, "details": ab.triage(),
                "fragmentation_heights_m": [] if frag is None else [float(frag.jump_h)],
                "fragmentation_delta_bic": None if frag is None else float(sb["single"].bic - frag.bic)}
    checks = {"entry_speed_le_20_kms": facts["entry_speed_ms"] <= 20000,
              "luminous_at_or_below_35_km": facts["deepest_luminous_m"] <= 35000,
              "radar_fall_signature": bool(facts["radar_signature"])}
    return {"method": "simple rule (no speed profile published)", "rule": SIMPLE_TRIAGE, "checks": checks, "facts": facts,
            "verdict": "candidate" if all(checks.values()) else "not a candidate", "fragmentation_heights_m": []}


def blind_field(end: F.EndState, atm: F.Atmosphere, release_heights_m: list[float], speed_at=None,
                n_per_mass: int = N_PER_MASS, rho_m: float = 3500.0, seed: int = 0) -> dict[str, Any]:
    """BLIND_RULE: stones from the luminous end, plus (same number per bin) from each detected fragmentation height."""
    field_ = F.strewn_field(end, atm, MASSES_KG, n_per_mass=n_per_mass, shape_range=SHAPE_RANGE, rho_m=rho_m,
                            wind_sigma=WIND_SIGMA, seed=seed)
    for k, h in enumerate(release_heights_m):
        extra = F.strewn_field(end, atm, MASSES_KG, n_per_mass=n_per_mass, shape_range=SHAPE_RANGE, rho_m=rho_m,
                               wind_sigma=WIND_SIGMA, seed=seed + 1 + k, release=(h, h, speed_at))
        field_["points"] += extra["points"]
    return field_


def regions(end: F.EndState, field_: dict[str, Any], levels=(0.68, 0.95), grid: int = 160) -> list[dict[str, Any]]:
    """Per mass bin, the highest-density landing regions, defined exactly as blind_contours scores a find: a place is
    inside the p region when fewer than a fraction p of the simulated stones land at a higher density than it. The
    density threshold for p is therefore the (1 - p) quantile of the simulated stones' own densities. Outlines are
    drawn on a grid for maps; scoring uses the stored points and the same kernel estimate, not the outline."""
    from contourpy import contour_generator
    from scipy.stats import gaussian_kde

    out = []
    for m in MASSES_KG:
        pts = [p for p in field_["points"] if abs(p["mass_kg"] - m) < 1e-9]
        ac = np.array([W.along_cross(end, p["lat"], p["lon"]) for p in pts])
        kde = gaussian_kde(ac.T)
        dens = kde(ac.T)
        pad = 3 * ac.std(axis=0) + 200
        xa = np.linspace(ac[:, 0].min() - pad[0], ac[:, 0].max() + pad[0], grid)
        xc = np.linspace(ac[:, 1].min() - pad[1], ac[:, 1].max() + pad[1], grid)
        ga, gc = np.meshgrid(xa, xc)
        z = kde(np.vstack([ga.ravel(), gc.ravel()])).reshape(ga.shape)
        cell = (xa[1] - xa[0]) * (xc[1] - xc[0])
        reg = {"mass_kg": m, "n": len(pts), "points_lat_lon": [[round(p["lat"], 6), round(p["lon"], 6)] for p in pts],
               "median_lat_lon": [float(np.median([p["lat"] for p in pts])), float(np.median([p["lon"] for p in pts]))]}
        for lv in levels:
            thr = float(np.quantile(dens, 1 - lv))
            lines = contour_generator(xa, xc, z).lines(thr)
            reg[f"hdr{int(lv * 100)}"] = {"density_threshold": thr, "area_km2": float((z >= thr).sum() * cell / 1e6),
                                          "outlines_lon_lat": [_to_lonlat(end, ln) for ln in lines if len(ln) > 3]}
        out.append(reg)
    return out


def _to_lonlat(end: F.EndState, line: np.ndarray) -> list[list[float]]:
    """Along and across the ground track (W.along_cross) back to longitude and latitude."""
    az = math.radians(end.azimuth_deg)
    res = []
    for al, cr in line[:: max(1, len(line) // 120)]:
        de = al * math.sin(az) + cr * math.cos(az)
        dn = al * math.cos(az) - cr * math.sin(az)
        lat = end.lat + math.degrees(dn / F.EARTH_R_M)
        lon = end.lon + math.degrees(de / (F.EARTH_R_M * math.cos(math.radians(end.lat))))
        res.append([round(lon, 6), round(lat, 6)])
    return res


def score_find(end: F.EndState, region: dict[str, Any], lat: float, lon: float) -> dict[str, Any]:
    """The scoring rule as code: the find's highest-density level within its mass bin's stored landing points."""
    from scipy.stats import gaussian_kde

    ac = np.array([W.along_cross(end, a, b) for a, b in region["points_lat_lon"]])
    kde = gaussian_kde(ac.T)
    at = float(kde(np.array([[x] for x in W.along_cross(end, lat, lon)]))[0])
    level = float(np.mean(kde(ac.T) > at))
    return {"hdr_level": level, "inside_68": level <= 0.68, "inside_95": level <= 0.95}


def mass_bin(mass_kg: float) -> float:
    return min(MASSES_KG, key=lambda x: abs(math.log(x / mass_kg)))


# --- Winchcombe: the same pipeline must give the committed blind result -----------------------------------------------

def winchcombe_check(n_per_mass: int = N_PER_MASS, use_msis: bool = True, fragmentation_search: bool = True) -> dict[str, Any]:
    """Run this pipeline on Winchcombe's committed data and count finds inside their 68% and 95% regions."""
    end = W.end_state()
    atm = F.standard_atmosphere()
    if use_msis:
        try:
            atm = F.msis_atmosphere(W.EVENT_UTC, end.lat, end.lon)
        except (ImportError, OSError, ValueError):
            pass
    windy = W.read_winds(atm, "winds_nottingham_03354_2021-03-01_00Z.csv")
    releases: list[float] = []
    speed_at = None
    if fragmentation_search:
        t, h, d = W.read_points()
        hs, vs = W.speeds(t, h, d)
        tri = triage((hs, vs), end.slope_deg, atm, {})
        releases = tri["fragmentation_heights_m"]
        order = np.argsort(hs)
        speed_at = lambda hh: np.interp(hh, hs[order], vs[order])
    fld = blind_field(end, windy, releases, speed_at, n_per_mass=n_per_mass, rho_m=2090.0)
    finds = W.read_finds()
    for f in finds:
        f["along_m"], f["across_m"] = W.along_cross(end, f["lat"], f["lon"])
    W.blind_contours(end, fld, MASSES_KG, finds)
    return {"finds": len(finds), "inside_68": sum(f["blind"]["inside_68"] for f in finds),
            "inside_95": sum(f["blind"]["inside_95"] for f in finds), "release_heights_m": releases,
            "atmosphere": windy.source, "per_find": [{"site": f["site"], "mass_g": f["mass_g"], **f["blind"]} for f in finds]}


# --- the event: Washington State, 2026 August 14 07:48 UTC -------------------------------------------------------------

class Event:
    """A published fireball: its id, time, title and where each input comes from (a plain class, so the script also
    loads by file path)."""

    def __init__(self, slug: str, when_utc: str, title: str, sources: dict[str, dict[str, str]]) -> None:
        self.slug, self.when_utc, self.title, self.sources = slug, when_utc, title, sources


WA = Event(
    "wa-20260814", "2026-08-14T07:48:33", "Fireball over Douglas County, Washington, 2026 August 14 07:48 UTC",
    {"nasa_meo": {"url": "https://fireballs.ndc.nasa.gov/skyfalls/events/20260814-074833", "file": "nasa_skyfalls/20260814-074833.html",
                  "terms": "NASA Meteoroid Environment Office public event page (US Government work, no copyright in the US)"},
     "cneos": {"url": "https://ssd-api.jpl.nasa.gov/fireball.api?date-min=2026-08-14&date-max=2026-08-15&req-loc=true&vel-comp=true&req-alt=true",
               "file": "cneos/cneos_20260814.json", "terms": "NASA/JPL CNEOS fireball API (US Government data, public)"},
     "sounding": {"url": "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-y2d/USM00072786-data-beg2026.txt.zip",
                  "file": "igra/USM00072786-data-beg2026.txt.zip",
                  "terms": "NOAA NCEI IGRA2 (US Government data, public domain); Spokane WA, WMO 72786"},
     "elevation": {"url": "https://api.open-meteo.com/v1/elevation?latitude=47.80,47.85,47.90,47.95,48.00,47.85,47.85,47.95,47.95&longitude=-119.41,-119.41,-119.41,-119.41,-119.41,-119.30,-119.52,-119.30,-119.52",
                   "file": "elev/elev.json", "terms": "Open-Meteo elevation API (Copernicus DEM GLO-90; data CC BY 4.0, attribution)"}})
SOUNDING_TIME = (2026, 8, 14, 6)  # the Spokane launch nearest the fireball (05:33 UTC, 2.3 h before; next is 17:02 UTC)
LIMITS = [
    ("the published solution ends at the 45.5 km fragmentation at the published 14.6 km/s, so stones start there and "
     "lose no mass while they slow (the dark-flight model has no ablation); real stones shrink and slow sooner, so the "
     "regions are probably placed too far along the track"),
    "no speed profile is published: triage uses the simple rule, not alpha-beta, and no fragmentation fit is possible",
    "position, height and angle errors are not published; they are assumed (2 km, 2 km, 3 degrees)",
    "the satellite record (CNEOS) puts the deep flight about 40 km from the camera line; only the camera line is used",
    "the meteorite type, and so its density (3500 kg/m^3 assumed), is unknown",
]
# Not published, so assumed and fixed here before the field was simulated: the trajectory's place and angle errors.
ASSUMED_SIGMA = {"pos_m": 2000.0, "h_m": 2000.0, "angle_deg": 3.0}


def load_wa() -> dict[str, Any]:
    """Fetch (or read cached) every input of the Washington event and parse it."""
    files = {k: fetch(s["url"], CACHE / s["file"]) for k, s in WA.sources.items()}
    import zipfile

    with zipfile.ZipFile(files["sounding"]) as z:
        snd = parse_igra2(z.read(z.namelist()[0]).decode(), *SOUNDING_TIME)
    meo = parse_nasa_skyfalls(files["nasa_meo"].read_text())
    cneos = parse_cneos(json.loads(files["cneos"].read_text()), WA.when_utc)
    ground = float(np.median(json.loads(files["elevation"].read_text())["elevation"]))
    return {"meo": meo, "cneos": cneos, "sounding": snd, "ground_m": ground,
            "checksums": {k: sha256_file(p) for k, p in files.items()}}


def wa_end_state(inp: dict[str, Any]) -> tuple[F.EndState, dict[str, float]]:
    """The luminous end is where the camera trajectory solution ends (the fragmentation NASA reports); the speed there
    is the published speed, with the gap to the satellite speed at peak brightness as its one-sigma error."""
    m = inp["meo"]
    start, end = (m["start_lat"], m["start_lon"], m["start_h_m"]), (m["end_lat"], m["end_lon"], m["end_h_m"])
    geo = track_direction(start, end)
    sig_v = abs(m["speed_ms"] - inp["cneos"]["speed_ms"])
    es = F.EndState(m["end_lat"], m["end_lon"], m["end_h_m"], m["speed_ms"], geo["azimuth_deg"], geo["slope_deg"],
                    sigma_pos_m=ASSUMED_SIGMA["pos_m"], sigma_h_m=ASSUMED_SIGMA["h_m"], sigma_speed_ms=sig_v,
                    sigma_angle_deg=ASSUMED_SIGMA["angle_deg"], ground_m=inp["ground_m"])
    return es, geo


def run_wa(n_per_mass: int = N_PER_MASS) -> dict[str, Any]:
    inp = load_wa()
    end, geo = wa_end_state(inp)
    m, c = inp["meo"], inp["cneos"]
    try:
        atm = F.msis_atmosphere(WA.when_utc, end.lat, end.lon)
    except (ImportError, OSError, ValueError):
        atm = F.standard_atmosphere()
    snd = inp["sounding"]
    windy = atm.with_wind(snd["height_m"], snd["east"], snd["north"], f"IGRA2 Spokane 72786 {snd['header'][13:26]} UTC")
    facts = {"entry_speed_ms": m["speed_ms"], "deepest_luminous_m": min(m["end_h_m"], c["alt_m"]),
             "deepest_luminous_source": "CNEOS peak brightness" if c["alt_m"] < m["end_h_m"] else "NASA camera solution end",
             "radar_signature": m["radar_signature"]}
    tri = triage(None, end.slope_deg, atm, facts)
    out: dict[str, Any] = {"generated": "scripts/fireball_new.py wa-20260814", "event": WA.title, "inputs": inp | {"sounding": {
        "header": snd["header"], "levels": len(snd["height_m"]), "top_m": float(snd["height_m"].max())}},
        "track": geo, "triage": tri, "end_state": end.__dict__, "atmosphere": windy.source,
        "conflicts": [(f"CNEOS places peak brightness at {c['lat']:.1f}, {c['lon']:.1f}, {c['alt_m'] / 1000:.1f} km; the camera "
                      f"line passes that latitude near {_height_at_lat(m, c['lat']) / 1000:.0f} km, so the two records "
                      "disagree on where the deep flight was; the camera solution (a triangulation) sets the end state, "
                      "CNEOS only the triage"),
                      (f"CNEOS velocity components have norm {c.get('components_norm_ms', float('nan')) / 1000:.1f} km/s against its "
                      f"stated {c['speed_ms'] / 1000:.1f} km/s, so they are not used")]}
    if tri["verdict"] != "candidate":
        return out
    out["orbit"] = orbit_with_errors(WA.when_utc, (m["start_lat"], m["start_lon"], m["start_h_m"]),
                                     (m["end_lat"], m["end_lon"], m["end_h_m"]), m["speed_ms"], 1000.0, ASSUMED_SIGMA["pos_m"])
    fld = blind_field(end, windy, tri["fragmentation_heights_m"], n_per_mass=n_per_mass, rho_m=3500.0)
    out["regions"] = regions(end, fld)
    return out


def _height_at_lat(m: dict[str, float], lat: float) -> float:
    f = (lat - m["start_lat"]) / (m["end_lat"] - m["start_lat"])
    return m["start_h_m"] + f * (m["end_h_m"] - m["start_h_m"])


def prediction_body(res: dict[str, Any]) -> dict[str, Any]:
    """What gets hashed: inputs with sources and checksums, the regions per mass bin, and the rule that scores them."""
    inputs = [{"name": k, "source": s["url"], "terms": s["terms"], "sha256": res["inputs"]["checksums"][k],
               "retrieved_utc": "2026-11-02"} for k, s in WA.sources.items()]
    items = [{"label": f"{r['mass_kg'] * 1000:g} g bin", "mass_kg": r["mass_kg"], "median_lat_lon": r["median_lat_lon"],
              "hdr68": r["hdr68"], "hdr95": r["hdr95"], "landing_points_lat_lon": r["points_lat_lon"]} for r in res["regions"]]
    return {
        "title": WA.title + ": where its meteorites landed, predicted blind",
        "question": "Where do recovered meteorites of each mass lie on the ground, if any are found?",
        "inputs": inputs,
        "method": {"blind_rule": W.BLIND_RULE, "triage": res["triage"], "end_state": res["end_state"],
                   "assumed_errors": ASSUMED_SIGMA, "meteoroid_density_kg_m3": 3500.0,
                   "density_note": "type unknown; ordinary chondrites are the most common falls",
                   "shape_range": SHAPE_RANGE, "wind_scale_sigma": WIND_SIGMA, "atmosphere": res["atmosphere"],
                   "conflicts": res["conflicts"], "orbit": {k: res["orbit"][k] for k in ("radiant", "percentiles_2.5_16_50_84_97.5", "fraction_bound")},
                   "track_record": "on Winchcombe (2021) the same rule put 1 of 9 finds inside its 95% region: the "
                                   "simulated spread across the track was too narrow",
                   "limits": LIMITS},
        "prediction": {"items": items, "frame": "along/across the ground track from the end state (fireball_demo.along_cross)"},
        "scoring": {"rule": ("a hit when a recovered meteorite's reported find site lies inside the predicted 95% region "
                             "of its mass bin (the bin nearest its reported mass); score after finds are published. Inside "
                             "means fewer than 95% of that bin's stored landing points have a higher Gaussian kernel "
                             "density (scipy gaussian_kde, default bandwidth, in metres along and across the track) than "
                             "the find site (scripts/fireball_new.py score_find)"),
                    "score_after_utc": "2027-01-01T00:00:00Z"},
    }


def write_sources(res: dict[str, Any], path: Path) -> None:
    rows = "\n".join(f"| {k} | {s['url']} | {s['terms']} | 2026-11-02 | `{res['inputs']['checksums'][k]}` |"
                     for k, s in WA.sources.items())
    path.write_text(
        "# New fireball (Washington, 2026 August 14): sources\n\n"
        "| Input | URL | Licence and terms | Retrieved | SHA-256 of the cached file |\n|---|---|---|---|---|\n" + rows + "\n\n"
        "Command: `PYTHONPATH=backend .venv/bin/python scripts/fireball_new.py wa-20260814 --register` (downloads cached in "
        "`var/fireball_new/`).\n\n"
        "Searched and not usable for this event: the Global Meteor Network trajectory summaries (CC BY 4.0) have no "
        "solution for it and none of their 2026 events ends below 32 km; the University of Wyoming archive returned no "
        "Spokane (72786) sounding for the date, so the same launch is taken from NOAA's IGRA2. No find had been reported "
        "in public sources as of the retrieval date.\n")


def main() -> None:
    args = sys.argv[1:]
    n = 40 if "--quick" in args else N_PER_MASS
    if not args or args[0] == "check":
        res = winchcombe_check(n_per_mass=n)
        print("Winchcombe through this pipeline:", {k: res[k] for k in ("finds", "inside_68", "inside_95", "release_heights_m")})
        return
    if args[0] != WA.slug:
        raise SystemExit(f"unknown event {args[0]!r}; known: check, {WA.slug}")
    res = run_wa(n)
    d = OUT / WA.slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(res, indent=1, default=float))
    write_sources(res, d / "SOURCES.md")
    print("triage:", res["triage"]["verdict"], res["triage"].get("checks"))
    for r in res.get("regions", []):
        print(f"{r['mass_kg'] * 1000:g} g: median {r['median_lat_lon']}, 95% area {r['hdr95']['area_km2']:.1f} km2")
    if "--register" in args and res["triage"]["verdict"] == "candidate" and "--quick" not in args:
        from physicslens import prereg

        p = prereg.register("fireballs", WA.slug, prediction_body(res))
        print("registered", p.relative_to(ROOT), prereg.sha256(p), datetime.now(UTC).isoformat(timespec="seconds"))


if __name__ == "__main__":
    main()
