"""Halley's comet: the inverse-square law from a video of its 1986 passage, then its return in 2061.

  python scripts/halley_demo.py fetch      NASA/JPL data -> data/halley/ (free public APIs)
  python scripts/halley_demo.py render     -> var/halley/halley.mp4, the video PhysicsLens watches
  python scripts/halley_demo.py analyze    the normal PhysicsLens pipeline on that video (PAID: SAM 3.1 + Muse Spark Open)
  python scripts/halley_demo.py predict    integrate the recovered law to the next perihelion; compare with JPL
  python scripts/halley_demo.py planets    the same return with the giant planets' pulls added: where the miss comes from
  python scripts/halley_demo.py prediction   the discovered law alone and with the planets added, against JPL, with
                                           an error budget -> data/halley/prediction_table.json (and the gallery)
  python scripts/halley_demo.py apparitions  Halley's perihelia 1835 to 2061 from our engine against JPL's: why one
                                           passage cannot fix the next return better than the planets allow

Halley's orbit is a hard test for a law found from pixels: it is a long ellipse (eccentricity 0.967), so near the Sun
the pull is thousands of times stronger than at its far end, and an inverse-square law that is even slightly wrong in
its exponent bends the passage visibly. The video shows the 1986 passage, where the orbit turns hardest: a comet that
moves 60 times faster at perihelion than at aphelion cannot be followed by a tracker over a whole orbit at one frame
rate and scale, but near the Sun it can.

Data: positions of 1P/Halley (JPL small-body record 90000030, the solution fitted to observations through the 1986
apparition) and Earth from JPL Horizons, heliocentric, ecliptic J2000, AU and days. They draw the video and, afterwards,
check the answer. The analysis sees pixels: a dot moving around a fixed point. The prediction integrates the accepted
law alone (Sun only: no planets, no outgassing), so its 2061 date shows how much of Halley's return the Sun explains.

The planets step answers what the rest is. A video of one passage gives the comet's track in its own plane, but the
planets pull from outside that plane, so it starts from JPL's 1985 position and velocity of the comet (3D) and adds
Newton's pull of Jupiter, Saturn, Uranus and Neptune (positions and masses from JPL Horizons; heliocentric, with the
indirect term for the Sun being pulled too). Each source of error is then switched on alone: the Sun's GM from JPL or
from the video, with or without the planets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

DATA = ROOT / "data" / "halley"
OUT = ROOT / "var" / "halley"
HORIZONS = "https://ssd.jpl.nasa.gov/api/horizons.api"
BODIES = {"halley": "90000030;", "earth": "399"}
SPANS = {"": ("1982-01-01", "1990-01-01"), "_2061": ("2059-01-01", "2063-12-31")}
# Horizons ids: the planet's system barycentre for its position, the planet for its GM. The inner planets moved the
# 2061 date by half a day in a trial with all eight, so only the giants are kept (smaller data, same answer).
GIANTS = {"jupiter": ("5", "599"), "saturn": ("6", "699"), "uranus": ("7", "799"), "neptune": ("8", "899")}
PLANET_SPAN = ("1984-06-01", "2063-12-31")
TRAIN_START, TRAIN_STOP = "1985-01-01", "1987-05-01"  # the 1986 passage (perihelion 1986-02-09)
SIZE = 1080
FPS = 30
MAX_FRAMES = 590  # the analysis takes clips up to 20 s
AU_KM = 149_597_870.7
JD_UNIX0 = 2440587.5


def _get(url: str, params: dict[str, str]) -> dict:
    with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(params)}", timeout=120) as r:
        return json.loads(r.read())


def fetch() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for suffix, (start, stop) in SPANS.items():
        for name, command in BODIES.items():
            if suffix and name == "earth":
                continue
            params = {"format": "json", "COMMAND": f"'{command}'", "EPHEM_TYPE": "VECTORS", "CENTER": "'500@10'",
                      "START_TIME": f"'{start}'", "STOP_TIME": f"'{stop}'", "STEP_SIZE": "'1 d'",
                      "VEC_TABLE": "'2'", "REF_PLANE": "'ECLIPTIC'", "OUT_UNITS": "'AU-D'", "CSV_FORMAT": "'YES'"}
            d = _get(HORIZONS, params)
            text = d["result"]
            if "$$SOE" not in text:
                sys.exit(f"Horizons answered without a vector table for {name}: {text[:300]}")
            rows = []
            for line in text.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines():
                f = [v.strip() for v in line.split(",")]
                rows.append([float(f[0]), *(float(v) for v in f[2:8])])
            header = (f"# JPL Horizons API ({d['signature']['source']} {d['signature']['version']}), fetched "
                      f"{datetime.now(UTC):%Y-%m-%d}, COMMAND={command}, CENTER=500@10 (Sun), ECLIPTIC J2000, AU and AU/day, TDB\n"
                      "# jd_tdb,x,y,z,vx,vy,vz")
            np.savetxt(DATA / f"{name}{suffix}.csv", np.array(rows), delimiter=",", header=header, comments="", fmt="%.15g")
            print(f"{name}{suffix}: {len(rows)} rows")
    sun = _get(HORIZONS, {"format": "json", "COMMAND": "'10'", "OBJ_DATA": "'YES'", "MAKE_EPHEM": "'NO'"})["result"]
    line = next(ln for ln in sun.splitlines() if "GM, km^3/s^2" in ln)
    gm_km3_s2 = float(line.split("=")[1].split()[0])
    (DATA / "sun_gm.json").write_text(json.dumps({
        "gm_km3_s2": gm_km3_s2, "gm_au3_d2": gm_km3_s2 * 86400**2 / AU_KM**3,
        "source": f"JPL Horizons API, Sun (10) physical data, fetched {datetime.now(UTC):%Y-%m-%d}"}, indent=1))


def fetch_planets() -> None:
    import re
    gms = {}
    for name, (bary, body) in GIANTS.items():
        txt = _get(HORIZONS, {"format": "json", "COMMAND": f"'{body}'", "OBJ_DATA": "'YES'", "MAKE_EPHEM": "'NO'"})["result"]
        m = re.search(r"GM[ ,(]*km\^3/s\^2\)?\s*=\s*([\d.]+)", txt)
        if not m:
            sys.exit(f"Horizons gave no GM for {name}")
        gms[name] = float(m.group(1))
        d = _get(HORIZONS, {"format": "json", "COMMAND": f"'{bary}'", "EPHEM_TYPE": "VECTORS", "CENTER": "'500@10'",
                            "START_TIME": f"'{PLANET_SPAN[0]}'", "STOP_TIME": f"'{PLANET_SPAN[1]}'", "STEP_SIZE": "'8 d'",
                            "VEC_TABLE": "'1'", "REF_PLANE": "'ECLIPTIC'", "OUT_UNITS": "'AU-D'", "CSV_FORMAT": "'YES'"})
        rows = []
        for line in d["result"].split("$$SOE")[1].split("$$EOE")[0].strip().splitlines():
            f = [v.strip() for v in line.split(",")]
            rows.append([float(f[0]), *(float(v) for v in f[2:5])])
        header = (f"# JPL Horizons API ({d['signature']['source']} {d['signature']['version']}), fetched "
                  f"{datetime.now(UTC):%Y-%m-%d}, COMMAND={bary} ({name} system barycentre), CENTER=500@10 (Sun), "
                  "ECLIPTIC J2000, AU, TDB\n# jd_tdb,x,y,z")
        np.savetxt(DATA / f"{name}.csv", np.array(rows), delimiter=",", header=header, comments="", fmt="%.12g")
        print(f"{name}: {len(rows)} rows, GM {gms[name]} km^3/s^2")
    (DATA / "planets_gm.json").write_text(json.dumps({
        "gm_km3_s2": gms, "note": "GM of each planet alone (Horizons physical data); moons are left out",
        "source": f"JPL Horizons API, physical data of 599, 699, 799, 899, fetched {datetime.now(UTC):%Y-%m-%d}"}, indent=1))


def load(name: str) -> np.ndarray:
    return np.loadtxt(DATA / f"{name}.csv", delimiter=",", skiprows=2)


def jd(date: str) -> float:
    return datetime.fromisoformat(date).replace(tzinfo=UTC).timestamp() / 86400 + JD_UNIX0


def date_of(jd_tdb: float) -> str:
    return datetime.fromtimestamp((jd_tdb - JD_UNIX0) * 86400, UTC).strftime("%Y-%m-%d")


def plane_basis(c: np.ndarray) -> np.ndarray:
    """Rows e1, e2, n: n along the comet's mean angular momentum over the video, so the orbit is seen face on."""
    train = c[(c[:, 0] >= jd(TRAIN_START)) & (c[:, 0] < jd(TRAIN_STOP))]
    h = np.cross(train[:, 1:4], train[:, 4:7]).mean(axis=0)
    n = h / np.linalg.norm(h)
    e1 = np.array([1.0, 0.0, 0.0]) - n[0] * n
    e1 /= np.linalg.norm(e1)
    return np.vstack([e1, np.cross(n, e1), n])


def layout(xy: np.ndarray) -> tuple[float, np.ndarray]:
    """Scale and Sun position so the whole passage and the Sun fit the frame with a margin."""
    pts = np.vstack([xy, [[0.0, 0.0]]])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    scale = 0.8 * SIZE / float((hi - lo).max())
    centre = (lo + hi) / 2
    sun_px = np.array([SIZE / 2 - centre[0] * scale, SIZE / 2 + centre[1] * scale])
    return scale, sun_px


def frames_for_render() -> tuple[list, object, np.ndarray, np.ndarray, float, np.ndarray, float]:
    from physicslens.sky.render import Drawn, Frame
    comet, earth = load("halley"), load("earth")
    basis = plane_basis(comet)
    step = (jd(TRAIN_STOP) - jd(TRAIN_START)) / MAX_FRAMES
    frames_jd = jd(TRAIN_START) + step * np.arange(MAX_FRAMES)
    interp = lambda tab: np.column_stack([np.interp(frames_jd, tab[:, 0], tab[:, i]) for i in (1, 2, 3)])
    c_xy = interp(comet) @ basis[:2].T
    e_xy = interp(earth) @ basis[:2].T
    scale, sun_px = layout(c_xy)
    to_px = lambda xy: np.column_stack([sun_px[0] + xy[:, 0] * scale, sun_px[1] - xy[:, 1] * scale])
    bodies = [Drawn("Sun", sun_px, 18, (60, 200, 255)), Drawn("Earth", to_px(e_xy), 10, (230, 150, 60)),
              Drawn("Halley", to_px(c_xy), 13, (235, 235, 235))]
    frame = Frame(SIZE, FPS, [date_of(w) for w in frames_jd],
                  [f"Rendered from NASA/JPL Horizons data. 1 frame = {step:.2f} days.",
                   "white: comet 1P/Halley   blue: Earth   yellow: Sun"])
    return bodies, frame, basis, frames_jd, scale, sun_px, step


def render() -> Path:
    from physicslens.sky.render import render as draw
    bodies, frame, basis, frames_jd, scale, sun_px, step = frames_for_render()
    speed = np.linalg.norm(np.diff(bodies[2].px, axis=0), axis=1)
    print(f"scale {scale:.1f} px/AU, one frame = {step:.2f} days, fastest move {speed.max():.1f} px/frame")
    out = draw(bodies, frame, OUT / "halley.mp4", {
        "start_jd": float(frames_jd[0]), "step_days": step, "px_per_au": scale, "sun_px": sun_px.tolist(),
        "basis": basis.tolist()}, threads=2, meta_path=OUT / "render.json")
    print(f"{out}: {len(frames_jd)} frames, {len(frames_jd) / FPS:.1f} s")
    return out


FRAME_NOTE = ("Rendered face-on video of an orbital plane. The coordinate origin (0, 0) is the fixed yellow body. Units: "
              "wherever the numbers say m, read AU (astronomical units); wherever they say s, read day. Write any law for "
              "the white dot's acceleration in x, y (AU) and vx, vy (AU/day), with parameter units in AU and days.")


def analyze() -> Path:
    import os
    os.environ.setdefault("APP_ENV", "dev")
    from physicslens.costguard import CostGuardClient
    from physicslens.pipeline import RunOptions
    from physicslens.pipeline import analyze as run_pipeline
    from physicslens.settings import load_settings
    meta = json.loads((OUT / "render.json").read_text())
    settings = load_settings()
    if settings.mock_models:
        sys.exit("analyze needs MOCK_MODELS=0: it calls SAM 3.1 and Muse Spark Open for real.")
    client = CostGuardClient(settings)
    options = RunOptions(
        audience="dev", object_hint="the white dot (the comet)",
        known_px_per_unit=meta["px_per_au"], origin_px=tuple(meta["sun_px"]),
        slow_motion=1.0 / (meta["step_days"] * meta["fps"]),  # physics time in days
        units={"length": "AU", "time": "day"}, frame_note=FRAME_NOTE,
    )
    run_dir = OUT / "run"
    result = run_pipeline(client, OUT / "halley.mp4", run_dir, f"halley-{int(time.time())}", options,
                          lambda m: print("  ", m, flush=True))
    (run_dir / "result.json").write_text(json.dumps(result, indent=1))
    print("verdict:", result["status"], "|", (result.get("equation") or {}).get("title"), "|", result["reason"])
    return run_dir / "result.json"


def predict() -> dict:
    from physicslens.physics.custom import spec_for
    from physicslens.physics.models import derivative
    from scipy.integrate import solve_ivp
    meta = json.loads((OUT / "render.json").read_text())
    result = json.loads((OUT / "run" / "result.json").read_text())
    if result.get("status") != "accepted" or "fit_state" not in result:
        sys.exit(f"No accepted law to predict with (verdict: {result.get('status')}): {result.get('reason')}")
    fs = result["fit_state"]
    spec = spec_for(fs)
    jd0 = meta["start_jd"] + fs["first_frame"] * meta["step_days"]
    later = load("halley_2061")
    jpl_peri = float(later[np.argmin(np.linalg.norm(later[:, 1:4], axis=1)), 0])
    # Seventy-five years past many perihelia: integrated here with tight tolerances, not the fitter's quick simulate.
    span = jpl_peri + 3 * 365 - jd0
    sol = solve_ivp(lambda t, s: derivative(spec, s, fs["params"], fs["ctx"], t), (0.0, span), np.asarray(fs["y0"]),
                    method="DOP853", rtol=1e-10, atol=1e-12, dense_output=True)
    if not sol.success:
        sys.exit(f"The recovered law could not be integrated to 2061: {sol.message}")
    t = np.arange(0.0, span, 0.25)
    r = np.linalg.norm(sol.sol(t)[:2], axis=0)
    # The next perihelion after the video: the first minimum of distance more than 40 years on.
    far = t > 40 * 365
    k = int(np.argmin(np.where(far, r, np.inf)))
    pred_peri = jd0 + float(t[k])
    params = fs["params"]
    units = {p["name"]: p.get("unit", "") for p in (fs.get("free_model") or {}).get("params", [])}
    gm_name = next((n for n, u in units.items() if "AU^3" in u.replace(" ", "")), None)
    sun = json.loads((DATA / "sun_gm.json").read_text())
    gm = params.get(gm_name) if gm_name else None
    out = {
        "law": spec.title, "latex": spec.latex, "params": params,
        "gm_recovered_au3_d2": gm, "gm_jpl_au3_d2": sun["gm_au3_d2"], "gm_source": sun["source"],
        "gm_error_percent": None if gm is None else (gm / sun["gm_au3_d2"] - 1) * 100,
        "video": f"{TRAIN_START} to {TRAIN_STOP}, one frame per {meta['step_days']:.2f} days",
        "jpl_perihelion_2061": date_of(jpl_peri),
        "predicted_next_perihelion": date_of(pred_peri),
        "prediction_error_days": pred_peri - jpl_peri,
        "predicted_perihelion_distance_au": float(r[k]),
        "jpl_perihelion_distance_au": float(np.linalg.norm(later[:, 1:4], axis=1).min()),
    }
    (OUT / "prediction.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def return_date(gm_sun: float, planets: bool) -> tuple[float, float]:
    """Days by which the integrated 2061 perihelion misses JPL's, and the perihelion distance (AU)."""
    from physicslens.physics.nbody import Body, propagate
    start = load("halley")
    s0 = start[np.argmin(np.abs(start[:, 0] - jd(TRAIN_START)))]
    later = load("halley_2061")
    jpl_peri = float(later[np.argmin(np.linalg.norm(later[:, 1:4], axis=1)), 0])
    gm_planets = json.loads((DATA / "planets_gm.json").read_text())["gm_km3_s2"] if planets else {}
    bodies = [Body.from_table(name, gm, load(name)) for name, gm in gm_planets.items()]
    at = propagate(s0[1:7], s0[0], jpl_peri + 900, gm_sun, bodies, rtol=1e-11, atol=1e-13)
    when = s0[0] + np.arange(40 * 365, jpl_peri + 900 - s0[0], 0.05)
    r = np.linalg.norm(at(when)[:3], axis=0)
    k = int(np.argmin(r))
    return float(when[k]) - jpl_peri, float(r[k])


def planets() -> dict:
    """Where the Sun-only miss comes from: each source switched on alone, from the same 1985 start."""
    gm_jpl = json.loads((DATA / "sun_gm.json").read_text())["gm_au3_d2"]
    gm_video = json.loads((OUT / "prediction.json").read_text())["gm_recovered_au3_d2"]
    rows = []
    for label, gm, with_planets in [("Sun only, JPL's GM", gm_jpl, False), ("Sun only, GM from the video", gm_video, False),
                                    ("Sun and giant planets, JPL's GM", gm_jpl, True),
                                    ("Sun and giant planets, GM from the video", gm_video, True)]:
        days, q = return_date(gm, with_planets)
        rows.append({"case": label, "error_days": round(days, 1), "perihelion_au": round(q, 4)})
        print(f"{label:42s} {days:+8.1f} days   q = {q:.4f} AU", flush=True)
    out = {"start": f"JPL state of 1P/Halley on {TRAIN_START}", "jpl_perihelion_2061": date_of(
        float(load("halley_2061")[np.argmin(np.linalg.norm(load("halley_2061")[:, 1:4], axis=1)), 0])),
        "gm_sun_video_au3_d2": gm_video, "gm_sun_jpl_au3_d2": gm_jpl, "cases": rows}
    (OUT / "planets.json").write_text(json.dumps(out, indent=1))
    return out


def apparitions() -> dict:
    """Every perihelion from 1835 to 2061: our engine (every planet, from JPL's J2000 state, backward and forward)
    against JPL's osculating time of perihelion near each one. The orbit-to-orbit period changes by months, which is
    why a single passage cannot pin down the next return better than the planets allow."""
    from physicslens.sky.ephem import J2000, catalogue_system, sun_gm
    from physicslens.sky.nbody import Options, simulate
    path = DATA / "halley_elements.csv"
    if not path.is_file():
        d = _get(HORIZONS, {"format": "json", "COMMAND": "'90000030;'", "EPHEM_TYPE": "ELEMENTS", "CENTER": "'500@10'",
                            "START_TIME": "'1835-01-01'", "STOP_TIME": "'2062-12-31'", "STEP_SIZE": "'30 d'",
                            "REF_PLANE": "'ECLIPTIC'", "CSV_FORMAT": "'YES'"})
        rows = [[float(f[0]), float(f[2]), float(f[3]), float(f[7])] for f in
                ([v.strip() for v in ln.split(",")] for ln in d["result"].split("$$SOE")[1].split("$$EOE")[0].strip().splitlines())]
        np.savetxt(path, np.array(rows), delimiter=",", comments="", fmt="%.12g",
                   header=f"# JPL Horizons API ({d['signature']['source']} {d['signature']['version']}), fetched "
                          f"{datetime.now(UTC):%Y-%m-%d}, osculating elements of 1P/Halley (90000030), heliocentric, "
                          "every 30 days\n# jd_tdb,eccentricity,perihelion_distance_km,time_of_perihelion_jd")
    el = np.loadtxt(path, delimiter=",", skiprows=2)
    jpl: list[float] = []
    for tp in el[:, 3]:  # one perihelion per apparition: the Tp from the elements nearest to it in time
        if not jpl or abs(tp - jpl[-1]) > 20 * 365.25:
            jpl.append(float(el[np.argmin(np.abs(el[:, 0] - tp)), 3]))
    massive = ["Mercury", "Venus", "Earth", "Moon", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto", "Ceres"]
    found = []
    for lo, hi in ((jpl[0] - 400, J2000), (J2000, jpl[-1] + 400)):
        grid = np.arange(lo, hi, 1.0)
        backward = hi == J2000
        jd_eval = grid[::-1] if backward else grid
        # 170 years backward with Mercury in: allowed a quarter of an hour (it is a one-off, offline step).
        run = simulate(sun_gm(), catalogue_system([*massive, "Halley"]), J2000, jd_eval, Options(max_seconds=900))
        r = np.linalg.norm(run.of("Halley")[0], axis=1)
        jd_sorted, r_sorted = (jd_eval[::-1], r[::-1]) if backward else (jd_eval, r)
        for k in range(1, len(r_sorted) - 1):
            if r_sorted[k] < r_sorted[k - 1] and r_sorted[k] <= r_sorted[k + 1] and r_sorted[k] < 1.0:
                a, b, c = r_sorted[k - 1], r_sorted[k], r_sorted[k + 1]  # a parabola through three daily samples
                found.append(float(jd_sorted[k] + 0.5 * (a - c) / (a - 2 * b + c)))
    rows = []
    for t_jpl in jpl:
        ours = min(found, key=lambda t: abs(t - t_jpl))
        rows.append({"jpl_perihelion": date_of(t_jpl), "ours_minus_jpl_days": ours - t_jpl, "jpl_jd": t_jpl, "ours_jd": ours})
    for prev, cur in pairwise(rows):
        cur["jpl_period_years"] = (cur["jpl_jd"] - prev["jpl_jd"]) / 365.25
        cur["ours_period_years"] = (cur["ours_jd"] - prev["ours_jd"]) / 365.25
    periods = [r["jpl_period_years"] for r in rows if "jpl_period_years" in r]
    out = {"source": "our engine from JPL's J2000 state, every planet; JPL osculating time of perihelion",
           "perihelia": rows, "period_spread_years": [min(periods), max(periods)]}
    (DATA / "apparitions.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in rows:
        print(r["jpl_perihelion"], f"{r['ours_minus_jpl_days']:+.2f} d", f"{r.get('jpl_period_years', 0):.2f} y")
    return out


def prediction(draws: int = 12) -> dict:
    """The law found from the video (its fitted GM and fitted starting state), carried to 2061 alone and with the
    planets added as known forcing (from JPL, not from the video), against JPL; then where the remaining miss comes
    from: the GM's own uncertainty, the fitted state's (Monte Carlo of the fit's covariance), and the physics left
    out (relativity; the comet's outgassing, which JPL models)."""
    from physicslens.sky.ephem import J2000, catalogue_system, sun_gm
    from physicslens.sky.nbody import Options, simulate
    from physicslens.sky.prediction import PROVENANCE, lift, planet_paths, propagate, state_covariance
    result = json.loads((ROOT / "gallery" / "gallery-halley" / "result.json").read_text())
    meta = json.loads((DATA / "render.json").read_text())
    fs = result["fit_state"]
    jd0, state = lift(fs, meta)
    later = load("halley_2061")
    r_jpl = np.linalg.norm(later[:, 1:4], axis=1)
    jpl_peri = float(later[np.argmin(r_jpl), 0])
    jd_eval = np.concatenate([np.arange(jd0 + 30, jpl_peri - 900, 30.0), np.arange(jpl_peri - 900, jpl_peri + 900, 0.05)])
    near = jd_eval >= jpl_peri - 900
    paths = planet_paths(jd0, float(jd_eval[-1]))

    def peri(gm: float, st: np.ndarray, planets: bool, gr: bool = False) -> tuple[float, np.ndarray]:
        pos = propagate(gm, st, jd0, jd_eval, planets, gr, prescribed=paths)
        r = np.linalg.norm(pos, axis=1)[near]
        return float(jd_eval[near][np.argmin(r)] - jpl_peri), r

    mu = fs["params"]["mu"]
    mu_std = next(p["std"] for p in result["parameters"] if p["name"] == "mu")
    alone, r_alone = peri(mu, state, False)
    added, r_added = peri(mu, state, True)
    plus, _ = peri(mu + mu_std, state, True)
    minus, _ = peri(mu - mu_std, state, True)
    mean, cov = state_covariance(result)
    rng = np.random.default_rng(0)
    mc = [peri(d[0], lift({**fs, "y0": list(d[1:])}, meta)[1], True)[0]
          for d in rng.multivariate_normal(mean, cov, size=draws)]
    # Missing physics, measured from JPL's own starting state so the video's errors are out of it.
    massive = ["Mercury", "Venus", "Earth", "Moon", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto", "Ceres"]
    grid = np.concatenate([[J2000], np.arange(jpl_peri - 900, jpl_peri + 900, 0.05)])
    def jpl_start(gr: bool) -> float:
        run = simulate(sun_gm(), catalogue_system([*massive, "Halley"]), J2000, grid, Options(gr=gr, max_seconds=900))
        r = np.linalg.norm(run.of("Halley")[0][1:], axis=1)
        return float(grid[1:][np.argmin(r)] - jpl_peri)
    newton, with_gr = jpl_start(False), jpl_start(True)
    rows = [
        {"case": "Discovered law alone (the video's Sun, the video's start)", "days_from_jpl": alone},
        {"case": f"Discovered law with the planets ({PROVENANCE})", "days_from_jpl": added},
        {"case": "JPL", "date": date_of(jpl_peri), "days_from_jpl": 0.0},
    ]
    out = {
        "law": result["equation"]["title"], "start": date_of(jd0), "rows": rows,
        "error_budget_days": {
            "gm_one_sigma": abs(plus - minus) / 2,
            "gm_and_state_monte_carlo_std": float(np.std(mc)), "monte_carlo_draws": draws,
            "relativity": with_gr - newton,
            "left_after_planets_and_relativity_from_jpl_state": with_gr,
        },
        "notes": ("Relativity and the remainder are measured from JPL's J2000 state of the comet with every planet: "
                  "the remainder is mostly the comet's outgassing, which JPL models and we do not."),
    }
    (DATA / "prediction_table.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    t_near = jd_eval[near]
    pick = np.linspace(0, len(t_near) - 1, 300).astype(int)
    chart = {"provenance": PROVENANCE, "x_label": "days from JPL's perihelion", "y_label": "distance from the Sun, AU",
             "t": (t_near[pick] - jpl_peri).tolist(), "alone": r_alone[pick].tolist(), "with_planets": r_added[pick].tolist(),
             "jpl": np.interp(t_near[pick], later[:, 0], r_jpl).tolist(), "rows": rows}
    (ROOT / "gallery" / "gallery-halley" / "prediction.json").write_text(json.dumps(chart, default=float) + "\n")
    print(json.dumps(out, indent=1, default=float))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fetch", "fetch-planets", "render", "analyze", "predict", "planets", "apparitions",
                                     "prediction"])
    a = ap.parse_args()
    {"fetch": fetch, "fetch-planets": fetch_planets, "render": render, "analyze": analyze, "predict": predict,
     "planets": planets, "apparitions": apparitions, "prediction": prediction}[a.step]()


if __name__ == "__main__":
    main()
