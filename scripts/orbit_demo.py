"""Orbit mode: rediscover Newton's law of gravity from a video of an asteroid, then predict where it will be.

  python scripts/orbit_demo.py fetch      NASA/JPL data -> data/orbit/ (free public APIs; cached for offline use)
  python scripts/orbit_demo.py render     -> var/orbit/apophis.mp4, the video PhysicsLens watches
  python scripts/orbit_demo.py analyze    the normal PhysicsLens pipeline on that video (PAID: SAM 3.1 + Muse Spark Open)
  python scripts/orbit_demo.py predict    integrate the recovered law to April 2029; compare with JPL; plots + report
  python scripts/orbit_demo.py fetch-planets   the planets' and Moon's positions and masses (free)
  python scripts/orbit_demo.py planets    the 2029 flyby with the planets and the Moon added: where the miss comes from
  python scripts/orbit_demo.py aftermath  Apophis's orbit before and after the 2029 flyby, our engine against JPL
  python scripts/orbit_demo.py prediction  the discovered law alone and with the planets added, against JPL, with an
                                          error budget -> data/orbit/prediction_table.json (and the gallery)
  python scripts/orbit_demo.py all        everything above (asks before the paid step)

Data: positions and velocities of 99942 Apophis and Earth from JPL Horizons (heliocentric, ecliptic J2000, AU and
days), and Apophis's official 2029 close approach from the JPL Close-Approach Data API. They are only used to draw
the video and, afterwards, to check the prediction. The analysis sees pixels: a dot moving around a fixed point.

The video is a top-down view of Apophis's own orbital plane (the plane perpendicular to its mean angular momentum
over the training years), so a 2D video loses as little as possible. Earth is drawn for context, projected onto the
same plane. Lengths are in AU and time in days (GM of the Sun is about 3e-4 AU^3/day^2 in these units).

The prediction integrates the law the pipeline accepted, exactly as it was fitted, forward from the fitted state; it
lives in the orbital plane, which is lifted back to 3D with the same basis to measure the distance to Earth. Sun only:
no planets, no relativity, no Yarkovsky effect. This is a deterministic prediction compared with JPL's answer, not an
impact probability.

The planets step starts from JPL's 3D state of Apophis on the video's last day and adds Newton's pull of every planet
and the Moon (physics/nbody.py), switching each source of error on alone: the Sun's GM from JPL or from the video,
with or without the planets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

DATA = ROOT / "data" / "orbit"
OUT = ROOT / "var" / "orbit"
HORIZONS = "https://ssd.jpl.nasa.gov/api/horizons.api"
CAD = "https://ssd-api.jpl.nasa.gov/cad.api"
BODIES = {"apophis": "99942;", "earth": "399"}
FETCH_START, FETCH_STOP = "2018-01-01", "2029-12-31"
TRAIN_START, TRAIN_STOP = "2022-01-01", "2026-01-01"  # what the video shows; everything after is the future
# One frame = 2.5 days: the asteroid moves about 19 px per frame, less than its own 26 px disc, so a video tracker can
# follow it. At 5 days it jumped 38 px past a 14 px dot and SAM lost it after 3 frames (a real, paid run).
STEP_DAYS = 2.5
SIZE = 1080
PX_PER_AU = 440.0
FPS = 30
AU_KM = 149_597_870.7
JD_UNIX0 = 2440587.5
# Horizons ids: (position, GM). Earth's position is earth.csv (399); the outer planets' are their system barycentres.
PERTURBERS = {"mercury": ("199", "199"), "venus": ("299", "299"), "earth": (None, "399"), "moon": ("301", "301"),
              "mars": ("4", "499"), "jupiter": ("5", "599"), "saturn": ("6", "699"), "uranus": ("7", "799"),
              "neptune": ("8", "899")}
PLANET_SPAN = ("2025-12-01", "2029-06-01")


# ---------- fetch ----------

def _get(url: str, params: dict[str, str]) -> dict:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(full, timeout=120) as r:
        return json.loads(r.read())


def fetch() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    # Daily for the whole span, and every 10 minutes around the flyby: a close pass lasts hours, not days.
    for suffix, start, stop, step in (("", FETCH_START, FETCH_STOP, "1 d"), ("_flyby", "2029-04-11", "2029-04-16", "10 m")):
        _fetch_vectors(suffix, start, stop, step)
    cad = _get(CAD, {"des": "99942", "date-min": "2029-01-01", "date-max": "2029-12-31", "dist-max": "0.2", "body": "Earth"})
    cad["fetched"] = f"{datetime.now(UTC):%Y-%m-%d}"
    (DATA / "cad_2029.json").write_text(json.dumps(cad, indent=1))
    sun = _get(HORIZONS, {"format": "json", "COMMAND": "'10'", "OBJ_DATA": "'YES'", "MAKE_EPHEM": "'NO'"})["result"]
    line = next(ln for ln in sun.splitlines() if "GM, km^3/s^2" in ln)
    gm_km3_s2 = float(line.split("=")[1].split()[0])
    (DATA / "sun_gm.json").write_text(json.dumps({
        "gm_km3_s2": gm_km3_s2, "gm_au3_d2": gm_km3_s2 * 86400**2 / AU_KM**3,
        "source": f"JPL Horizons API, Sun (10) physical data, fetched {datetime.now(UTC):%Y-%m-%d}"}, indent=1))
    print("GM Sun:", gm_km3_s2, "km^3/s^2")
    print("close approach:", cad["data"][0][3], "TDB,", float(cad["data"][0][4]) * AU_KM, "km")


def _fetch_vectors(suffix: str, start: str, stop: str, step: str) -> None:
    for name, command in BODIES.items():
        params = {"format": "json", "COMMAND": f"'{command}'", "EPHEM_TYPE": "VECTORS", "CENTER": "'500@10'",
                  "START_TIME": f"'{start}'", "STOP_TIME": f"'{stop}'", "STEP_SIZE": f"'{step}'",
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


def fetch_planets() -> None:
    import re
    gms = {}
    for name, (pos, body) in PERTURBERS.items():
        txt = _get(HORIZONS, {"format": "json", "COMMAND": f"'{body}'", "OBJ_DATA": "'YES'", "MAKE_EPHEM": "'NO'"})["result"]
        m = re.search(r"GM[ ,(]*km\^3/s\^2\)?\s*=\s*([\d.]+)", txt)
        if not m:
            sys.exit(f"Horizons gave no GM for {name}")
        gms[name] = float(m.group(1))
        if pos is None:
            continue
        d = _get(HORIZONS, {"format": "json", "COMMAND": f"'{pos}'", "EPHEM_TYPE": "VECTORS", "CENTER": "'500@10'",
                            "START_TIME": f"'{PLANET_SPAN[0]}'", "STOP_TIME": f"'{PLANET_SPAN[1]}'", "STEP_SIZE": "'1 d'",
                            "VEC_TABLE": "'1'", "REF_PLANE": "'ECLIPTIC'", "OUT_UNITS": "'AU-D'", "CSV_FORMAT": "'YES'"})
        rows = []
        for line in d["result"].split("$$SOE")[1].split("$$EOE")[0].strip().splitlines():
            f = [v.strip() for v in line.split(",")]
            rows.append([float(f[0]), *(float(v) for v in f[2:5])])
        header = (f"# JPL Horizons API ({d['signature']['source']} {d['signature']['version']}), fetched "
                  f"{datetime.now(UTC):%Y-%m-%d}, COMMAND={pos} ({name}), CENTER=500@10 (Sun), ECLIPTIC J2000, AU, TDB\n"
                  "# jd_tdb,x,y,z")
        np.savetxt(DATA / f"planet_{name}.csv", np.array(rows), delimiter=",", header=header, comments="", fmt="%.15g")
        print(f"{name}: {len(rows)} rows, GM {gms[name]} km^3/s^2")
    (DATA / "planets_gm.json").write_text(json.dumps({
        "gm_km3_s2": gms, "note": "GM of each body alone (Horizons physical data); planets' moons left out",
        "source": f"JPL Horizons API, physical data, fetched {datetime.now(UTC):%Y-%m-%d}"}, indent=1))


def load(name: str) -> np.ndarray:
    return np.loadtxt(DATA / f"{name}.csv", delimiter=",", skiprows=2)


def jd(date: str) -> float:
    return datetime.fromisoformat(date).replace(tzinfo=UTC).timestamp() / 86400 + JD_UNIX0


def date_of(jd_tdb: float) -> str:
    return datetime.fromtimestamp((jd_tdb - JD_UNIX0) * 86400, UTC).strftime("%Y-%m-%d %H:%M")


# ---------- the orbital plane ----------

def plane_basis(ast: np.ndarray) -> np.ndarray:
    """Rows e1, e2, n: n along the mean angular momentum over the training years, e1 the ecliptic x axis in the plane."""
    train = ast[(ast[:, 0] >= jd(TRAIN_START)) & (ast[:, 0] < jd(TRAIN_STOP))]
    h = np.cross(train[:, 1:4], train[:, 4:7]).mean(axis=0)
    n = h / np.linalg.norm(h)
    e1 = np.array([1.0, 0.0, 0.0]) - n[0] * n
    e1 /= np.linalg.norm(e1)
    return np.vstack([e1, np.cross(n, e1), n])


# ---------- render ----------

def to_px(xy: np.ndarray) -> np.ndarray:
    return np.column_stack([SIZE / 2 + xy[:, 0] * PX_PER_AU, SIZE / 2 - xy[:, 1] * PX_PER_AU])


def frames_for_render() -> tuple[list, object, np.ndarray, np.ndarray]:
    from physicslens.sky.render import Drawn, Frame
    ast, earth = load("apophis"), load("earth")
    basis = plane_basis(ast)
    frames_jd = np.arange(jd(TRAIN_START), jd(TRAIN_STOP), STEP_DAYS)
    idx_a = np.searchsorted(ast[:, 0], frames_jd)
    idx_e = np.searchsorted(earth[:, 0], frames_jd)
    bodies = [Drawn("Sun", np.array([SIZE / 2, SIZE / 2]), 18, (60, 200, 255)),
              Drawn("Earth", to_px(earth[idx_e, 1:4] @ basis[:2].T), 11, (230, 150, 60)),
              Drawn("Apophis", to_px(ast[idx_a, 1:4] @ basis[:2].T), 13, (235, 235, 235))]
    frame = Frame(SIZE, FPS, [date_of(w)[:10] for w in frames_jd],
                  ["Rendered from NASA/JPL Horizons data. 1 frame = 2.5 days.",
                   "white: asteroid 99942 Apophis   blue: Earth   yellow: Sun"])
    return bodies, frame, basis, frames_jd


def render() -> Path:
    from physicslens.sky.render import render as draw
    bodies, frame, basis, frames_jd = frames_for_render()
    out = draw(bodies, frame, OUT / "apophis.mp4", {
        "start_jd": float(frames_jd[0]), "step_days": STEP_DAYS, "px_per_au": PX_PER_AU, "sun_px": [SIZE / 2, SIZE / 2],
        "basis": basis.tolist()}, meta_path=OUT / "render.json")
    print(f"{out}: {len(frames_jd)} frames, {len(frames_jd) / FPS:.1f} s")
    return out


# ---------- analyze (paid) ----------

FRAME_NOTE = ("Rendered top-down video of an orbital plane. The coordinate origin (0, 0) is the fixed yellow body at the "
              "centre of the frame. Units: wherever the numbers say m, read AU (astronomical units); wherever they say s, "
              "read day. Write any law for the white dot's acceleration in x, y (AU) and vx, vy (AU/day), with parameter "
              "units in AU and days.")


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
    run_id = f"orbit-{int(time.time())}"
    options = RunOptions(
        audience="dev", object_hint="the white dot (the asteroid)",
        known_px_per_unit=meta["px_per_au"], origin_px=tuple(meta["sun_px"]),
        # Physics time in days: one frame is step_days, the file plays at fps frames per second.
        slow_motion=1.0 / (meta["step_days"] * meta["fps"]),
        units={"length": "AU", "time": "day"}, frame_note=FRAME_NOTE,
    )
    run_dir = OUT / "run"
    result = run_pipeline(client, OUT / "apophis.mp4", run_dir, run_id, options, lambda m: print("  ", m, flush=True))
    (run_dir / "result.json").write_text(json.dumps(result, indent=1))
    print("verdict:", result["status"], "|", (result.get("equation") or {}).get("title"), "|", result["reason"])
    return run_dir / "result.json"


# ---------- predict ----------

def predict() -> dict:
    from physicslens.physics.custom import spec_for
    from physicslens.physics.fit import simulate
    from scipy.interpolate import CubicSpline
    meta = json.loads((OUT / "render.json").read_text())
    result = json.loads((OUT / "run" / "result.json").read_text())
    if result.get("status") != "accepted" or "fit_state" not in result:
        sys.exit(f"No accepted law to predict with (verdict: {result.get('status')}): {result.get('reason')}")
    fs = result["fit_state"]
    spec = spec_for(fs)
    basis = np.asarray(meta["basis"])
    ast, earth = load("apophis"), load("earth")
    cad = json.loads((DATA / "cad_2029.json").read_text())
    jd_cad, dist_cad_au = float(cad["data"][0][2]), float(cad["data"][0][4])

    # The law's own time starts at the first tracked frame; days from then on.
    jd0 = meta["start_jd"] + fs["first_frame"] * meta["step_days"]
    jd_end = jd_cad + 30
    t_coarse = np.arange(0.0, jd_end - jd0, 1.0)
    window = (jd_cad - 2 - jd0, jd_cad + 2 - jd0)
    t = np.union1d(t_coarse, np.arange(window[0], window[1], 1 / 1440))  # 1-minute steps around the flyby
    xy = simulate(spec, fs["params"], np.asarray(fs["y0"]), t, fs["ctx"])
    if xy is None:
        sys.exit("The recovered law could not be integrated that far.")
    pred3 = xy @ basis[:2]  # back to 3D heliocentric ecliptic, in the orbital plane
    pred_jd = jd0 + t

    def at(table: np.ndarray, when: np.ndarray) -> np.ndarray:
        return np.column_stack([CubicSpline(table[:, 0], table[:, i])(when) for i in (1, 2, 3)])

    # Near the flyby, JPL's 10-minute table: daily samples cannot resolve a pass that lasts hours.
    ast_f, earth_f = load("apophis_flyby"), load("earth_flyby")
    fine = (pred_jd >= ast_f[0, 0]) & (pred_jd <= ast_f[-1, 0])
    true3, earth3 = at(ast, pred_jd), at(earth, pred_jd)
    true3[fine], earth3[fine] = at(ast_f, pred_jd[fine]), at(earth_f, pred_jd[fine])
    err_km = np.linalg.norm(pred3 - true3, axis=1) * AU_KM
    ahead = pred_jd - jd(TRAIN_STOP)

    near = fine & (pred_jd > jd_cad - 2) & (pred_jd < jd_cad + 2)
    d_pred = np.linalg.norm(pred3[near] - earth3[near], axis=1)
    i = int(np.argmin(d_pred))
    ca_jd, ca_au = float(pred_jd[near][i]), float(d_pred[i])
    # JPL's own trajectory, with Horizons positions, through the same closest-approach search, as a check of the method.
    d_true = np.linalg.norm(true3[near] - earth3[near], axis=1)
    j = int(np.argmin(d_true))

    params = fs["params"]
    # The GM-like parameter is the one the agent gave AU^3/day^2 units; compared with JPL's value, never a typed one.
    units = {p["name"]: p.get("unit", "") for p in (fs.get("free_model") or {}).get("params", [])}
    gm_name = next((n for n, u in units.items() if "AU^3" in u.replace(" ", "")), None)
    sun = json.loads((DATA / "sun_gm.json").read_text())
    gm = params.get(gm_name) if gm_name else None
    out = {
        "law": spec.title, "latex": spec.latex, "params": params,
        "gm_recovered_au3_d2": gm, "gm_jpl_au3_d2": sun["gm_au3_d2"], "gm_source": sun["source"],
        "gm_error_percent": None if gm is None else (gm / sun["gm_au3_d2"] - 1) * 100,
        "training": f"{TRAIN_START} to {TRAIN_STOP}, one frame per {meta['step_days']} days",
        "jpl_close_approach": {"date_tdb": cad["data"][0][3], "distance_km": dist_cad_au * AU_KM,
                               "source": f"{cad['signature']['source']} {cad['signature']['version']}, fetched {cad.get('fetched')}"},
        "predicted_close_approach": {"date_tdb": date_of(ca_jd), "distance_km": ca_au * AU_KM,
                                     "time_error_hours": (ca_jd - jd_cad) * 24},
        "horizons_positions_same_search": {"date_tdb": date_of(float(pred_jd[near][j])), "distance_km": float(d_true[j]) * AU_KM},
        "position_error_km": {f"{d} days ahead": float(np.interp(d, ahead, err_km)) for d in (0, 30, 365, 730, 1198)},
    }
    (OUT / "prediction.json").write_text(json.dumps(out, indent=1))
    _plot(ahead[ahead >= 0], err_km[ahead >= 0], out)
    print(json.dumps(out, indent=1))
    return out


def _plot(ahead: np.ndarray, err_km: np.ndarray, out: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.semilogy(ahead, err_km, color="#2b6cb0")
    ax.set_xlabel("days after the last frame of the video (2026-01-01)")
    ax.set_ylabel("predicted vs JPL position, km")
    ax.set_title("Apophis: the law recovered from the video, extrapolated (Sun only)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "prediction_error.png")


def flyby(gm_sun: float, planets: bool) -> tuple[float, float]:
    """Hours by which the integrated 2029 closest approach misses JPL's time, and its distance (km)."""
    from physicslens.physics.nbody import Body, propagate
    from scipy.interpolate import CubicSpline
    ast = load("apophis")
    s0 = ast[np.argmin(np.abs(ast[:, 0] - jd(TRAIN_STOP)))]
    cad = json.loads((DATA / "cad_2029.json").read_text())
    jd_cad = float(cad["data"][0][2])
    bodies = []
    if planets:
        gms = json.loads((DATA / "planets_gm.json").read_text())["gm_km3_s2"]
        bodies = [Body.from_table(n, gm, load("earth" if n == "earth" else f"planet_{n}")) for n, gm in gms.items()]
    at = propagate(s0[1:7], s0[0], jd_cad + 2, gm_sun, bodies)
    earth = load("earth_flyby")
    when = np.arange(jd_cad - 1.9, jd_cad + 1.9, 1 / 1440)  # one-minute steps; JPL's 10-minute Earth table between
    d = np.linalg.norm(at(when)[:3].T - CubicSpline(earth[:, 0], earth[:, 1:4])(when), axis=1) * AU_KM
    k = int(np.argmin(d))
    return (float(when[k]) - jd_cad) * 24, float(d[k])


def planets() -> dict:
    """Where the Sun-only miss comes from: each source switched on alone, from the same 2026 start."""
    gm_jpl = json.loads((DATA / "sun_gm.json").read_text())["gm_au3_d2"]
    gm_video = json.loads((OUT / "prediction.json").read_text())["gm_recovered_au3_d2"]
    rows = []
    for label, gm, with_planets in [("Sun only, JPL's GM", gm_jpl, False), ("Sun only, GM from the video", gm_video, False),
                                    ("Sun, planets and Moon, JPL's GM", gm_jpl, True),
                                    ("Sun, planets and Moon, GM from the video", gm_video, True)]:
        hours, km = flyby(gm, with_planets)
        rows.append({"case": label, "time_error_hours": round(hours, 2), "distance_km": round(km)})
        print(f"{label:42s} {hours:+7.2f} h   {km:12,.0f} km", flush=True)
    cad = json.loads((DATA / "cad_2029.json").read_text())
    out = {"start": f"JPL state of 99942 Apophis on {TRAIN_STOP}",
           "jpl": {"date_tdb": cad["data"][0][3], "distance_km": float(cad["data"][0][4]) * AU_KM},
           "gm_sun_video_au3_d2": gm_video, "gm_sun_jpl_au3_d2": gm_jpl, "cases": rows}
    (OUT / "planets.json").write_text(json.dumps(out, indent=1))
    return out


def aftermath() -> dict:
    """How much the 2029 flyby changes Apophis's orbit: osculating elements on 2029-01-01 and 2029-08-01 from our
    engine (every planet and the Moon, from JPL's J2000 state) and from JPL's own ephemeris."""
    from physicslens.sky.elements import to_elements
    from physicslens.sky.ephem import J2000, catalogue_system, sun_gm
    from physicslens.sky.nbody import simulate
    path = DATA / "apophis_aftermath.csv"
    if not path.is_file():
        params = {"format": "json", "COMMAND": "'99942;'", "EPHEM_TYPE": "VECTORS", "CENTER": "'500@10'",
                  "START_TIME": "'2028-06-01'", "STOP_TIME": "'2030-06-01'", "STEP_SIZE": "'1 d'", "VEC_TABLE": "'2'",
                  "REF_PLANE": "'ECLIPTIC'", "OUT_UNITS": "'AU-D'", "CSV_FORMAT": "'YES'"}
        d = _get(HORIZONS, params)
        rows = [[float(f[0]), *(float(v) for v in f[2:8])] for f in
                ([v.strip() for v in ln.split(",")] for ln in d["result"].split("$$SOE")[1].split("$$EOE")[0].strip().splitlines())]
        np.savetxt(path, np.array(rows), delimiter=",", comments="", fmt="%.15g",
                   header=f"# JPL Horizons API ({d['signature']['source']} {d['signature']['version']}), fetched "
                          f"{datetime.now(UTC):%Y-%m-%d}, COMMAND=99942;, CENTER=500@10 (Sun), ECLIPTIC J2000, AU and AU/day, TDB\n"
                          "# jd_tdb,x,y,z,vx,vy,vz")
    jpl = load("apophis_aftermath")
    when = {"before": jd("2029-01-01"), "after": jd("2029-08-01")}
    massive = ["Mercury", "Venus", "Earth", "Moon", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto", "Ceres"]
    run = simulate(sun_gm(), catalogue_system([*massive, "Apophis"]), J2000, np.array(list(when.values())))
    pos, vel = run.of("Apophis")
    out: dict = {"source": "our engine from JPL's J2000 state; JPL Horizons ephemeris", "elements": {}}
    for k, (label, t) in enumerate(when.items()):
        ours = to_elements(np.concatenate([pos[k], vel[k]]), sun_gm())
        row = jpl[np.argmin(np.abs(jpl[:, 0] - t))]
        theirs = to_elements(row[1:7], sun_gm())
        out["elements"][label] = {
            "date": date_of(t)[:10],
            "ours": {"a_au": ours.a, "e": ours.e, "period_days": ours.period, "q_au": ours.q},
            "jpl": {"a_au": theirs.a, "e": theirs.e, "period_days": theirs.period, "q_au": theirs.q}}
    b, a = out["elements"]["before"], out["elements"]["after"]
    out["change"] = {"ours": {"a_au": a["ours"]["a_au"] - b["ours"]["a_au"],
                              "period_days": a["ours"]["period_days"] - b["ours"]["period_days"]},
                     "jpl": {"a_au": a["jpl"]["a_au"] - b["jpl"]["a_au"],
                             "period_days": a["jpl"]["period_days"] - b["jpl"]["period_days"]}}
    (DATA / "aftermath.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return out


def prediction(draws: int = 12) -> dict:
    """The law found from the video (fitted GM, its small velocity term, fitted starting state) carried to the 2029
    flyby alone and with the planets added as known forcing (from JPL, not from the video), against JPL's close
    approach; then the error budget, as for Halley."""
    from physicslens.sky.ephem import J2000, catalogue_system, sun_gm
    from physicslens.sky.nbody import Options, simulate
    from physicslens.sky.prediction import PROVENANCE, law_extra, lift, planet_paths, propagate, state_covariance
    from scipy.interpolate import CubicSpline
    result = json.loads((ROOT / "gallery" / "gallery-apophis" / "result.json").read_text())
    meta = json.loads((DATA / "render.json").read_text())
    fs = result["fit_state"]
    jd0, state = lift(fs, meta)
    extra = law_extra(fs, meta)
    cad = json.loads((DATA / "cad_2029.json").read_text())
    jd_cad, d_cad = float(cad["data"][0][2]), float(cad["data"][0][4]) * AU_KM
    earth = load("earth_flyby")
    earth_at = CubicSpline(earth[:, 0], earth[:, 1:4])
    fine = np.arange(jd_cad - 1.9, jd_cad + 1.9, 1 / 1440)
    jd_eval = np.concatenate([np.arange(jd0 + 10, jd_cad - 2, 10.0), fine])
    near = jd_eval >= fine[0]
    paths = planet_paths(jd0, float(jd_eval[-1]))

    def approach(gm: float, st: np.ndarray, planets: bool) -> tuple[float, float, np.ndarray]:
        pos = propagate(gm, st, jd0, jd_eval, planets, extra=extra, prescribed=paths)[near]
        d = np.linalg.norm(pos - earth_at(fine), axis=1) * AU_KM
        k = int(np.argmin(d))
        return float((fine[k] - jd_cad) * 24), float(d[k]), d

    mu = fs["params"]["mu"]
    mu_std = next(p["std"] for p in result["parameters"] if p["name"] == "mu")
    h_alone, km_alone, _ = approach(mu, state, False)
    h_added, km_added, _ = approach(mu, state, True)
    h_plus, km_plus, _ = approach(mu + mu_std, state, True)
    h_minus, km_minus, _ = approach(mu - mu_std, state, True)
    mean, cov = state_covariance(result)
    rng = np.random.default_rng(0)
    mc = [approach(d[0], lift({**fs, "y0": list(d[1:])}, meta)[1], True) for d in rng.multivariate_normal(mean, cov, draws)]
    massive = ["Mercury", "Venus", "Earth", "Moon", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto", "Ceres"]
    grid = np.concatenate([[J2000], fine])

    def jpl_start(gr: bool) -> tuple[float, float]:
        run = simulate(sun_gm(), catalogue_system([*massive, "Apophis"]), J2000, grid, Options(gr=gr, max_seconds=900))
        d = np.linalg.norm(run.of("Apophis")[0][1:] - run.of("Earth")[0][1:], axis=1) * AU_KM
        k = int(np.argmin(d))
        return float((fine[k] - jd_cad) * 24), float(d[k])

    newton, with_gr = jpl_start(False), jpl_start(True)
    rows = [
        {"case": "Discovered law alone (the video's Sun, the video's start)", "hours_from_jpl": h_alone, "km": km_alone},
        {"case": f"Discovered law with the planets ({PROVENANCE})", "hours_from_jpl": h_added, "km": km_added},
        {"case": "JPL close-approach record", "date": cad["data"][0][3], "hours_from_jpl": 0.0, "km": d_cad},
    ]
    out = {
        "law": result["equation"]["title"], "start": date_of(jd0), "rows": rows,
        "error_budget": {
            "gm_one_sigma_hours": abs(h_plus - h_minus) / 2, "gm_one_sigma_km": abs(km_plus - km_minus) / 2,
            "gm_and_state_monte_carlo_std_hours": float(np.std([m[0] for m in mc])),
            "gm_and_state_monte_carlo_std_km": float(np.std([m[1] for m in mc])), "monte_carlo_draws": draws,
            "relativity_hours": with_gr[0] - newton[0], "relativity_km": with_gr[1] - newton[1],
            "left_after_planets_and_relativity_from_jpl_state": {"hours": with_gr[0], "km": with_gr[1] - d_cad},
        },
        "notes": ("Relativity and the remainder are measured from JPL's J2000 state of Apophis with every planet and the "
                  "Moon: the remainder is mostly the Yarkovsky push of sunlight, which JPL models and we do not."),
    }
    (DATA / "prediction_table.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    _, _, d_alone = approach(mu, state, False)
    _, _, d_added = approach(mu, state, True)
    pick = np.linspace(0, len(fine) - 1, 300).astype(int)
    jpl_d = np.linalg.norm(CubicSpline(load("apophis_flyby")[:, 0], load("apophis_flyby")[:, 1:4])(fine) - earth_at(fine),
                           axis=1) * AU_KM
    chart = {"provenance": PROVENANCE, "x_label": "hours from JPL's closest approach", "y_label": "distance from Earth, km",
             "t": ((fine[pick] - jd_cad) * 24).tolist(), "alone": d_alone[pick].tolist(), "with_planets": d_added[pick].tolist(),
             "jpl": jpl_d[pick].tolist(), "rows": rows}
    (ROOT / "gallery" / "gallery-apophis" / "prediction.json").write_text(json.dumps(chart, default=float) + "\n")
    print(json.dumps(out, indent=1, default=float))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fetch", "fetch-planets", "render", "analyze", "predict", "planets", "aftermath",
                                     "prediction", "all"])
    a = ap.parse_args()
    if a.step in ("fetch", "all") and (a.step == "fetch" or not (DATA / "apophis.csv").exists()):
        fetch()
    if a.step in ("render", "all"):
        render()
    if a.step == "analyze" or (a.step == "all" and input("Run the paid analysis (SAM 3.1 + Muse Spark Open, billed through the cost guard)? [y/N] ") == "y"):
        analyze()
    if a.step in ("predict", "all"):
        predict()
    if a.step == "fetch-planets":
        fetch_planets()
    if a.step == "planets":
        planets()
    if a.step == "aftermath":
        aftermath()
    if a.step == "prediction":
        prediction()


if __name__ == "__main__":
    main()
