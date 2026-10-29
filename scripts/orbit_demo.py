"""Orbit mode: rediscover Newton's law of gravity from a video of an asteroid, then predict where it will be.

  python scripts/orbit_demo.py fetch      NASA/JPL data -> data/orbit/ (free public APIs; cached for offline use)
  python scripts/orbit_demo.py render     -> var/orbit/apophis.mp4, the video PhysicsLens watches
  python scripts/orbit_demo.py analyze    the normal PhysicsLens pipeline on that video (PAID: SAM 3.1 + Muse Spark Open)
  python scripts/orbit_demo.py predict    integrate the recovered law to April 2029; compare with JPL; plots + report
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
"""

from __future__ import annotations

import argparse
import json
import subprocess
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


def render() -> Path:
    import cv2
    ast, earth = load("apophis"), load("earth")
    basis = plane_basis(ast)
    frames_jd = np.arange(jd(TRAIN_START), jd(TRAIN_STOP), STEP_DAYS)
    idx_a = np.searchsorted(ast[:, 0], frames_jd)
    idx_e = np.searchsorted(earth[:, 0], frames_jd)
    a_px = to_px(ast[idx_a, 1:4] @ basis[:2].T)
    e_px = to_px(earth[idx_e, 1:4] @ basis[:2].T)
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "apophis_raw.mp4"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (SIZE, SIZE))
    sub = 8  # draw at 8x and shrink: smooth, sub-pixel-accurate discs
    for k, when in enumerate(frames_jd):
        big = np.zeros((SIZE * sub, SIZE * sub, 3), np.uint8)
        big[:] = (18, 14, 10)
        c = (SIZE * sub // 2, SIZE * sub // 2)
        cv2.circle(big, c, 18 * sub, (60, 200, 255), -1, cv2.LINE_AA)  # Sun
        cv2.circle(big, tuple(int(v * sub) for v in e_px[k]), 11 * sub, (230, 150, 60), -1, cv2.LINE_AA)  # Earth
        cv2.circle(big, tuple(int(v * sub) for v in a_px[k]), 13 * sub, (235, 235, 235), -1, cv2.LINE_AA)  # Apophis
        img = cv2.resize(big, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        cv2.putText(img, date_of(when)[:10], (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(img, "Rendered from NASA/JPL Horizons data. 1 frame = 2.5 days.", (24, SIZE - 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(img, "white: asteroid 99942 Apophis   blue: Earth   yellow: Sun", (24, SIZE - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (150, 150, 150), 1, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    out = OUT / "apophis.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "16", "-movflags", "+faststart", str(out)], check=True)
    raw.unlink()
    (OUT / "render.json").write_text(json.dumps({
        "start_jd": float(frames_jd[0]), "step_days": STEP_DAYS, "frames": len(frames_jd), "fps": FPS,
        "px_per_au": PX_PER_AU, "sun_px": [SIZE / 2, SIZE / 2], "basis": basis.tolist()}, indent=1))
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fetch", "render", "analyze", "predict", "all"])
    a = ap.parse_args()
    if a.step in ("fetch", "all") and (a.step == "fetch" or not (DATA / "apophis.csv").exists()):
        fetch()
    if a.step in ("render", "all"):
        render()
    if a.step == "analyze" or (a.step == "all" and input("Run the paid analysis (SAM 3.1 + Muse Spark Open, about $0.30)? [y/N] ") == "y"):
        analyze()
    if a.step in ("predict", "all"):
        predict()


if __name__ == "__main__":
    main()
