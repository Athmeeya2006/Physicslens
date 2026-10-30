"""Halley's comet: the inverse-square law from a video of its 1986 passage, then its return in 2061.

  python scripts/halley_demo.py fetch      NASA/JPL data -> data/halley/ (free public APIs)
  python scripts/halley_demo.py render     -> var/halley/halley.mp4, the video PhysicsLens watches
  python scripts/halley_demo.py analyze    the normal PhysicsLens pipeline on that video (PAID: SAM 3.1 + Muse Spark Open)
  python scripts/halley_demo.py predict    integrate the recovered law to the next perihelion; compare with JPL

Halley's orbit is a hard test for a law found from pixels: it is a long ellipse (eccentricity 0.967), so near the Sun
the pull is thousands of times stronger than at its far end, and an inverse-square law that is even slightly wrong in
its exponent bends the passage visibly. The video shows the 1986 passage, where the orbit turns hardest: a comet that
moves 60 times faster at perihelion than at aphelion cannot be followed by a tracker over a whole orbit at one frame
rate and scale, but near the Sun it can.

Data: positions of 1P/Halley (JPL small-body record 90000030, the solution fitted to observations through the 1986
apparition) and Earth from JPL Horizons, heliocentric, ecliptic J2000, AU and days. They draw the video and, afterwards,
check the answer. The analysis sees pixels: a dot moving around a fixed point. The prediction integrates the accepted
law alone (Sun only: no planets, no outgassing), so its 2061 date shows how much of Halley's return the Sun explains.
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

DATA = ROOT / "data" / "halley"
OUT = ROOT / "var" / "halley"
HORIZONS = "https://ssd.jpl.nasa.gov/api/horizons.api"
BODIES = {"halley": "90000030;", "earth": "399"}
SPANS = {"": ("1982-01-01", "1990-01-01"), "_2061": ("2059-01-01", "2063-12-31")}
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


def render() -> Path:
    import cv2
    comet, earth = load("halley"), load("earth")
    basis = plane_basis(comet)
    step = (jd(TRAIN_STOP) - jd(TRAIN_START)) / MAX_FRAMES
    frames_jd = jd(TRAIN_START) + step * np.arange(MAX_FRAMES)
    interp = lambda tab: np.column_stack([np.interp(frames_jd, tab[:, 0], tab[:, i]) for i in (1, 2, 3)])  # noqa: E731
    c_xy = interp(comet) @ basis[:2].T
    e_xy = interp(earth) @ basis[:2].T
    scale, sun_px = layout(c_xy)
    to_px = lambda xy: np.column_stack([sun_px[0] + xy[:, 0] * scale, sun_px[1] - xy[:, 1] * scale])  # noqa: E731
    c_px, e_px = to_px(c_xy), to_px(e_xy)
    speed = np.linalg.norm(np.diff(c_px, axis=0), axis=1)
    print(f"scale {scale:.1f} px/AU, one frame = {step:.2f} days, fastest move {speed.max():.1f} px/frame")
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "halley_raw.mp4"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (SIZE, SIZE))
    sub = 8  # draw at 8x and shrink: smooth, sub-pixel-accurate discs
    for k, when in enumerate(frames_jd):
        big = np.zeros((SIZE * sub, SIZE * sub, 3), np.uint8)
        big[:] = (18, 14, 10)
        cv2.circle(big, tuple(int(v * sub) for v in sun_px), 18 * sub, (60, 200, 255), -1, cv2.LINE_AA)  # Sun
        cv2.circle(big, tuple(int(v * sub) for v in e_px[k]), 10 * sub, (230, 150, 60), -1, cv2.LINE_AA)  # Earth
        cv2.circle(big, tuple(int(v * sub) for v in c_px[k]), 13 * sub, (235, 235, 235), -1, cv2.LINE_AA)  # Halley
        img = cv2.resize(big, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        cv2.putText(img, date_of(when), (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(img, f"Rendered from NASA/JPL Horizons data. 1 frame = {step:.2f} days.", (24, SIZE - 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(img, "white: comet 1P/Halley   blue: Earth   yellow: Sun", (24, SIZE - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (150, 150, 150), 1, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    out = OUT / "halley.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "16", "-threads", "2", "-movflags", "+faststart", str(out)], check=True)
    raw.unlink()
    (OUT / "render.json").write_text(json.dumps({
        "start_jd": float(frames_jd[0]), "step_days": step, "frames": len(frames_jd), "fps": FPS,
        "px_per_au": scale, "sun_px": sun_px.tolist(), "basis": basis.tolist()}, indent=1))
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
    sol = solve_ivp(lambda _t, s: spec.rhs(s, fs["params"], fs["ctx"]), (0.0, span), np.asarray(fs["y0"]),
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fetch", "render", "analyze", "predict"])
    a = ap.parse_args()
    {"fetch": fetch, "render": render, "analyze": analyze, "predict": predict}[a.step]()


if __name__ == "__main__":
    main()
