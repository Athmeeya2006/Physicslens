"""Weighing Jupiter from video: each Galilean moon, filmed as a rendered top-down video, gives Jupiter's GM on its own.

  python scripts/jupiter_demo.py fetch                  JPL Horizons -> data/jupiter/ (free; cached for offline use)
  python scripts/jupiter_demo.py render                 -> var/jupiter/<moon>.mp4
  python scripts/jupiter_demo.py analyze [MOON ...]     the normal pipeline on each video (PAID: SAM 3.1 + Muse Spark Open)
  python scripts/jupiter_demo.py report                 GM per moon, combined, against JPL; Kepler's third law across moons

Each video shows 1.25 orbits of one moon around Jupiter, viewed top-down in that moon's orbital plane, 360 frames.
Lengths are in Jupiter radii (R_J, Jupiter's equatorial radius from Horizons) and time in hours, so GM is about
4.5 R_J^3/h^2 and fits the expression language's bounds. As in orbit mode, JPL's positions only draw the video and,
afterwards, check the answers; the pipeline sees pixels.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))
from orbit_demo import HORIZONS, _get

DATA = ROOT / "data" / "jupiter"
OUT = ROOT / "var" / "jupiter"
MOONS = {"io": ("501", 42.46), "europa": ("502", 85.23), "ganymede": ("503", 171.71), "callisto": ("504", 400.54)}
START = "2026-01-01"
FRAMES = 360  # 1.25 orbits, 288 frames each: about 9 px per frame on a 430 px orbit
ORBITS = 1.25  # at 2 orbits (15 px per frame) SAM lost a plain dot after 5 frames on a real run
SIZE = 1080
ORBIT_PX = 430
FPS = 30


def step_minutes(period_h: float) -> int:
    return max(1, round(period_h * ORBITS * 60 / FRAMES))


def fetch() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    phys = _get(HORIZONS, {"format": "json", "COMMAND": "'599'", "OBJ_DATA": "'YES'", "MAKE_EPHEM": "'NO'"})["result"]
    gm = float(next(ln for ln in phys.splitlines() if "GM (km^3/s^2)" in ln).split("=")[1].split()[0])
    radius = float(next(ln for ln in phys.splitlines() if "Equat. radius" in ln).split("=")[1].split("+")[0])
    (DATA / "jupiter.json").write_text(json.dumps({
        "gm_km3_s2": gm, "equatorial_radius_km": radius,
        "source": f"JPL Horizons API, Jupiter (599) physical data, fetched {datetime.now(UTC):%Y-%m-%d}"}, indent=1))
    for moon, (command, period_h) in MOONS.items():
        stop = datetime.fromisoformat(START).timestamp() + period_h * ORBITS * 3600 * 1.02
        params = {"format": "json", "COMMAND": f"'{command}'", "EPHEM_TYPE": "VECTORS", "CENTER": "'500@599'",
                  "START_TIME": f"'{START}'", "STOP_TIME": f"'{datetime.fromtimestamp(stop, UTC):%Y-%m-%d %H:%M}'",
                  "STEP_SIZE": f"'{step_minutes(period_h)} m'", "VEC_TABLE": "'2'", "REF_PLANE": "'ECLIPTIC'",
                  "OUT_UNITS": "'KM-S'", "CSV_FORMAT": "'YES'"}
        d = _get(HORIZONS, params)
        rows = [[float(f[0]), *(float(v) for v in f[2:8])] for f in
                (ln.split(",") for ln in d["result"].split("$$SOE")[1].split("$$EOE")[0].strip().splitlines())]
        header = (f"# JPL Horizons API ({d['signature']['source']} {d['signature']['version']}), fetched "
                  f"{datetime.now(UTC):%Y-%m-%d}, COMMAND={command}, CENTER=500@599 (Jupiter), ECLIPTIC J2000, km and km/s, TDB\n"
                  "# jd_tdb,x,y,z,vx,vy,vz")
        np.savetxt(DATA / f"{moon}.csv", np.array(rows), delimiter=",", header=header, comments="", fmt="%.15g")
        print(f"{moon}: {len(rows)} rows, every {step_minutes(period_h)} min")
        time.sleep(1)


def load(moon: str) -> np.ndarray:
    return np.loadtxt(DATA / f"{moon}.csv", delimiter=",", skiprows=2)


def plane_basis(rv: np.ndarray) -> np.ndarray:
    h = np.cross(rv[:, 1:4], rv[:, 4:7]).mean(axis=0)
    n = h / np.linalg.norm(h)
    e1 = np.array([1.0, 0.0, 0.0]) - n[0] * n
    e1 /= np.linalg.norm(e1)
    return np.vstack([e1, np.cross(n, e1), n])


def render(moon: str) -> None:
    import cv2
    rj = json.loads((DATA / "jupiter.json").read_text())["equatorial_radius_km"]
    rv = load(moon)[:FRAMES]
    basis = plane_basis(rv)
    xy = rv[:, 1:4] @ basis[:2].T / rj  # Jupiter radii, in the moon's orbital plane
    px_per_rj = ORBIT_PX / float(np.max(np.hypot(*xy.T)))
    px = np.column_stack([SIZE / 2 + xy[:, 0] * px_per_rj, SIZE / 2 - xy[:, 1] * px_per_rj])
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / f"{moon}_raw.mp4"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (SIZE, SIZE))
    sub = 8
    step_h = (rv[1, 0] - rv[0, 0]) * 24
    for k in range(len(rv)):
        big = np.zeros((SIZE * sub, SIZE * sub, 3), np.uint8)
        big[:] = (18, 14, 10)
        c = (SIZE * sub // 2, SIZE * sub // 2)
        cv2.circle(big, c, max(int(px_per_rj * sub), 10 * sub), (120, 170, 215), -1, cv2.LINE_AA)  # Jupiter, to scale
        cv2.circle(big, tuple(int(v * sub) for v in px[k]), 16 * sub, (235, 235, 235), -1, cv2.LINE_AA)  # the moon (32 px)
        img = cv2.resize(big, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        cv2.putText(img, f"{moon.capitalize()}   t = {k * step_h:6.1f} h", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(img, f"Rendered from NASA/JPL Horizons data. 1 frame = {step_h * 60:.0f} min.", (24, SIZE - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (150, 150, 150), 1, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    out = OUT / f"{moon}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                    "-movflags", "+faststart", str(out)], check=True)
    raw.unlink()
    (OUT / f"{moon}.json").write_text(json.dumps({"step_hours": step_h, "frames": len(rv), "fps": FPS,
                                                  "px_per_rj": px_per_rj, "jupiter_px": [SIZE / 2, SIZE / 2],
                                                  "basis": basis.tolist()}, indent=1))
    print(f"{out}: {len(rv)} frames, {step_h * 60:.0f} min per frame, {px_per_rj:.1f} px per R_J")


FRAME_NOTE = ("Rendered top-down video of an orbital plane. The coordinate origin (0, 0) is the fixed body at the centre "
              "of the frame. Units: wherever the numbers say m, read R_J (Jupiter radii); wherever they say s, read hour. "
              "Write any law for the white dot's acceleration in x, y (R_J) and vx, vy (R_J/hour), with parameter units "
              "in R_J and hours.")


def analyze(moon: str) -> None:
    import os
    os.environ.setdefault("APP_ENV", "dev")
    from physicslens.costguard import CostGuardClient
    from physicslens.pipeline import RunOptions
    from physicslens.pipeline import analyze as run_pipeline
    from physicslens.settings import load_settings
    meta = json.loads((OUT / f"{moon}.json").read_text())
    settings = load_settings()
    if settings.mock_models:
        sys.exit("analyze needs MOCK_MODELS=0.")
    options = RunOptions(audience="dev", object_hint="the white dot (the moon)", known_px_per_unit=meta["px_per_rj"],
                         origin_px=tuple(meta["jupiter_px"]), slow_motion=1.0 / (meta["step_hours"] * meta["fps"]),
                         units={"length": "R_J", "time": "hour"}, frame_note=FRAME_NOTE)
    run_dir = OUT / f"run_{moon}"
    result = run_pipeline(CostGuardClient(settings), OUT / f"{moon}.mp4", run_dir, f"jupiter-{moon}-{int(time.time())}",
                          options, lambda m: print(f"  {moon}:", m, flush=True))
    (run_dir / "result.json").write_text(json.dumps(result, indent=1))
    print(f"{moon}: {result['status']} | {(result.get('equation') or {}).get('title')} | {result['reason']}")


def _gm_param(result: dict) -> tuple[float, float] | None:
    """The parameter the agent gave cubic-length-per-squared-time units: its GM, whatever it named it."""
    for p in result.get("parameters", []):
        unit = (p.get("unit") or "").replace(" ", "")
        if "^3" in unit and p.get("value", 0) > 0:
            return float(p["value"]), float(p.get("std") or 0.0)
    return None


def measured_orbit(result: dict) -> dict | None:
    """Period from the angle swept around the centre, distance from the mean radius: both from the track alone."""
    fs = result.get("fit_state")
    pts = result.get("real_track_px")
    if not fs or not pts or len(pts) < 20:
        return None
    f, px, py = (np.asarray(v, float) for v in zip(*pts, strict=True))
    ox, oy = fs["origin_px"]
    x, y = (px - ox) / fs["px_per_m"], -(py - oy) / fs["px_per_m"]
    t = (f - f[0]) / fs["fps"]
    angle = np.unwrap(np.arctan2(y, x))
    rate, intercept = np.polyfit(t, angle, 1)
    resid = angle - (rate * t + intercept)
    rate_std = float(np.std(resid) / (np.std(t) * math.sqrt(len(t))))
    period = 2 * math.pi / abs(rate)
    # A tracked centroid can sit about a pixel off the drawn centre (Europa's fit put the attractor 1.2 px away), so
    # the distance carries one pixel of systematic uncertainty, not the far smaller scatter of the mean.
    return {"period_h": period, "period_std_h": period * rate_std / abs(rate),
            "mean_distance_rj": float(np.mean(np.hypot(x, y))), "distance_std_rj": 1.0 / fs["px_per_m"]}


def report() -> dict:
    from physicslens.physics.scaling import LAWS, assess_scaling
    jup = json.loads((DATA / "jupiter.json").read_text())
    rj = jup["equatorial_radius_km"]
    gm_true = jup["gm_km3_s2"] * 3600**2 / rj**3  # R_J^3 / h^2
    rows, x, y, s = [], [], [], []
    for moon in MOONS:
        f = OUT / f"run_{moon}" / "result.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        gm = _gm_param(r) if r.get("status") == "accepted" else None
        row = {"moon": moon, "verdict": r["status"], "law": (r.get("equation") or {}).get("title"), "reason": r["reason"]}
        if gm:
            row.update(gm=gm[0], gm_std=gm[1], gm_error_percent=(gm[0] / gm_true - 1) * 100)
        track = measured_orbit(r)
        if track:
            # Straight from the tracked positions, not from the fitted law: Kepler's third law is then a real test.
            row.update(track)
            x.append(track["mean_distance_rj"])
            y.append(track["period_h"])
            # The law is period against distance, so the distance's uncertainty enters as 1.5 * relative error.
            rel_a = track["distance_std_rj"] / track["mean_distance_rj"]
            s.append(math.hypot(track["period_std_h"], 1.5 * rel_a * track["period_h"]))
        rows.append(row)
    good = [r for r in rows if "gm" in r]
    combined = None
    if good:
        w = np.array([1 / max(r["gm_std"], 1e-9) ** 2 for r in good])
        v = float(np.sum(w * [r["gm"] for r in good]) / w.sum())
        combined = {"gm": v, "gm_std": float(1 / math.sqrt(w.sum())), "gm_error_percent": (v / gm_true - 1) * 100}
    kepler = None
    if len(x) >= 3:
        verdict = assess_scaling(x, y, s)
        w = verdict.winner
        kepler = {"status": verdict.status, "reason": verdict.reason,
                  "law": w and LAWS[w.law].title, "params": w and w.params}
        if w is not None and w.law == "power":
            # period = A * distance^b: b should be 1.5 (Kepler), and A = 2 pi / sqrt(GM) weighs Jupiter a third way,
            # from measured periods and distances only.
            a_coef, b = w.params
            b_std = math.sqrt(max(w.cov[1, 1], 0.0))
            gm_k = (2 * math.pi / a_coef) ** 2
            kepler.update(exponent=b, exponent_std=b_std, gm_from_kepler=gm_k,
                          gm_from_kepler_error_percent=(gm_k / gm_true - 1) * 100)
    out = {"gm_jpl_rj3_h2": gm_true, "gm_source": jup["source"], "moons": rows, "combined": combined, "kepler_third_law": kepler,
           "units": "R_J = Jupiter equatorial radius (from Horizons); h = hour"}
    (OUT / "report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fetch", "render", "analyze", "report"])
    ap.add_argument("moons", nargs="*", default=list(MOONS))
    a = ap.parse_args()
    if a.step == "fetch":
        fetch()
    for moon in a.moons:
        if a.step == "render":
            render(moon)
        elif a.step == "analyze":
            analyze(moon)
    if a.step == "report":
        report()


if __name__ == "__main__":
    main()
