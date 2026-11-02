"""More orbits through the same video pipeline: Mercury, Mars, Ceres and comet 2P/Encke, each a rendered top-down
video of its own orbital plane, analyzed exactly like the Apophis clip, then checked against JPL.

  python scripts/orbit_more.py fetch [names]      JPL Horizons vectors -> data/orbit_more/ (free)
  python scripts/orbit_more.py render [names]     -> var/orbit_more/<name>/clip.mp4 (free)
  python scripts/orbit_more.py analyze [names]    the normal pipeline on each video (PAID: SAM 3.1 + Muse Spark Open)
  python scripts/orbit_more.py check [names]      the accepted law against JPL: GM of the Sun, and positions after the
                                                  video ends -> data/orbit_more/results.json (free)

Each orbit is drawn so the dot moves less than its own size per frame (a tracker can follow it) and the whole orbit
fits the frame. The analysis sees only pixels; JPL's positions are used to draw the video and, afterwards, to check
the law. The law lives in the orbital plane and is lifted back to 3D with the same basis to measure the error.
Encke tests the law where it is hardest: a comet on an orbit that dives from beyond Jupiter's distance to inside
Mercury's (eccentricity 0.85), with gas jets that push it slightly (no gravity law includes those).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from orbit_demo import AU_KM, FPS, FRAME_NOTE, HORIZONS, SIZE, _get, date_of, jd

DATA = ROOT / "data" / "orbit_more"
OUT = ROOT / "var" / "orbit_more"

# name: Horizons command, label, training span (what the video shows), forecast span after it, days per frame,
# pixels per AU. Frame steps keep the fastest motion under about 15 px per frame (the dot is 26 px across).
TARGETS: dict[str, dict] = {
    "mercury": {"command": "199", "label": "Mercury", "train": ("2024-01-01", "2024-07-01"), "ahead_days": 365,
                "step": 0.4, "px_per_au": 1000.0},
    "mars": {"command": "499", "label": "Mars", "train": ("2022-01-01", "2025-01-01"), "ahead_days": 730,
             "step": 3.0, "px_per_au": 280.0},
    "ceres": {"command": "1;", "label": "dwarf planet 1 Ceres", "train": ("2019-01-01", "2025-01-01"), "ahead_days": 1095,
              "step": 8.0, "px_per_au": 155.0},
    "encke": {"command": "DES=2P;CAP;NOFRAG", "label": "comet 2P/Encke", "train": ("2020-01-01", "2023-06-01"),
              "ahead_days": 730, "step": 2.0, "px_per_au": 115.0},
}


def names(args: list[str]) -> list[str]:
    return [a for a in args if a in TARGETS] or list(TARGETS)


def fetch(name: str) -> None:
    t = TARGETS[name]
    start = t["train"][0]
    stop = date_of(jd(t["train"][1]) + t["ahead_days"] + 2)[:10]
    params = {"format": "json", "COMMAND": f"'{t['command']}'", "EPHEM_TYPE": "VECTORS", "CENTER": "'500@10'",
              "START_TIME": f"'{start}'", "STOP_TIME": f"'{stop}'", "STEP_SIZE": "'1 d'" if t["step"] >= 1 else "'2 h'",
              "VEC_TABLE": "'2'", "REF_PLANE": "'ECLIPTIC'", "OUT_UNITS": "'AU-D'", "CSV_FORMAT": "'YES'"}
    d = _get(HORIZONS, params)
    text = d["result"]
    if "$$SOE" not in text:
        sys.exit(f"Horizons answered without a vector table for {name}: {text[:400]}")
    rows = []
    for line in text.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines():
        f = [v.strip() for v in line.split(",")]
        rows.append([float(f[0]), *(float(v) for v in f[2:8])])
    DATA.mkdir(parents=True, exist_ok=True)
    from datetime import UTC, datetime
    header = (f"# JPL Horizons API ({d['signature']['source']} {d['signature']['version']}), fetched "
              f"{datetime.now(UTC):%Y-%m-%d}, COMMAND={t['command']}, CENTER=500@10 (Sun), ECLIPTIC J2000, AU and AU/day, TDB\n"
              "# jd_tdb,x,y,z,vx,vy,vz")
    np.savetxt(DATA / f"{name}.csv", np.array(rows), delimiter=",", header=header, comments="", fmt="%.15g")
    print(f"{name}: {len(rows)} rows")


def load(name: str) -> np.ndarray:
    return np.loadtxt(DATA / f"{name}.csv", delimiter=",", skiprows=2)


def basis_for(name: str) -> np.ndarray:
    """Rows e1, e2, n: n along the mean angular momentum over the video, e1 the ecliptic x axis in that plane."""
    t = TARGETS[name]
    s = load(name)
    tr = s[(s[:, 0] >= jd(t["train"][0])) & (s[:, 0] < jd(t["train"][1]))]
    h = np.cross(tr[:, 1:4], tr[:, 4:7]).mean(axis=0)
    n = h / np.linalg.norm(h)
    e1 = np.array([1.0, 0.0, 0.0]) - n[0] * n
    e1 /= np.linalg.norm(e1)
    return np.vstack([e1, np.cross(n, e1), n])


def at(table: np.ndarray, when: np.ndarray) -> np.ndarray:
    from scipy.interpolate import CubicSpline
    return np.column_stack([CubicSpline(table[:, 0], table[:, i])(when) for i in (1, 2, 3)])


def render(name: str) -> None:
    from physicslens.sky.render import Drawn, Frame
    from physicslens.sky.render import render as draw
    t = TARGETS[name]
    s = load(name)
    basis = basis_for(name)
    frames_jd = np.arange(jd(t["train"][0]), jd(t["train"][1]), t["step"])
    xy = at(s, frames_jd) @ basis[:2].T
    px = np.column_stack([SIZE / 2 + xy[:, 0] * t["px_per_au"], SIZE / 2 - xy[:, 1] * t["px_per_au"]])
    if px.min() < 30 or px.max() > SIZE - 30:
        sys.exit(f"{name}: the orbit leaves the frame; lower px_per_au")
    step_px = np.linalg.norm(np.diff(px, axis=0), axis=1).max()
    if step_px > 18:
        sys.exit(f"{name}: {step_px:.0f} px per frame at the fastest; shorten the step")
    bodies = [Drawn("Sun", np.array([SIZE / 2, SIZE / 2]), 18, (60, 200, 255)),
              Drawn(t["label"], px, 13, (235, 235, 235))]
    frame = Frame(SIZE, FPS, [date_of(w)[:10] for w in frames_jd],
                  [f"Rendered from NASA/JPL Horizons data. 1 frame = {t['step']:g} days.",
                   f"white: {t['label']}   yellow: Sun"])
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    draw(bodies, frame, d / "clip.mp4", {"start_jd": float(frames_jd[0]), "step_days": t["step"],
                                          "px_per_au": t["px_per_au"], "sun_px": [SIZE / 2, SIZE / 2],
                                          "basis": basis.tolist()}, meta_path=d / "render.json")
    print(f"{name}: {len(frames_jd)} frames ({len(frames_jd) / FPS:.1f} s), fastest {step_px:.1f} px per frame")


def analyze(name: str) -> None:
    os.environ.setdefault("APP_ENV", "dev")
    from physicslens.costguard import CostGuardClient
    from physicslens.pipeline import RunOptions
    from physicslens.pipeline import analyze as run_pipeline
    from physicslens.settings import load_settings
    t = TARGETS[name]
    d = OUT / name
    meta = json.loads((d / "render.json").read_text())
    settings = load_settings()
    if settings.mock_models:
        sys.exit("analyze needs MOCK_MODELS=0: it calls SAM 3.1 and Muse Spark Open for real.")
    client = CostGuardClient(settings)
    options = RunOptions(
        audience="dev", object_hint=f"the white dot ({t['label']})",
        known_px_per_unit=meta["px_per_au"], origin_px=tuple(meta["sun_px"]),
        slow_motion=1.0 / (meta["step_days"] * meta["fps"]),
        units={"length": "AU", "time": "day"}, frame_note=FRAME_NOTE,
    )
    result = run_pipeline(client, d / "clip.mp4", d / "run", f"orbit-{name}-{int(time.time())}", options,
                          lambda m: print("  ", m, flush=True))
    (d / "run" / "result.json").write_text(json.dumps(result, indent=1, default=str))
    print(name, "verdict:", result["status"], "|", (result.get("equation") or {}).get("title"), "|", result["reason"])


def check(name: str) -> dict:
    from physicslens.physics.custom import spec_for
    from physicslens.physics.fit import simulate
    t = TARGETS[name]
    d = OUT / name
    meta = json.loads((d / "render.json").read_text())
    result = json.loads((d / "run" / "result.json").read_text())
    out: dict = {"object": t["label"], "status": result.get("status"), "reason": result.get("reason"),
                 "training": f"{t['train'][0]} to {t['train'][1]}, one frame per {t['step']:g} days",
                 "frames": result.get("clip", {}).get("frames"),
                 "candidates": [{"title": c.get("title"), "verdict": c.get("verdict")} for c in result.get("candidates", [])][:6]}
    sun = json.loads((ROOT / "data" / "orbit" / "sun_gm.json").read_text())
    out["gm_jpl_au3_d2"] = sun["gm_au3_d2"]
    if result.get("status") != "accepted" or "fit_state" not in result:
        return out
    fs = result["fit_state"]
    spec = spec_for(fs)
    params = fs["params"]
    units = {p["name"]: p.get("unit", "") for p in (fs.get("free_model") or {}).get("params", [])}
    gm_name = next((n for n, u in units.items() if "AU^3" in u.replace(" ", "")), None)
    gm = params.get(gm_name) if gm_name else None
    p = next((q for q in result.get("parameters", []) if q["name"] == gm_name), None)
    out |= {"law": spec.title, "latex": spec.latex, "params": params, "gm_recovered_au3_d2": gm,
            "gm_sigma_au3_d2": p["std"] if p else None, "gm_ci95_au3_d2": p["ci95"] if p else None,
            "jpl_inside_95": bool(p and p["ci95"][0] <= sun["gm_au3_d2"] <= p["ci95"][1]),
            "gm_error_percent": None if gm is None else (gm / sun["gm_au3_d2"] - 1) * 100,
            "reasoning_calls": [c.get("model") for c in result.get("llm_calls", [])]}
    s = load(name)
    basis = np.asarray(meta["basis"])
    jd0 = meta["start_jd"] + fs["first_frame"] * meta["step_days"]
    jd_end_train = jd(t["train"][1])
    tt = np.arange(0.0, jd_end_train + t["ahead_days"] - jd0, 1.0)
    xy = simulate(spec, params, np.asarray(fs["y0"]), tt, fs["ctx"])
    if xy is None:
        out["forecast"] = "the law could not be integrated that far"
        return out
    when = jd0 + tt
    err_km = np.linalg.norm(xy @ basis[:2] - at(s, when), axis=1) * AU_KM
    r_km = np.linalg.norm(at(s, when), axis=1) * AU_KM
    ahead = when - jd_end_train
    marks = [0, 90, 365, t["ahead_days"]]
    out["position_error_after_video"] = {f"{m} days": {"km": float(np.interp(m, ahead, err_km)),
                                                       "of_distance_from_sun": float(np.interp(m, ahead, err_km / r_km))}
                                         for m in marks if m <= ahead[-1]}
    return out


def main() -> None:
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "check":
        res = {}
        p = DATA / "results.json"
        if p.is_file():
            res = json.loads(p.read_text())
        for n in names(rest):
            res[n] = check(n)
            print(json.dumps(res[n], default=float)[:900])
        p.write_text(json.dumps({"generated": "scripts/orbit_more.py check", **{k: v for k, v in res.items() if k != "generated"}},
                                indent=1, default=float))
        return
    for n in names(rest):
        {"fetch": fetch, "render": render, "analyze": analyze}[cmd](n)


if __name__ == "__main__":
    main()
