"""Solar-system scenarios (data/sky/scenarios/*.yaml) with PhysicsLens's own N-body engine.

  python scripts/sky_demo.py list                 the scenarios
  python scripts/sky_demo.py fetch                JPL data -> data/sky/ (free, once; same as scripts/sky_fetch.py)
  python scripts/sky_demo.py render SCENARIO      -> var/sky/SCENARIO.mp4, the video the pipeline would watch
  python scripts/sky_demo.py track SCENARIO       direct-track discovery: free, fixed candidate laws, "simulation, no pixels"
  python scripts/sky_demo.py analyze SCENARIO     the normal pipeline on the rendered video (PAID: SAM 3.1 + Muse Spark
                                                  Open; asks first)
  python scripts/sky_demo.py compare SCENARIO     the scenario against JPL (when it is real) and against its baseline
  python scripts/sky_demo.py validate             every check against JPL -> data/sky/validation_results.json (the
                                                  numbers the README quotes)
  python scripts/sky_demo.py leverrier            the hidden-planet test at two noise levels -> data/sky/leverrier.json

Two modes, kept apart on purpose. A video discovery sees pixels, like every other PhysicsLens result. A direct-track
discovery fits the simulated track with tracker-sized noise: free and instant, for the lab, and labelled as such.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.sky import scenarios as S
from physicslens.sky.discover import discover

OUT = ROOT / "var" / "sky"
def _render(name: str) -> Path:
    out = OUT / f"{name}.mp4"
    meta = S.render_scenario(S.run(S.load(name)), out)
    print(f"{out}: {meta['frames']} frames, {meta['px_per_au']:.1f} px/AU")
    return out


def _track(name: str, noise_px: float) -> dict:
    sr = S.run(S.load(name))
    perturbers = tuple(n for n in sr.run.names if n != sr.scenario.target and n in ("Jupiter", "Saturn", "Neptune",
                                                                                     "Earth"))[:1]
    result = discover(sr, noise_px=noise_px, perturbers=perturbers)
    result["scenario"] = sr.scenario.summary()
    if sr.detectability is not None:
        result["detectability"] = {"max_px": sr.detectability.max_px, "visible": sr.detectability.visible,
                                   "message": sr.detectability.message}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}_track.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({k: result[k] for k in ("source", "status", "reason")}, indent=1))
    return result


def _compare(name: str) -> dict:
    sc = S.load(name)
    extra = None
    if sc.predict_until:
        extra = np.arange(S.jd_of(sc.start) + sc.clip_days, S.jd_of(sc.predict_until), 0.05)
    sr = S.run(sc, extra)
    out: dict = {"scenario": sc.summary(), "energy_drift_per_century": sr.run.energy_drift, "warnings": sr.run.warnings}
    if sc.compare.endswith("cad_2029.json"):
        cad = json.loads((ROOT / sc.compare).read_text())
        jd_cad, d_cad = float(cad["data"][0][2]), float(cad["data"][0][4]) * 149_597_870.7
        near = (sr.run.jd > jd_cad - 2) & (sr.run.jd < jd_cad + 2)
        if "Earth" in sr.run.names:
            d = np.linalg.norm(sr.run.of(sc.target)[0][near] - sr.run.of("Earth")[0][near], axis=1) * 149_597_870.7
            k = int(np.argmin(d))
            out["flyby"] = {"hours_from_jpl": float((sr.run.jd[near][k] - jd_cad) * 24), "km": float(d[k]), "jpl_km": d_cad}
        else:
            out["flyby"] = "Earth is not in this scenario: no flyby to measure."
    if sc.compare.endswith("halley_2061.csv"):
        later = np.loadtxt(ROOT / sc.compare, delimiter=",", skiprows=2)
        jpl = float(later[np.argmin(np.linalg.norm(later[:, 1:4], axis=1)), 0])
        late = sr.run.jd > S.jd_of(sc.start) + 40 * 365.25
        r = np.linalg.norm(sr.run.of(sc.target)[0][late], axis=1)
        out["perihelion"] = {"days_from_jpl": float(sr.run.jd[late][int(np.argmin(r))] - jpl), "q_au": float(r.min())}
    if sr.detectability is not None:
        out["detectability"] = sr.detectability.message
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}_compare.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return out


def _analyze(name: str) -> None:
    import os
    os.environ.setdefault("APP_ENV", "dev")
    video = OUT / f"{name}.mp4"
    if not video.is_file():
        _render(name)
    if input("Run the paid analysis on the rendered video (SAM 3.1 + Muse Spark Open, billed through the cost guard)? [y/N] ") != "y":
        print("Skipped.")
        return
    from physicslens.costguard import CostGuardClient
    from physicslens.pipeline import RunOptions
    from physicslens.pipeline import analyze as run_pipeline
    from physicslens.settings import load_settings
    meta = json.loads(video.with_suffix(".json").read_text())
    client = CostGuardClient(load_settings())
    note = ("Rendered top-down video of an orbital plane. The coordinate origin (0, 0) is the fixed yellow body at the "
            "centre of the frame. Units: wherever the numbers say m, read AU; wherever they say s, read day.")
    run_dir = OUT / f"{name}_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = run_pipeline(client, video, run_dir, f"sky-{name}", RunOptions(
        audience="dev", object_hint="the white dot", known_px_per_unit=meta["px_per_au"],
        origin_px=tuple(meta["sun_px"]), units={"length": "AU", "time": "day"}, frame_note=note,
        # Physics time in days, as orbit_demo.py: one frame is step_days, the file plays at fps frames per second.
        slow_motion=1.0 / (meta["step_days"] * meta["fps"])), lambda m: print("  ", m, flush=True))
    (run_dir / "result.json").write_text(json.dumps(result, indent=1, default=float))
    print("verdict:", result["status"], "|", result["reason"])


def _validate() -> dict:
    import time

    from physicslens.sky import validate as V
    from physicslens.sky.ephem import sun_gm
    from physicslens.sky.nbody import Options, simulate
    started = time.time()
    truth = V.validation_positions()
    gr = simulate(sun_gm(), V.catalogue_system(V.MASSIVE), V.J2000, truth["Earth"][:, 0], Options(gr=True))
    sun_only = S.run(S.load("apophis_sun_only"))
    exact, noisy = discover(sun_only, noise_px=0.0), discover(sun_only, noise_px=2.0)
    mu = lambda r: next(p for p in r["parameters"] if p["name"] == "mu")
    out = {
        "planets_ten_years_au": V.planets_ten_years(),
        "planets_ten_years_with_relativity_au": {n: float(np.max(np.linalg.norm(gr.of(n)[0] - rows[:, 1:4], axis=1)))
                                                 for n, rows in truth.items()},
        "frames_agree_au": V.frames_agree(),
        "mercury_relativity_arcsec_per_century": V.mercury_relativity(years=100),
        "apophis_2029_from_j2000": V.apophis_2029(),
        "halley_2061_from_j2000": V.halley_2061(planets=True),
        "halley_2061_from_j2000_sun_only": V.halley_2061(planets=False),
        "performance": V.performance(),
        "direct_track_apophis_sun_only": {
            "no_noise_gm_rel_error": mu(exact)["value"] / sun_gm() - 1,
            "two_px_gm_rel_error": mu(noisy)["value"] / sun_gm() - 1, "two_px_gm_rel_std": mu(noisy)["std"] / sun_gm()},
        "detectability_px": {n: S.run(S.load(n)).detectability.max_px
                             for n in ("apophis_with_planets", "jupiter_x10", "jupiter_x100", "mercury_gr_exaggerated",
                                       "uranus_neptune_hidden", "rogue_star_flyby")},
    }
    out["seconds"] = time.time() - started
    path = ROOT / "data" / "sky" / "validation_results.json"
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out, indent=1, default=float))
    return out


def _leverrier() -> dict:
    """Uranus 1780 to 1846 with Neptune pulling but hidden: what the system finds, against Neptune's true values."""
    from physicslens.sky.ephem import masses_km3_s2
    from physicslens.sky.nbody import gm_au3_d2
    sr = S.run(S.load("uranus_neptune_hidden"))
    rows = np.isin(sr.run.jd, sr.frames_jd)
    nep = sr.run.of("Neptune")[0][rows] @ sr.basis[:2].T
    truth = {"gm_au3_d2": gm_au3_d2(masses_km3_s2()["Neptune"]), "radius_au_mean": float(np.hypot(*nep.T).mean()),
             "phase_at_start_rad": float(np.arctan2(nep[0, 1], nep[0, 0]))}
    out = {"truth": truth, "detectability_at_video_scale": sr.detectability.message, "runs": {}}
    for noise in (2.0, 0.02):
        r = discover(sr, noise_px=noise, bootstrap_reps=8)
        keep = {k: r.get(k) for k in ("status", "reason", "perturbation", "residual_spectrum", "equation_diff")}
        keep["law"] = (r.get("law") or {}).get("title")
        keep["parameters"] = r.get("parameters")
        out["runs"][f"{noise:g} px"] = keep
        print(noise, keep["law"], "|", keep["perturbation"] or (keep["equation_diff"] or {}).get("summary"), flush=True)
    (ROOT / "data" / "sky" / "leverrier.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["list", "fetch", "render", "track", "analyze", "compare", "validate", "leverrier"])
    ap.add_argument("scenario", nargs="?")
    ap.add_argument("--noise-px", type=float, default=2.0)
    a = ap.parse_args()
    if a.step == "list":
        for n in S.names():
            sc = S.load(n)
            print(f"{n:28s} {sc.title}")
        return
    if a.step == "validate":
        _validate()
        return
    if a.step == "leverrier":
        _leverrier()
        return
    if a.step == "fetch":
        import runpy
        runpy.run_path(str(ROOT / "scripts" / "sky_fetch.py"), run_name="__main__")
        return
    if not a.scenario:
        sys.exit("Name a scenario (python scripts/sky_demo.py list).")
    {"render": _render, "analyze": _analyze, "compare": _compare}.get(a.step, lambda n: _track(n, a.noise_px))(a.scenario)


if __name__ == "__main__":
    main()
