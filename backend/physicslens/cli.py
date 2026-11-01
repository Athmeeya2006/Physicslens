"""The command line: the same analysis as the web app, and the Solar System Lab, from a terminal.

  physicslens analyze clip.mp4 [--length 0.5] [--size 0.06] [--export mjcf,json,py,urdf,sdf] [--out DIR]
  physicslens sky list
  physicslens sky simulate SCENARIO [--mass Jupiter=10] [--years 20]
  physicslens sky discover SCENARIO [--noise-px 2]

`analyze` makes the same paid model calls as an upload (SAM 3.1 and the reasoning model), every one through the
cost guard and its budgets; set PHYSICSLENS_KILL=1 to block them all. `sky` is local math on cached JPL data: free.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any


def _analyze(a: argparse.Namespace) -> int:
    from .costguard import CostGuardClient
    from .export.mjcf import build_mjcf
    from .export.robot import sdf, urdf
    from .export.script import parameters_json, python_script
    from .pipeline import RunOptions, analyze
    from .settings import load_settings
    video = Path(a.video)
    if not video.is_file():
        print(f"No such file: {video}", file=sys.stderr)
        return 2
    out = Path(a.out or f"physicslens-{video.stem}")
    out.mkdir(parents=True, exist_ok=True)
    client = CostGuardClient(load_settings())
    result = analyze(client, video, out / "run", f"cli-{uuid.uuid4().hex[:8]}",
                     RunOptions(pendulum_length_m=a.length, object_size_m=a.size, object_hint=a.hint),
                     lambda m: print("  ", m, file=sys.stderr, flush=True))
    (out / "result.json").write_text(json.dumps(result, indent=1, default=float))
    eq = (result.get("equation") or {}).get("title")
    print(f"{result['status']}: {eq or ''}  {result['reason']}")
    for p in result.get("parameters", []):
        print(f"  {p['name']} = {p['value']:.6g} +- {p['std']:.2g} {p['unit']}")
    writers = {"json": lambda: parameters_json(result), "py": lambda: python_script(result),
               "mjcf": lambda: build_mjcf(result["fit_state"])[0], "urdf": lambda: urdf(result["fit_state"]),
               "sdf": lambda: sdf(result["fit_state"])}
    ext = {"json": "json", "py": "py", "mjcf": "xml", "urdf": "urdf", "sdf": "sdf"}
    for kind in [k for k in (a.export or "").split(",") if k]:
        if kind not in writers:
            print(f"Unknown export {kind!r} ({', '.join(writers)})", file=sys.stderr)
            continue
        if "fit_state" not in result:
            print(f"No fitted model to export as {kind}.", file=sys.stderr)
            continue
        path = out / f"physicslens.{ext[kind]}"
        path.write_text(writers[kind]())
        print(f"  wrote {path}")
    return 0 if result["status"] == "accepted" else 1


def _sky(a: argparse.Namespace) -> int:
    from .sky import scenarios as S
    from .sky_api import DiscoverRequest, SimulateRequest, _simulation, scenario_for
    if a.action == "list":
        for n in S.names():
            print(f"{n:28s} {S.load(n).title}")
        return 0
    if not a.scenario:
        print("Name a scenario (physicslens sky list).", file=sys.stderr)
        return 2
    scale: dict[str, float] = {}
    for item in a.mass or []:
        name, _, value = item.partition("=")
        scale[name] = float(value)
    req: dict[str, Any] = {"scenario": a.scenario, "mass_scale": scale, **({"years": a.years} if a.years else {})}
    try:
        if a.action == "simulate":
            sim = _simulation(scenario_for(SimulateRequest(**req)))
            keep = {k: sim[k] for k in ("target", "precession_arcsec_per_century", "energy_drift_per_century",
                                        "detectability", "warnings", "source")}
            keep["elements_end"] = sim["elements"][-1] if sim["elements"] else None
            print(json.dumps(keep, indent=1, default=float))
            return 0
        from .sky.discover import discover
        r = DiscoverRequest(**req, noise_px=a.noise_px)
        found = discover(S.run(scenario_for(r)), noise_px=r.noise_px)
        print(json.dumps({k: found.get(k) for k in ("source", "status", "reason", "law", "parameters", "perturbation",
                                                     "equation_diff")}, indent=1, default=float))
        return 0
    except Exception as e:  # noqa: BLE001  a bad scenario or override: the reason, not a traceback
        print(f"{type(e).__name__}: {getattr(e, 'detail', e)}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="physicslens", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    an = sub.add_parser("analyze", help="find and verify the equation of motion in a video (paid model calls)")
    an.add_argument("video")
    an.add_argument("--length", type=float, help="pendulum length in metres")
    an.add_argument("--size", type=float, help="the moving object's longest side in metres")
    an.add_argument("--hint", help="what to track, e.g. 'the red ball'")
    an.add_argument("--export", help="comma-separated: json, py, mjcf, urdf, sdf")
    an.add_argument("--out", help="output folder (default physicslens-<clip name>)")
    sky = sub.add_parser("sky", help="the Solar System Lab: free, local, from cached JPL data")
    sky.add_argument("action", choices=["list", "simulate", "discover"])
    sky.add_argument("scenario", nargs="?")
    sky.add_argument("--mass", action="append", help="NAME=SCALE, e.g. Jupiter=10 (repeatable)")
    sky.add_argument("--years", type=float)
    sky.add_argument("--noise-px", type=float, default=2.0)
    a = ap.parse_args(argv)
    return _analyze(a) if a.command == "analyze" else _sky(a)


if __name__ == "__main__":
    sys.exit(main())
