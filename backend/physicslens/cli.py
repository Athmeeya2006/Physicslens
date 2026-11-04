"""The command line: the same analysis as the web app, and the Solar System Lab, from a terminal.

  physicslens analyze clip.mp4 [--length 0.5] [--size 0.06] [--export mjcf,json,py,urdf,sdf] [--out DIR]
                              [--lens lens.json] [--mock]
  physicslens lens calibrate board.mp4 [--out lens.json]
  physicslens sky list
  physicslens sky simulate SCENARIO [--mass Jupiter=10] [--years 20]
  physicslens sky discover SCENARIO [--noise-px 2]

`analyze` makes the same paid model calls as an upload (SAM 3.1 and the reasoning model), every one through the
cost guard and its budgets; set PHYSICSLENS_KILL=1 to block them all. --mock replaces every model call with a canned
test answer (a dry run of the plumbing, never a real result; refused when APP_ENV is demo or prod). `sky` and `lens`
are local math: free.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any


def _analyze(a: argparse.Namespace) -> int:
    import os
    if a.mock:
        os.environ["MOCK_MODELS"] = "1"  # before settings load; settings refuse it outside dev and test
    from .agent.llm import LLMError
    from .costguard import CostGuardClient, CostGuardError
    from .export.mjcf import build_mjcf
    from .export.robot import sdf, urdf
    from .export.script import parameters_json, python_script
    from .perception.media import MediaError
    from .pipeline import AnalysisError, RunOptions, analyze
    from .settings import load_settings
    video = Path(a.video)
    if not video.is_file():
        print(f"No such file: {video}", file=sys.stderr)
        return 2
    out = Path(a.out or f"physicslens-{video.stem}")
    out.mkdir(parents=True, exist_ok=True)
    lens = None
    if a.lens:
        from .perception.lens import Lens, LensError
        try:
            lens = Lens.from_dict(json.loads(Path(a.lens).read_text())).as_dict()
        except (OSError, ValueError, LensError) as e:
            print(f"Lens calibration {a.lens}: {e}", file=sys.stderr)
            return 2
    client = CostGuardClient(load_settings())
    try:
        result = analyze(client, video, out / "run", f"cli-{uuid.uuid4().hex[:8]}",
                         RunOptions(pendulum_length_m=a.length, object_size_m=a.size, object_hint=a.hint, lens=lens),
                         lambda m: print("  ", m, file=sys.stderr, flush=True))
    except (AnalysisError, LLMError, MediaError, CostGuardError) as e:
        # The same stops the web runner reports: said plainly, with a failing exit code, never a traceback.
        print(f"stopped: {e}", file=sys.stderr)
        return 1
    (out / "result.json").write_text(json.dumps(result, indent=1, default=float))
    eq = (result.get("equation") or {}).get("title")
    print(f"{result['status']}: {eq or ''}  {result['reason']}")
    for p in result.get("parameters", []):
        print(f"  {p['name']} = {p['value']:.6g} +- {p['std']:.2g} {p['unit']}")
    for p in result.get("agreed_parameters", []):
        print(f"  {p['name']} = {p['value']:.6g} +- {p['std']:.2g} {p['unit']} (every tied law agrees)")
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


def _lens(a: argparse.Namespace) -> int:
    from .perception.lens import LensError, calibrate_video
    try:
        lens = calibrate_video(Path(a.video))
    except LensError as e:
        print(f"Calibration failed: {e}", file=sys.stderr)
        return 1
    Path(a.out).write_text(json.dumps(lens.as_dict(), indent=1))
    print(f"focal {(lens.fx + lens.fy) / 2:.1f} px, k1 {lens.dist[0]:.4f}, {lens.views} views, "
          f"{lens.rms_px:.2f} px error: wrote {a.out}")
    return 0


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


def _robot_type(dataset: str) -> str:
    """The robot a public LeRobot recording was made on, from its meta/info.json on the Hugging Face hub."""
    import json
    import urllib.request
    with urllib.request.urlopen(f"https://huggingface.co/datasets/{dataset}/resolve/main/meta/info.json", timeout=60) as r:
        return str(json.loads(r.read()).get("robot_type") or "")


def _aloha(a: argparse.Namespace, root: Path) -> int:
    """Calibrate an ALOHA's twelve arm joints from a public recording: the same identification as the ALOHA case."""
    import json
    import os
    import subprocess
    py = sys.executable
    env = dict(os.environ, PL_ALOHA_DATASET=a.dataset, PYTHONPATH=str(root / "backend"))
    for step in (["fetch"], []):
        subprocess.run([py, str(root / "scripts" / "aloha_demo.py"), *step], check=True, cwd=root, env=env, stdout=subprocess.DEVNULL)
    out = root / "data" / "robot" / "aloha" / a.dataset.replace("/", "__")
    r = json.loads((out / "result.json").read_text())
    print(f"{a.dataset}: {r['data']['episodes']} episodes, {len(r['data']['test_episodes'])} held out for the check")
    for name, j in r["joints"].items():
        p = {k: v["value"] for k, v in j["params"].items()}
        print(f"  {name:20s} {j['title']}: delay {p['d'] * 1000:.0f} ms" + (f", deadband {p['db']:.2f} deg" if "db" in p else ""))
    h = r["heldout"]
    print(f"held-out error over the twelve arm joints: {h['identified']['arm_total']['rms_deg']:.2f} deg corrected, "
          f"{h['default']['arm_total']['rms_deg']:.2f} deg for the published model")
    rel = out.relative_to(root)
    print(f"files: {rel / 'result.json'} (every fitted value with its interval), {rel / 'aloha_identified.xml'} (the MuJoCo model)")
    return 0


def _robot(a: argparse.Namespace) -> int:
    """Calibrate one arm from its own public LeRobot recordings: an SO-100 (the robot arm case's identification) or
    an ALOHA (the ALOHA case's), chosen by the recording's robot type."""
    import json
    import os
    import subprocess
    root = Path(__file__).resolve().parents[2]
    if "aloha" in _robot_type(a.dataset).lower():
        return _aloha(a, root)
    py = sys.executable
    if not (root / "var" / "robot" / "raw" / "menagerie" / "so_arm100.xml").exists():
        subprocess.run([py, str(root / "scripts" / "fetch_robot.py")], check=True, cwd=root)
    subprocess.run([py, str(root / "scripts" / "fetch_robot_fleet.py"), a.dataset], check=True, cwd=root)
    slug = a.dataset.replace("/", "__")
    env = dict(os.environ, PL_ROBOT_FLEET=slug, PYTHONPATH=str(root / "backend"))
    subprocess.run([py, str(root / "scripts" / "robot_demo.py")], check=True, cwd=root, env=env, stdout=subprocess.DEVNULL)
    out = root / "data" / "robot" / "fleet" / slug
    r = json.loads((out / "result.json").read_text())
    print(f"{a.dataset}: {r['data']['episodes']} episodes, {len(r['data']['test_episodes'])} held out for the check")
    for name, j in r["joints"].items():
        p = {k: v["value"] for k, v in j["params"].items()}
        bits = [f"delay {p['d'] * 1000:.0f} ms" if "d" in p else "", f"lag {p['tau'] * 1000:.0f} ms" if "tau" in p else "",
                f"deadband {p['db']:.2f} deg" if "db" in p else "", f"wn {p['wn']:.1f} rad/s" if "wn" in p else ""]
        print(f"  {name:13s} {j['title']}: " + ", ".join(b for b in bits if b))
    servo = r["heldout"]["identified_servo_laws"]["arm_total"]
    print(f"held-out error of this arm's laws: {servo['rms_deg']:.2f} deg rms over the five arm joints "
          f"(95%: {servo['ci95'][0]:.2f} to {servo['ci95'][1]:.2f})")
    rel = out.relative_to(root)
    print(f"files: {rel / 'result.json'} (every fitted value with its interval), {rel / 'so_arm100_identified.xml'} (the MuJoCo model)")
    return 0


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
    an.add_argument("--lens", help="a lens calibration JSON from 'physicslens lens calibrate'")
    an.add_argument("--mock", action="store_true", help="canned model answers: test the plumbing, spend nothing")
    lens = sub.add_parser("lens", help="measure a phone's lens from a video of the printed board: free, local")
    lens.add_argument("action", choices=["calibrate"])
    lens.add_argument("video")
    lens.add_argument("--out", default="lens.json")
    sky = sub.add_parser("sky", help="the Solar System Lab: free, local, from cached JPL data")
    sky.add_argument("action", choices=["list", "simulate", "discover"])
    sky.add_argument("scenario", nargs="?")
    sky.add_argument("--mass", action="append", help="NAME=SCALE, e.g. Jupiter=10 (repeatable)")
    sky.add_argument("--years", type=float)
    sky.add_argument("--noise-px", type=float, default=2.0)
    rb = sub.add_parser("robot", help="calibrate an SO-100 or ALOHA arm from its own LeRobot recordings: free, local")
    rb.add_argument("action", choices=["calibrate"])
    rb.add_argument("dataset", help="a public LeRobot dataset of one SO-100 arm (format v2.1), e.g. 00ri/so100_battery")
    a = ap.parse_args(argv)
    return {"analyze": _analyze, "lens": _lens, "sky": _sky, "robot": _robot}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
