"""Run analyses with every reasoning step on Muse Glimmer (open weights), served by llama.cpp on a GPU notebook.

The GPU machine never holds a Meta key. SAM 3.1 tracking runs on the developer's machine through costguard,
and its results reach the GPU machine as cached responses, so the two sides take turns:

  local   prepare    freeze each clip into a bundle (the exact clip and keyframe bytes both sides use)
  remote  run        Glimmer reasons; Meta is cache-only, so a clip without SAM results stops at tracking
  local   import     merge Glimmer's cached answers from the remote output into var/cache.db
  local   track      re-run each bundle: Glimmer is cache-only here, SAM is called for real (costguard caps apply)
  local   export     write the SAM results the remote side needs into a small cache file
  remote  run        now complete: results carry Glimmer as the model behind every reasoning step

Bundles, caches and results never hold keys: cached values are model outputs only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import traceback
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.costguard import CacheOnlyMiss, CostGuardClient
from physicslens.perception import media
from physicslens.pipeline import RunOptions, analyze, prepare
from physicslens.settings import GLIMMER_MODEL, SAM_MODEL, load_settings

BUNDLES = ROOT / "var" / "glimmer" / "bundles"


def _client() -> CostGuardClient:
    return CostGuardClient(load_settings())


def cmd_prepare(a: argparse.Namespace) -> None:
    out = BUNDLES / a.name
    run_dir = out / "_prep"
    client = _client()
    clip, info, frames = prepare(Path(a.video), run_dir, RunOptions(), client.limits.get("media") or {},
                                 start_s=a.start, seconds=a.seconds)
    media.save_prepared(clip, info, frames, out)
    shutil.rmtree(run_dir)
    (out / "options.json").write_text(json.dumps({"pendulum_length_m": a.length, "object_size_m": a.size,
                                                   "slow_motion": a.slowmo, "object_hint": a.hint}))
    print(f"bundle {out}: {info.frames} frames at {info.fps:.2f} fps, {len(frames)} keyframes")


def _run_bundles(bundles: Path, out: Path, names: list[str] | None) -> dict[str, str]:
    client = _client()
    status: dict[str, str] = {}
    for b in sorted(p for p in bundles.iterdir() if (p / "info.json").is_file()):
        if names and b.name not in names:
            continue
        opts = json.loads((b / "options.json").read_text()) if (b / "options.json").is_file() else {}
        run_dir = out / b.name
        try:
            result = analyze(client, b / "clip.mp4", run_dir, f"glimmer-{b.name}",
                             RunOptions(pendulum_length_m=opts.get("pendulum_length_m"),
                                        object_size_m=opts.get("object_size_m"),
                                        slow_motion=opts.get("slow_motion", 1.0),
                                        object_hint=opts.get("object_hint"), prepared=b),
                             progress=lambda m, n=b.name: print(f"[{n}] {m}", flush=True))
        except CacheOnlyMiss as e:
            status[b.name] = f"waiting: {e}"
            print(f"[{b.name}] stopped: {e}", flush=True)
            continue
        except Exception as e:  # noqa: BLE001  one bad clip must not lose the others' results (traceback printed)
            status[b.name] = f"error: {type(e).__name__}: {e}"
            traceback.print_exc()
            continue
        (run_dir / "result.json").write_text(json.dumps(result))
        status[b.name] = f"done: {result['status']}"
        print(f"[{b.name}] done: {result['status']}: {result['reason']}", flush=True)
    return status


def cmd_run(a: argparse.Namespace) -> None:
    """On the GPU machine: REASONING_PROVIDER=glimmer, PHYSICSLENS_CACHE_ONLY=meta_api."""
    if os.environ.get("REASONING_PROVIDER") != "glimmer" or "meta_api" not in os.environ.get("PHYSICSLENS_CACHE_ONLY", ""):
        sys.exit("run expects REASONING_PROVIDER=glimmer and PHYSICSLENS_CACHE_ONLY=meta_api")
    out = Path(a.out)
    status = _run_bundles(Path(a.bundles), out, a.only)
    (out / "status.json").write_text(json.dumps(status, indent=1))
    _copy_rows(load_settings().cache_path, out / "glimmer_cache.db", [GLIMMER_MODEL])
    print(json.dumps(status, indent=1))


def cmd_track(a: argparse.Namespace) -> None:
    """Locally: replays Glimmer's answers from the cache and calls SAM for real where it is missing."""
    os.environ["REASONING_PROVIDER"] = "glimmer"
    os.environ["PHYSICSLENS_CACHE_ONLY"] = "glimmer"
    status = _run_bundles(BUNDLES, ROOT / "var" / "glimmer" / "local_runs", a.only)
    print(json.dumps(status, indent=1))


def cmd_import(a: argparse.Namespace) -> None:
    n = _copy_rows(Path(a.db), load_settings().cache_path, [GLIMMER_MODEL])
    print(f"imported {n} Glimmer answers")


def cmd_export(a: argparse.Namespace) -> None:
    n = _copy_rows(load_settings().cache_path, Path(a.out), [SAM_MODEL, GLIMMER_MODEL])
    print(f"exported {n} cached SAM and Glimmer responses to {a.out}")


def _copy_rows(src: Path, dst: Path, models: list[str]) -> int:
    from physicslens.costguard.cache import ResponseCache

    ResponseCache(dst)  # creates the schema
    marks = ",".join("?" * len(models))
    with closing(sqlite3.connect(src)) as s, closing(sqlite3.connect(dst)) as d, d:
        rows = s.execute(f"SELECT key, model, value, created, expires FROM cache WHERE model IN ({marks})", models).fetchall()
        d.executemany("INSERT OR REPLACE INTO cache (key, model, value, created, expires) VALUES (?, ?, ?, ?, ?)", rows)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("video")
    p.add_argument("name")
    p.add_argument("--length", type=float, default=None, help="pendulum length in m, if known")
    p.add_argument("--size", type=float, default=None, help="longest dimension of the moving object in m, if known")
    p.add_argument("--hint", default=None, help="what to measure, in the user's words")
    p.add_argument("--slowmo", type=float, default=1.0, help="capture fps / playback fps for slow-motion footage")
    p.add_argument("--start", type=float, default=0.0, help="seconds into the source video")
    p.add_argument("--seconds", type=float, default=None, help="length to keep")
    p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("run")
    p.add_argument("--bundles", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--only", nargs="*")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("track")
    p.add_argument("--only", nargs="*")
    p.set_defaults(fn=cmd_track)
    p = sub.add_parser("import")
    p.add_argument("db")
    p.set_defaults(fn=cmd_import)
    p = sub.add_parser("export")
    p.add_argument("out")
    p.set_defaults(fn=cmd_export)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
