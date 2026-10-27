"""Promote a finished, real analysis into the public gallery (tracked in git, served read-only, free to view).

Only real runs belong here: every gallery result must come from the actual pipeline with Meta's models.
Usage: python scripts/add_to_gallery.py RUN_ID --id gallery-pendulum --title "..." --description "..." \
           --credit "Video: ... (license)" [--order 1]
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.export.mp4 import render_share_video


def make_poster(video: Path, out: Path) -> None:
    """A real frame (with the ghost and equation overlay) as the gallery thumbnail."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                           capture_output=True, text=True, check=True)
    at = 0.4 * float(probe.stdout.strip() or 1)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{at:.2f}", "-i", str(video), "-frames:v", "1",
                    "-vf", "scale=640:-2", "-q:v", "4", str(out)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--id", required=True, help="gallery id, e.g. gallery-pendulum")
    ap.add_argument("--title", required=True)
    ap.add_argument("--description", required=True)
    ap.add_argument("--credit", required=True, help="who filmed the clip and under which license")
    ap.add_argument("--order", type=int, default=50)
    ap.add_argument("--runs", default=str(ROOT / "var" / "runs"))
    args = ap.parse_args()

    src = Path(args.runs) / args.run_id
    result = json.loads((src / "result.json").read_text())
    if not result.get("llm_calls") or not any(s.get("step") == "tracking" for s in result.get("trail", [])):
        sys.exit("Refusing: this result did not come from a real pipeline run (no model calls recorded).")
    dst = ROOT / "gallery" / args.id
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "result.json", dst / "result.json")
    shutil.copy(src / "clip.mp4", dst / "clip.mp4")
    render_share_video(result, dst / "clip.mp4", dst / "share.mp4")
    make_poster(dst / "share.mp4", dst / "poster.jpg")
    meta = {"title": args.title, "description": args.description, "credit": args.credit, "order": args.order}
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Added {args.id}: {sum(f.stat().st_size for f in dst.iterdir()) // 1024} KB")


if __name__ == "__main__":
    main()
