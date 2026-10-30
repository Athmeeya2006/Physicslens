"""Clean videos for the home page's discoveries: each gallery result's ghost, trail and force arrows, without the
equation box (the page typesets the law itself, large), sized for the web.

  python scripts/build_stories.py      -> frontend/public/stories/<gallery id>.mp4 and .jpg

Local work only: re-draws stored results onto their stored clips, no model calls.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.export.mp4 import render_share_video  # noqa: E402

GALLERY = ROOT / "gallery"
OUT = ROOT / "frontend" / "public" / "stories"
STORIES = ("gallery-halley", "gallery-jupiter", "gallery-pendulum", "gallery-paperbag", "gallery-spring")
# The orbit renders carry their own legend in the bottom rows of the clip: crop it off (fraction of the height kept).
KEEP_HEIGHT = {"gallery-halley": 0.82, "gallery-jupiter": 0.86}
WIDTH = 720


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for gid in STORIES:
        src = GALLERY / gid
        result = json.loads((src / "result.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            full = Path(tmp) / "full.mp4"
            render_share_video(result, src / "clip.mp4", full, panel=False)
            keep = KEEP_HEIGHT.get(gid, 1.0)
            vf = f"crop=iw:trunc(ih*{keep}/2)*2:0:0,scale={WIDTH}:-2"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(full), "-vf", vf, "-an", "-c:v", "libx264",
                            "-preset", "slow", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            "-threads", "2", str(OUT / f"{gid}.mp4")], check=True)
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                    str(OUT / f"{gid}.mp4")], capture_output=True, text=True, check=True).stdout)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{dur * 0.4:.2f}", "-i", str(OUT / f"{gid}.mp4"),
                        "-frames:v", "1", "-q:v", "4", str(OUT / f"{gid}.jpg")], check=True)
        print(gid, f"{(OUT / f'{gid}.mp4').stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
