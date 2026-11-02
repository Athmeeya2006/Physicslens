"""Download exactly the public files the robot arm case needs, pinned to fixed revisions and checked by sha256.

  python scripts/fetch_robot.py

Writes into var/robot/raw/ (gitignored). Two sources, both Apache-2.0:
  - the LeRobot dataset lerobot/svla_so100_pickplace (50 teleoperated episodes of an SO-100 arm): only the joint
    table and its metadata (about 0.6 MB); the two camera videos (about 470 MB) are not needed and not fetched;
  - the MuJoCo Menagerie model of the same arm (trs_so_arm100): the MJCF, its meshes, licence and readme.
Free: no model calls, no account, no key. See data/robot/SOURCES.md.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "var" / "robot" / "raw"

HF_REPO = "lerobot/svla_so100_pickplace"
HF_REV = "728583b5eaf9e739a7f119e2def466fa1d552402"
MENAGERIE_REV = "b0655057a9df2abe39ea7fc989914ba6f248301c"

HF_FILES = ["README.md", "meta/info.json", "meta/stats.json", "meta/tasks.parquet",
            "meta/episodes/chunk-000/file-000.parquet", "data/chunk-000/file-000.parquet"]
MESHES = ["Base", "Base_Motor", "Fixed_Jaw", "Fixed_Jaw_Collision_1", "Fixed_Jaw_Collision_2", "Fixed_Jaw_Motor",
          "Lower_Arm", "Lower_Arm_Motor", "Moving_Jaw", "Moving_Jaw_Collision_1", "Moving_Jaw_Collision_2",
          "Moving_Jaw_Collision_3", "Rotation_Pitch", "Rotation_Pitch_Motor", "Upper_Arm", "Upper_Arm_Motor",
          "Wrist_Pitch_Roll", "Wrist_Pitch_Roll_Motor"]
MENAGERIE_FILES = ["so_arm100.xml", "scene.xml", "LICENSE", "README.md", *[f"assets/{m}.stl" for m in MESHES]]

# sha256 of every file, recorded from the first download (--print-hashes). For the parquet files they equal the
# Hub's own LFS object ids. A mismatch stops the run: every number in data/robot/result.json depends on these bytes.
PINNED_FILE = ROOT / "data" / "robot" / "sha256.txt"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _get(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r:
        tmp.write_bytes(r.read())
    tmp.replace(dest)


def targets() -> list[tuple[str, str, Path]]:
    """(key, url, local path) for every file, the key being how sha256.txt names it."""
    out = []
    for f in HF_FILES:
        out.append((f"lerobot/{f}", f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REV}/{f}", RAW / "lerobot" / f))
    for f in MENAGERIE_FILES:
        url = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{MENAGERIE_REV}/trs_so_arm100/{f}"
        out.append((f"menagerie/{f}", url, RAW / "menagerie" / f))
    return out


def pinned() -> dict[str, str]:
    if not PINNED_FILE.exists():
        return {}
    rows = [ln.split() for ln in PINNED_FILE.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return {k: h for h, k in rows}


def main(argv: list[str]) -> int:
    pins = pinned()
    record = "--print-hashes" in argv
    lines, bad = [], []
    for key, url, dest in targets():
        if not dest.exists():
            print(f"fetching {key}")
            _get(url, dest)
        h = _sha(dest)
        lines.append(f"{h}  {key}")
        if key in pins and pins[key] != h:
            bad.append(key)
    if record:
        print("\n".join(lines))
        return 0
    if not pins:
        print("no data/robot/sha256.txt to check against; run with --print-hashes to see them", file=sys.stderr)
        return 1
    missing = [k for k, _, _ in targets() if k not in pins]
    if bad or missing:
        print(f"checksum mismatch: {bad}; not pinned: {missing}", file=sys.stderr)
        return 1
    print(f"all {len(lines)} files present and match data/robot/sha256.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
