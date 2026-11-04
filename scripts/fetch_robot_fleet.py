"""Download the joint tables of SO-100 arms recorded by different owners, pinned to the revision fetched.

  python scripts/fetch_robot_fleet.py

For each public LeRobot dataset below (Apache-2.0, one SO-100 arm, teleoperated with a leader arm, 30 fps, LeRobot
format v2.0 or v2.1): meta/info.json and every episode's parquet (the command and the measured joint angles); no videos.
Writes var/robot/fleet/<owner>__<name>/ and pins the revision in its source.json. Free: no account, no key.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLEET = ROOT / "var" / "robot" / os.environ.get("PL_ROBOT_GROUP", "fleet")

# Five owners, each a different physical arm: single-arm SO-100, 40 to 50 episodes, Apache-2.0 (checked on the hub).
DATASETS = ["00ri/so100_battery", "jccj/so100_block_in_cup", "julyfun/so100_pick_and_place_v4_50",
            "youliangtan/so100_strawberry_grape", "sergiov2000/so100_lego_and_ball_40_lego",
            "pranavsaroha/so100_medicine_0605"]


def slug(repo: str) -> str:
    return repo.replace("/", "__")


def fetch(repo: str) -> None:
    out = FLEET / slug(repo)
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}", timeout=60) as r:
        meta = json.loads(r.read())
    rev = meta["sha"]
    licence = next((t.split(":", 1)[1] for t in meta.get("tags", []) if t.startswith("license:")), None)
    base = f"https://huggingface.co/datasets/{repo}/resolve/{rev}"
    (out / "meta").mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{base}/meta/info.json", timeout=60) as r:
        info = json.loads(r.read())
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=1))
    # v2.0 and v2.1 store the joint table the same way (one parquet per episode, same columns); v2.1 adds statistics.
    if info.get("codebase_version") not in ("v2.0", "v2.1"):
        raise SystemExit(f"{repo}: LeRobot format {info.get('codebase_version')}, expected v2.0 or v2.1")
    n = int(info["total_episodes"])
    for e in range(n):
        rel = info["data_path"].format(episode_chunk=e // int(info["chunks_size"]), episode_index=e)
        p = out / rel
        if not p.is_file():
            p.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(f"{base}/{rel}", timeout=120) as r:
                p.write_bytes(r.read())
    (out / "source.json").write_text(json.dumps({"dataset": repo, "revision": rev, "licence": licence,
                                                 "episodes": n, "fps": info["fps"], "robot_type": info.get("robot_type")}, indent=1))
    print(repo, rev[:10], licence, n, "episodes", flush=True)


if __name__ == "__main__":
    for repo in sys.argv[1:] or DATASETS:
        fetch(repo)
