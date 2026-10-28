"""Publish a finished study to the gallery: the study itself, plus each recording as its own (unlisted) gallery result.

Input is a folder saved from a real study run: study.json, report.md, model.xml (optional) and one folder per
recording (<run_id>/result.json and clip.mp4). Run ids are rewritten to gallery ids so every link works statically.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from add_to_gallery import make_poster  # noqa: E402
from physicslens.export.mp4 import render_share_video  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--id", required=True, help="gallery id of the study, e.g. gallery-truck")
    ap.add_argument("--title", required=True)
    ap.add_argument("--description", required=True)
    ap.add_argument("--credit", required=True)
    ap.add_argument("--order", type=int, default=50)
    a = ap.parse_args()

    src = Path(a.folder)
    study = json.loads((src / "study.json").read_text())
    if not study.get("turns") or not any(t.get("model") for t in study.get("trail", [])):
        sys.exit("Refusing: this study has no recorded agent decisions; only real study runs belong in the gallery.")
    renames = {}
    for i, rec in enumerate(study["recordings"], start=1):
        gid = f"{a.id}-{i}"
        renames[rec["run_id"]] = gid
        result = json.loads((src / rec["run_id"] / "result.json").read_text())
        dst = ROOT / "gallery" / gid
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / rec["run_id"] / "result.json", dst / "result.json")
        shutil.copy(src / rec["run_id"] / "clip.mp4", dst / "clip.mp4")
        render_share_video(result, dst / "clip.mp4", dst / "share.mp4")
        make_poster(dst / "share.mp4", dst / "poster.jpg")
        meta = {"title": f"{a.title}: {rec['label']}", "description": f"Recording {i} of the study '{study['goal']}'.",
                "credit": a.credit, "order": a.order, "hidden": True, "study": a.id}
        (dst / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  recording {gid}: {result.get('status')}")

    for rec in study["recordings"]:
        rec["run_id"] = renames[rec["run_id"]]
    study["id"] = a.id
    dst = ROOT / "gallery" / a.id
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "study.json").write_text(json.dumps(study, indent=1))
    report = (src / "report.md").read_text()
    for old, new in renames.items():
        report = report.replace(old, new)
    (dst / "report.md").write_text(report)
    if (src / "model.xml").is_file():
        shutil.copy(src / "model.xml", dst / "model.xml")
    shutil.copy(ROOT / "gallery" / f"{a.id}-1" / "poster.jpg", dst / "poster.jpg")
    (dst / "meta.json").write_text(json.dumps({"title": a.title, "description": a.description, "credit": a.credit,
                                               "order": a.order, "kind": "study"}, indent=2))
    print(f"Added study {a.id} with {len(renames)} recordings")


if __name__ == "__main__":
    main()
