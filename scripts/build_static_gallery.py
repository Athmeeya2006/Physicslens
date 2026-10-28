"""Bundle the gallery into the frontend as static files, so the site works without the backend.

Writes frontend/public/gallery/<id>/ (result, clip, share video, poster and the three exports) plus
frontend/public/gallery/index.json. Run before building the static site.
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.export.mjcf import build_mjcf
from physicslens.export.report import report_markdown  # noqa: E402
from physicslens.export.script import parameters_json, python_script

SRC = ROOT / "gallery"
DST = ROOT / "frontend" / "public" / "gallery"


def main() -> None:
    shutil.rmtree(DST, ignore_errors=True)
    DST.mkdir(parents=True)
    index = []
    for meta_file in sorted(SRC.glob("*/meta.json")):
        src = meta_file.parent
        gid = src.name
        dst = DST / gid
        dst.mkdir()
        for name in ("result.json", "clip.mp4", "share.mp4", "poster.jpg"):
            if (src / name).is_file():
                shutil.copy(src / name, dst / name)
        result = json.loads((src / "result.json").read_text())
        meta = json.loads(meta_file.read_text())
        (dst / f"physicslens-{gid}-report.md").write_text(report_markdown(result, meta.get("title")))
        if "fit_state" in result:
            (dst / f"physicslens-{gid}.json").write_text(parameters_json(result))
            (dst / f"physicslens-{gid}.py").write_text(python_script(result))
            xml, notes = build_mjcf(result["fit_state"])
            if result.get("scale_note"):
                notes = [*notes, "No scale reference in the clip: lengths in this model are in arbitrary units."]
            if notes:
                xml = xml.replace("<worldbody>", "<!-- " + " ".join(notes).replace("--", "-") + " -->\n  <worldbody>", 1)
            (dst / f"physicslens-{gid}.xml").write_text(xml)
        eq = result.get("equation") or {}
        index.append({"id": gid, **meta, "status": result.get("status"), "family": result.get("family"),
                      "equation_title": eq.get("title"), "object": (result.get("scene") or {}).get("moving_object")})
    index.sort(key=lambda i: i.get("order", 99))
    (DST / "index.json").write_text(json.dumps(index, indent=1))
    print(f"Bundled {len(index)} gallery entries into {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
