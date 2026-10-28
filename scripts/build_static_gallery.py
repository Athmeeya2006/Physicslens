"""Bundle the gallery into the frontend as static files, so the site works without the backend.

Writes frontend/public/gallery/<id>/ (result, clip, share video, poster, any stored setup guide and the
exports) plus
frontend/public/gallery/index.json. Run before building the static site.
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.export.mjcf import bounce_mjcf, build_mjcf
from physicslens.export.report import report_markdown
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
        # A stored picture guide (drawn once with the access code) is shown to everyone without a new call.
        for name in ("result.json", "clip.mp4", "share.mp4", "poster.jpg",
                     "setup_guide.jpg", "setup_guide_frame.jpg", "setup_guide.json"):
            if (src / name).is_file():
                shutil.copy(src / name, dst / name)
        meta = json.loads(meta_file.read_text())
        if meta.get("kind") == "study":
            # A study: its plan, recordings and decisions, the report and the combined model, all static.
            shutil.copy(src / "study.json", dst / "study.json")
            shutil.copy(src / "report.md", dst / f"physicslens-{gid}.md")
            if (src / "model.xml").is_file():
                shutil.copy(src / "model.xml", dst / f"physicslens-{gid}.xml")
            study = json.loads((src / "study.json").read_text())
            index.append({"id": gid, **meta, "status": study.get("status")})
            continue
        result = json.loads((src / "result.json").read_text())
        (dst / f"physicslens-{gid}-report.md").write_text(report_markdown(result, meta.get("title")))
        contact = (result.get("bounces") or {}).get("mujoco_contact")
        if contact:
            b = result["bounces"]
            (dst / f"physicslens-{gid}-contact.xml").write_text(bounce_mjcf(b["g"], b["e_mean"], contact["radius_m"], contact["dampratio"]))
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
        if meta.get("hidden"):
            continue  # a study's recording: reachable from its study, not listed on its own
        index.append({"id": gid, **meta, "status": result.get("status"), "family": result.get("family"),
                      "equation_title": eq.get("title"), "object": (result.get("scene") or {}).get("moving_object")})
    index.sort(key=lambda i: i.get("order", 99))
    (DST / "index.json").write_text(json.dumps(index, indent=1))
    print(f"Bundled {len(index)} gallery entries into {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
