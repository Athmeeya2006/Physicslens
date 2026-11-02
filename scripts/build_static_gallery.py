"""Bundle the gallery into the frontend as static files, so the site works without the backend.

Writes frontend/public/gallery/<id>/ (result, clip, share video, poster, share image, the three exports, the sim-to-real comparison and any
stored setup guide) plus
frontend/public/r/<id>.html (a share page per result, served as /r/<id>), frontend/public/gallery/index.json and the open dataset (dataset.json, dataset.csv). Run before building the static site.
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.export.dataset import croissant, to_csv
from physicslens.export.dataset import entries as dataset_entries
from physicslens.export.mjcf import MjcfError, bounce_mjcf, build_mjcf
from physicslens.export.og import render as render_og
from physicslens.export.og import share_page
from physicslens.export.outcome import build_outcome
from physicslens.export.report import report_markdown
from physicslens.export.robot import RobotExportError, sdf, urdf
from physicslens.export.script import parameters_json, python_script
from physicslens.export.sim2real import Sim2RealError, sim2real

SRC = ROOT / "gallery"
DST = ROOT / "frontend" / "public" / "gallery"
SHARE = ROOT / "frontend" / "public" / "r"
SITE = "https://physicslens.vercel.app"


def main() -> None:
    for d in (DST, SHARE):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    index, dataset = [], []
    for meta_file in sorted(SRC.glob("*/meta.json")):
        src = meta_file.parent
        gid = src.name
        dst = DST / gid
        dst.mkdir()
        # A stored picture guide (drawn once with the access code) is shown to everyone without a new call.
        for name in ("result.json", "clip.mp4", "share.mp4", "poster.jpg",
                     "setup_guide.jpg", "setup_guide_frame.jpg", "setup_guide.json", "prediction.json"):
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
        if (src / "setup_guide.json").is_file():
            result["setup_guide_available"] = True  # only then does the page offer to show the stored picture
        if (src / "prediction.json").is_file():
            # Tells the page there is a forecast to fetch, so results without one make no request that would 404.
            result["prediction_available"] = True
        # The outcome card, checked against the gallery entry's independent truth when it has one.
        result["outcome"] = build_outcome(result, truth=meta.get("truth"))
        (dst / "result.json").write_text(json.dumps(result))
        (dst / f"physicslens-{gid}-report.md").write_text(report_markdown(result, meta.get("title")))
        contact = (result.get("bounces") or {}).get("mujoco_contact")
        if contact:
            b = result["bounces"]
            (dst / f"physicslens-{gid}-contact.xml").write_text(bounce_mjcf(b["g"], b["e_mean"], contact["radius_m"], contact["dampratio"]))
        if "fit_state" in result:
            (dst / f"physicslens-{gid}.json").write_text(parameters_json(result))
            (dst / f"physicslens-{gid}.py").write_text(python_script(result))
            try:
                xml, notes = build_mjcf(result["fit_state"])
            except MjcfError as e:  # e.g. a law the agent wrote for free planar motion: no MuJoCo body to map it to
                print(f"{gid}: no MuJoCo export ({e})")
            else:
                if result.get("scale_note"):
                    notes = [*notes, "No scale reference in the clip: lengths in this model are in arbitrary units."]
                if notes:
                    xml = xml.replace("<worldbody>", "<!-- " + " ".join(notes).replace("--", "-") + " -->\n  <worldbody>", 1)
                (dst / f"physicslens-{gid}.xml").write_text(xml)
            for kind, writer in (("urdf", urdf), ("sdf", sdf)):
                try:
                    (dst / f"physicslens-{gid}.{kind}").write_text(writer(result["fit_state"]))
                except RobotExportError:
                    pass  # only pendulums, springs and projectiles: the outcome card lists what exists

            if result.get("status") == "accepted":
                try:
                    (dst / "sim2real.json").write_text(json.dumps(sim2real(result)))
                except Sim2RealError as e:
                    print(f"{gid}: no sim-to-real comparison ({e})")
        # A link to /r/<id> previews as this result's own picture, then opens the result.
        render_og(result, src / "poster.jpg", dst / "og.jpg")
        (SHARE / f"{gid}.html").write_text(share_page(gid, result, SITE))
        dataset += dataset_entries(gid, result, meta)
        eq = result.get("equation") or {}
        if meta.get("hidden"):
            continue  # a study's recording: reachable from its study, not listed on its own
        index.append({"id": gid, **meta, "status": result.get("status"), "family": result.get("family"),
                      "equation_title": eq.get("title"), "object": (result.get("scene") or {}).get("moving_object")})
    index.sort(key=lambda i: i.get("order", 99))
    (DST / "index.json").write_text(json.dumps(index, indent=1))
    (DST / "dataset.json").write_text(json.dumps(dataset, indent=1))
    (DST / "dataset.csv").write_text(to_csv(dataset))
    # Croissant metadata hashes the exact published files, and the card travels with them.
    (DST / "croissant.json").write_text(json.dumps(
        croissant((DST / "dataset.json").read_bytes(), (DST / "dataset.csv").read_bytes(), SITE), indent=1))
    shutil.copy(SRC / "DATASET_CARD.md", DST / "DATASET_CARD.md")
    print(f"Bundled {len(index)} gallery entries into {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
