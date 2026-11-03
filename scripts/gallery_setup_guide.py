"""Draw a gallery result's requested experiment on its own frame (Muse Image) and store it with the gallery entry.

  python scripts/gallery_setup_guide.py gallery-pendulum-longer      # PAID: one Muse Image call (cents), cost-guarded

The same function the site's "Show me" button calls (agent/setup_guide.py): the open-weight model plans where the marks
go, Muse Image draws them, and the picture is stored as setup_guide.jpg beside the result, so visitors see it without
a new call. Uses the design step of the result's decision trail.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.agent.setup_guide import make_setup_guide
from physicslens.costguard import CostGuardClient
from physicslens.settings import load_settings


def main() -> None:
    gid = sys.argv[1]
    src = ROOT / "gallery" / gid
    result = json.loads((src / "result.json").read_text())
    step_ref = next(i for i, s in enumerate(result["trail"]) if s.get("step") == "design")
    client = CostGuardClient(load_settings())
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        shutil.copy(src / "clip.mp4", run_dir / "clip.mp4")
        guide = make_setup_guide(client, run_dir, result, step_ref, f"guide-{gid}", "dev")
        print(gid, guide["status"], "|", guide.get("instruction"), "|", guide.get("reason") or "", f"${guide.get('cost_usd', 0):.3f}")
        if guide["status"] != "picture":
            sys.exit("no picture was made; nothing stored")
        for f in ("setup_guide.jpg", "setup_guide_frame.jpg", "setup_guide.json"):
            shutil.copy(run_dir / f, src / f)
    result["trail"] = [*[s for s in result["trail"] if s.get("step") != "setup_guide"], guide["trail_item"]]
    (src / "result.json").write_text(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
