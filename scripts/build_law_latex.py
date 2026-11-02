"""Every library law's equation, for the site: frontend/public/laws.json maps a law's id to its title and LaTeX.

  python scripts/build_law_latex.py

Results stored before every trail step carried its equations name laws by id only; the decision trail looks the
equation up here, so every law the agent proposed, tested, rejected or kept is shown with its equation. Local, free.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.physics.models import MODELS

OUT = ROOT / "frontend" / "public" / "laws.json"


def main() -> None:
    laws = {m.id: {"title": m.title, "latex": m.latex} for m in MODELS.values() if not m.id.startswith("custom_")}
    OUT.write_text(json.dumps(laws, indent=0, sort_keys=True))
    print(f"{len(laws)} laws -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
