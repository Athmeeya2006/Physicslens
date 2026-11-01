"""Precompute every lab scenario for the static site: frontend/public/sky/<name>.json and index.json.

  python scripts/build_sky_static.py

The lab page shows these when no analysis server is online, and re-integrates a changed scenario in the browser from
the bundled starting states; with a server, the server's simulation replaces the preview. Generated, not committed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.sky import scenarios as S
from physicslens.sky_api import static_bundle

OUT = ROOT / "frontend" / "public" / "sky"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for name in S.names():
        bundle = static_bundle(name)
        (OUT / f"{name}.json").write_text(json.dumps(bundle, separators=(",", ":"), default=float))
        index.append(S.load(name).summary())
        print(f"{name}: {len(bundle['bodies'])} bodies, {bundle['seconds']:.1f} s", flush=True)
    (OUT / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
