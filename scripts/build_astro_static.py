"""Copy the astronomy results into the static site: frontend/public/astro/ (generated, not committed).

  python scripts/build_astro_static.py

Each case's results.json and its downloads (transit windows as CSV and ICS) are served as they are; the pages
(/timing, /fireball, /sun and the other astronomy cases) read them without a server.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "frontend" / "public" / "astro"
FILES = {
    "timing.json": ROOT / "data" / "astro" / "timing" / "results.json",
    "wasp12b_windows.csv": ROOT / "data" / "astro" / "timing" / "wasp12b_windows.csv",
    "wasp12b_windows.ics": ROOT / "data" / "astro" / "timing" / "wasp12b_windows.ics",
    "asteroids.json": ROOT / "data" / "astro" / "asteroids" / "results.json",
    "example_ades.psv": ROOT / "data" / "astro" / "asteroids" / "example_ades.psv",
    "occultation.json": ROOT / "data" / "astro" / "occultation" / "results.json",
    "satellites.json": ROOT / "data" / "astro" / "satellites" / "results.json",
    "sun.json": ROOT / "data" / "astro" / "sun" / "results.json",
    "sun_agent.json": ROOT / "data" / "astro" / "sun" / "agent.json",
    "fireball.json": ROOT / "data" / "astro" / "fireball" / "results.json",
    "winchcombe_strewn_field.geojson": ROOT / "data" / "astro" / "fireball" / "winchcombe_strewn_field.geojson",
    "winchcombe_strewn_field.kml": ROOT / "data" / "astro" / "fireball" / "winchcombe_strewn_field.kml",
}


def finite(v):
    """NaN and infinity are not JSON: a browser refuses the whole file (the satellite page showed only an error)."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: finite(x) for k, x in v.items()}
    if isinstance(v, list):
        return [finite(x) for x in v]
    return v


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for name, src in FILES.items():
        if src.is_file() and src.suffix == ".json":
            (DST / name).write_text(json.dumps(finite(json.loads(src.read_text())), allow_nan=False))
            print(f"{name}: {src.stat().st_size} bytes")
        elif src.is_file():
            shutil.copy(src, DST / name)
            print(f"{name}: {src.stat().st_size} bytes")
        else:
            print(f"{name}: missing ({src.relative_to(ROOT)}); run the case's script first")


if __name__ == "__main__":
    main()
