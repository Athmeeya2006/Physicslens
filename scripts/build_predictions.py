"""The /predictions page's data: every pre-registered prediction, intact or not, with its score if reality has given one.

  python scripts/build_predictions.py    # writes frontend/public/predictions.json

Reads predictions/ and outcomes/ through physicslens.prereg (which also checks each file against the hash written when
it was registered). Misses stay listed. Local and free.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens import prereg

OUT = ROOT / "frontend" / "public" / "predictions.json"


# What the page shows of each predicted item; outlines and simulated points stay in the prediction file itself.
KEEP = ("label", "value", "unit", "utc_approx", "ci68", "ci95", "sigma_days", "shift_min", "mass_kg", "median_lat_lon")


def slim(item: dict) -> dict:
    out = {k: item[k] for k in KEEP if k in item}
    for region in ("hdr68", "hdr95"):
        if isinstance(item.get(region), dict) and "area_km2" in item[region]:
            out[f"{region}_area_km2"] = item[region]["area_km2"]
    return out


def main() -> None:
    s = prereg.summary()
    for p in s["predictions"]:
        if p.get("prediction") and isinstance(p["prediction"].get("items"), list):
            p["prediction"] = {**{k: v for k, v in p["prediction"].items() if k not in ("items", "points", "outlines")},
                               "items": [slim(i) for i in p["prediction"]["items"]]}
    problems = prereg.verify()
    OUT.write_text(json.dumps({"generated": "scripts/build_predictions.py", **s, "integrity_problems": problems}, indent=1,
                              default=float))
    print(f"{len(s['predictions'])} predictions: {s['open']} open, {s['scored']} scored ({s['hits']} hits, {s['misses']} misses)"
          + (f"; integrity problems: {problems}" if problems else ""))


if __name__ == "__main__":
    main()
