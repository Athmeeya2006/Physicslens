"""Score pre-registered transit-timing predictions once their date has passed.

  python scripts/score_predictions.py            # every open transit-timing prediction whose score_after has passed
  python scripts/score_predictions.py --dry-run  # say what would be scored, write nothing

For each one: look for TESS SPOC 2-minute light curves of the star taken after the prediction was registered
(lightkurve, cached under var/ephemeris/score/), measure every transit's mid-time with the same code that made the
prediction (astro/transits.py), match each to the predicted transit of the same epoch, and write
outcomes/transit-timing/<id>.json through prereg.score. The prediction file is never touched. A predicted transit
nobody measured is "unmeasured", neither hit nor miss. Amateur timings (ExoClock, ETD) are not fetched
automatically: their pages are listed in each prediction's sources_to_check for a person to read.

Free (public MAST data), no model calls.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens import prereg
from physicslens.astro import ephemeris as E

CASE = "transit-timing"
CACHE = ROOT / "var" / "ephemeris" / "score"
UTC_FMT = "%Y-%m-%dT%H:%M:%SZ"


def match(doc: dict[str, Any], measured: list[dict[str, Any]]) -> dict[str, Any]:
    """Each predicted transit against the measured mid-time of the same epoch, if any."""
    items = doc["prediction"]["items"]
    period = doc["prediction"]["ephemeris"]["period_days"] if "ephemeris" in doc["prediction"] else None
    by_epoch: dict[int, dict[str, Any]] = {}
    for m in measured:
        i = int(np.argmin([abs(m["bjd_tdb"] - it["value"]) for it in items]))
        if period and abs(m["bjd_tdb"] - items[i]["value"]) > period / 4:
            continue  # not one of the predicted transits (outside the predicted span)
        by_epoch[i] = m
    rows = []
    for i, it in enumerate(items):
        if i in by_epoch:
            m = by_epoch[i]
            rows.append({**E.score_transit(it, m["bjd_tdb"], m["err_days"]), "source": m.get("source", "")})
        else:
            rows.append({"label": it["label"], "predicted": it["value"], "measured": None, "hit": None})
    hits = sum(1 for r in rows if r["hit"] is True)
    misses = sum(1 for r in rows if r["hit"] is False)
    unmeasured = sum(1 for r in rows if r["hit"] is None)
    share = hits / (hits + misses) if hits + misses else None
    strict = sum(1 for r in rows if r.get("inside_registered_ci95"))
    return {"items": rows, "hits": hits, "misses": misses, "unmeasured": unmeasured, "hit_share": share,
            "inside_registered_ci95": strict,
            "summary": (f"{hits} of {hits + misses} measured transits inside their 95% window (widened by each "
                        f"measurement's error); {strict} inside the window as registered; {unmeasured} not measured"
                        if hits + misses else f"none of the {len(rows)} predicted transits has been measured yet")}


def tess_fetch(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Mid-times from TESS SPOC 2-minute sectors observed after the prediction was registered."""
    import lightkurve as lk
    from astropy.time import Time
    from physicslens.astro.transits import transit_times
    eph = doc["prediction"]["ephemeris"]
    created = Time(doc["created_utc"].rstrip("Z"), scale="utc").mjd
    res = lk.search_lightcurve(eph["tic_id"], author="SPOC", exptime=120)
    out = []
    for i in range(len(res)):
        if float(res.table["t_min"][i]) <= created:
            continue  # data the prediction may have seen
        lc = res[i].download(download_dir=str(CACHE)).remove_nans().normalize()
        lc = lc[lc.quality == 0]
        t = np.asarray(lc.time.value, float) + 2457000.0  # BTJD to BJD_TDB
        try:
            r = transit_times(t, np.asarray(lc.flux.value, float), np.asarray(lc.flux_err.value, float),
                              eph["t0_bjd_tdb"], eph["period_days"], eph["duration_days"],
                              search_half_width=0.1, exp_time=E.TESS_2MIN_EXP_DAYS, refine=True)
        except ValueError as err:
            print(f"  {res.table['mission'][i]}: {err}")
            continue
        out += [{"bjd_tdb": row["bjd_tdb"], "err_days": row["err_days"],
                 "source": f"TESS SPOC 2-min {res.table['obs_id'][i]}"} for row in r["rows"]]
    return out


def score_all(preds: Path = prereg.PRED, outs: Path = prereg.OUTCOMES, now: datetime | None = None,
              fetch: Callable[[dict[str, Any]], list[dict[str, Any]]] = tess_fetch, dry_run: bool = False) -> list[Path]:
    """Score every transit-timing prediction that is due and not yet scored. Returns the outcome files written."""
    now = now or datetime.now(UTC)
    written = []
    for r in prereg.index_rows(preds):
        if r["case"] != CASE or (outs / CASE / f"{r['id']}.json").exists():
            continue
        if now.strftime(UTC_FMT) < r["score_after_utc"]:
            continue
        doc = json.loads(prereg.path_for(CASE, r["id"], preds).read_text())
        print(f"{r['id']}: due since {r['score_after_utc']}")
        if dry_run:
            continue
        outcome = match(doc, fetch(doc))
        outcome["method"] = "TESS SPOC 2-min sectors after registration, mid-times by astro/transits.py"
        written.append(prereg.score(CASE, r["id"], outcome, root=preds, outcomes=outs, now=now))
    return written


if __name__ == "__main__":
    for p in score_all(dry_run="--dry-run" in sys.argv):
        print("wrote", p.relative_to(ROOT))
