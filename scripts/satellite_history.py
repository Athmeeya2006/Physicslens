"""Score the ISS manoeuvre detector over the station's archived trajectory history.

Steps: (1) fetch every distinct archived copy of NASA's ISS ephemeris from the Internet Archive (cached, slow on
purpose), (2) run the detector over the whole history with its rule fixed in `physicslens.astro.sat_history`,
(3) only then match its detections to NASA's records: the planned-burn tables in the files' headers and the dated
reports in data/astro/satellites/announced_burns.json, (4) read the current official file and pre-register a detection
if, and only if, there is one.

    PYTHONPATH=backend .venv/bin/python scripts/satellite_history.py [--no-fetch]
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens import prereg
from physicslens.astro import sat_history as H

OEM_URL = "https://nasa-public-data.s3.amazonaws.com/iss-coords/current/ISS_OEM/ISS.OEM_J2K_EPH.txt"
CDX = ("http://web.archive.org/cdx/search/cdx?url=nasa-public-data.s3.amazonaws.com/iss-coords/current/ISS_OEM/"
       "ISS.OEM_J2K_EPH.txt&output=json&fl=timestamp,statuscode,digest,length&collapse=digest")
CACHE = ROOT / "var" / "astro" / "sat"
ARCHIVE = CACHE / "wayback"
OUT = ROOT / "data" / "astro" / "satellites" / "history.json"
ANNOUNCED = ROOT / "data" / "astro" / "satellites" / "announced_burns.json"
PAUSE_S = 4.0  # between Internet Archive requests: it is a free service and rate-limits bursts


def _get(url: str, timeout: float = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "physicslens-research (one-off archive read)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def complete(text: str) -> bool:
    """A whole ephemeris: the header at the top and a full state line at the end (a cut-off download has neither)."""
    lines = text.rstrip().splitlines()
    return bool(lines) and text.startswith("CCSDS") and len(lines[-1].split()) == 7


def fetch_archive() -> list[str]:
    """Every distinct capture (by content digest) of the ephemeris URL; already cached files are not refetched."""
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    index = CACHE / "cdx_oem.json"
    if not index.exists():
        index.write_bytes(_get(CDX))
    stamps = [r[0] for r in json.loads(index.read_text())[1:] if r[1] == "200"]
    for ts in stamps:
        path = ARCHIVE / f"oem_{ts}.txt"
        if path.exists() and complete(path.read_text(errors="replace")):
            continue
        try:
            path.write_bytes(_get(f"http://web.archive.org/web/{ts}id_/{OEM_URL}"))
        except OSError as e:
            print(f"capture {ts}: {e} (left out, rerun to retry)")
        time.sleep(PAUSE_S)
    return stamps


def current_file() -> Path:
    path = CACHE / "current_oem.txt"
    if not path.exists() or time.time() - path.stat().st_mtime > 86400:
        path.write_bytes(_get(OEM_URL))
    return path


def load_archive() -> tuple[list[H.OemFile], dict[datetime, str]]:
    """One file per NASA creation time (different captures of the same file are the same data)."""
    files: dict[datetime, H.OemFile] = {}
    names: dict[datetime, str] = {}
    bad = 0
    for p in sorted(ARCHIVE.glob("oem_*.txt")):
        text = p.read_text(errors="replace")
        try:
            if not complete(text):
                raise ValueError("cut off")
            f = H.read_oem_file(text)
        except ValueError:
            bad += 1
            continue
        files.setdefault(f.created, f)
        names.setdefault(f.created, p.name)
    if bad:
        print(f"{bad} captures were not complete ephemerides and were skipped")
    return [files[k] for k in sorted(files)], names


def iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def references(files: list[H.OemFile]) -> tuple[list[dict], list[dict], list[dict]]:
    doc = json.loads(ANNOUNCED.read_text())
    rows = [{**b, "tig": datetime.fromisoformat(b["tig_utc"]).replace(tzinfo=UTC)} for b in doc["burns"]]
    executed = [b for b in rows if b["kind"] != "aborted"]
    aborted = [b for b in rows if b["kind"] == "aborted"]
    # A plan listed for the same moment as an aborted attempt did not happen: it is neither a miss nor quiet data.
    planned = [p for p in H.planned_reference(files)
               if not any(abs((p["tig"] - a["tig"]).total_seconds()) < 6 * 3600 for a in aborted)]
    return executed, planned, aborted


def public(s: dict) -> dict:
    """JSON-ready copy of a score: datetimes as ISO strings, the false-alarm gaps trimmed to what a reader needs."""
    def conv(x):
        if isinstance(x, datetime):
            return iso(x)
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        if isinstance(x, list | tuple):
            return [conv(v) for v in x]
        if isinstance(x, float):
            return round(x, 4)
        return x
    return conv(s)


def gap_row(g: dict) -> dict:
    return {"t0": iso(g["t0"]), "t1": iso(g["t1"]), "days": round(g["days"], 3),
            "observed_da_km": round(g["observed_da_km"], 4), "drag_km_per_day": round(g["drag_km_per_day"], 4),
            "dv_ms": round(g["dv_ms"], 4), "scored": g["scored"], "detected": g["detected"]}


def dv_comparison(gap_list: list[dict], burns: list[dict]) -> list[dict]:
    """Estimated against NASA's delta-v, per scored gap that holds reference burns with a stated delta-v (a gap
    with two burns is compared with their sum)."""
    out = []
    for g in gap_list:
        inside = [b for b in burns if H.gap_for(b["tig"], gap_list) is g]
        if inside and all(b.get("dv_ms") is not None for b in inside):
            ref = sum(b["dv_ms"] for b in inside)
            out.append({"gap": [iso(g["t0"]), iso(g["t1"])], "burns": [b["event"] for b in inside],
                        "nasa_dv_ms": round(ref, 3), "estimated_dv_ms": round(g["dv_ms"], 3),
                        "difference_ms": round(g["dv_ms"] - ref, 3)})
    return out


def quiet_noise(gap_list: list[dict], every_burn: list[dict]) -> float:
    """Spread of the delta-v estimate where nothing fired (robust: 1.4826 x median absolute deviation)."""
    q = np.array([g["dv_ms"] for g in gap_list if g["scored"]
                  and not any(H._inside(b["tig"], g) for b in every_burn)])
    return float(1.4826 * np.median(np.abs(q - np.median(q)))) if len(q) else float("nan")


def sensitivity(gap_list: list[dict], burns: list[dict], every_burn: list[dict]) -> list[dict]:
    """The same scoring at other thresholds. Shown after the fact for context only: the result is the fixed rule."""
    out = []
    for thr in (0.1, 0.15, 0.2, 0.3, 0.5):
        gl = [{**g, "detected": g["scored"] and abs(g["dv_ms"]) >= thr} for g in gap_list]
        s = H.score(gl, burns, every_burn)
        out.append({"threshold_ms": thr, "recall": s["recall"], "detected": s["detected"],
                    "covered": s["burns_covered"], "false_alarms": s["false_alarms"]})
    return out


def forward_look(files: list[H.OemFile], names: dict, noise: float) -> dict:
    """The current official file: its header plan, any burn modelled into its forecast, and a pre-registration if the
    detector fires between it and the newest archived file."""
    path = current_file()
    text = path.read_text(errors="replace")
    cur = H.read_oem_file(text)
    prev = max((f for f in files if f.created < cur.created), key=lambda f: f.created, default=None)
    out = {"url": OEM_URL, "fetched": iso(datetime.fromtimestamp(path.stat().st_mtime, UTC)),
           "created": iso(cur.created), "sha256": hashlib.sha256(text.encode()).hexdigest(),
           "header_planned_burns": [{**e, "tig": iso(e["tig"])} for e in cur.planned if e["dv_ms"] > 0],
           "burns_modelled_in_forecast": [{**b, "after": iso(b["after"])} for b in H.burns_in_prediction(cur)],
           "note": ("NASA's header table and the forecast itself carry planned burns, so a 'prediction' read from them "
                    "would repeat NASA's plan, not test the detector. Only a detection between two published files "
                    "(a burn that already happened) is registered, and it states whether NASA had listed it.")}
    if prev is None:
        out["registered"] = None
        out["why_not"] = "no archived file older than the current one"
        return out
    days = (cur.start - prev.start).total_seconds() / 86400
    out["previous_file"] = {"created": iso(prev.created), "archive_capture": names[prev.created],
                            "gap_days": round(days, 2)}
    body = H.preregistration_body(prev, cur, noise, {"previous": f"Internet Archive capture {names[prev.created]}",
                                                     "current": OEM_URL})
    if body is None:
        out["registered"] = None
        out["why_not"] = (f"the gap to the newest archived file is {days:.1f} days, longer than the rule's "
                          f"{H.MAX_GAP_DAYS:.0f}" if days > H.MAX_GAP_DAYS else "no detection between the two files")
        return out
    pid = f"iss-burn-{cur.created:%Y%m%d}"
    try:
        out["registered"] = str(prereg.register("satellites", pid, body).relative_to(ROOT))
    except prereg.PreregError as e:
        out["registered"] = None
        out["why_not"] = str(e)
    return out


def main() -> None:
    if "--no-fetch" not in sys.argv:
        fetch_archive()
    files, names = load_archive()
    summaries = [H.file_summary(f) for f in files]
    gap_list = H.gaps(summaries)  # the detector has run: everything below is scoring against NASA's records
    executed, planned, aborted = references(files)
    every_burn = executed + planned
    noise = quiet_noise(gap_list, every_burn)
    scored = [g for g in gap_list if g["scored"]]
    # Records found only by looking up flagged gaps are kept apart: scoring with them is fair to the detector but
    # not blind, so the strict count of false alarms leaves them out.
    blind = [b for b in executed if not b.get("looked_up_after_detection")]
    strict = H.score(gap_list, blind, blind + planned)
    vs_records = H.score(gap_list, executed, every_burn)
    vs_plans = H.score(gap_list, planned, every_burn)
    doc = {
        "generated": "scripts/satellite_history.py",
        "rule": {"detect_dv_ms": H.DETECT_DV_MS, "max_gap_days": H.MAX_GAP_DAYS, "match_pad_days": H.MATCH_PAD_DAYS,
                 "drag_rate_days": H.DRAG_RATE_DAYS,
                 "statement": ("From each archived NASA ephemeris only the first orbit is used: its one-orbit mean "
                               "semi-major axis. Between consecutive files the change, minus the earlier file's own "
                               "drag decay, is a burn; |delta-v| >= 0.2 m/s in a gap of at most 7 days is a detection. "
                               "Fixed in code before the announcements were collected.")},
        "data": {"archive_captures": len(list(ARCHIVE.glob("oem_*.txt"))), "distinct_files": len(files),
                 "first": iso(files[0].created), "last": iso(files[-1].created),
                 "gaps": len(gap_list), "scored_gaps": len(scored),
                 "scored_days": round(sum(g["days"] for g in scored), 1),
                 "unscored_days": round(sum(g["days"] for g in gap_list if not g["scored"]), 1),
                 "quiet_dv_noise_ms": round(noise, 4)},
        "vs_nasa_records": public(vs_records),
        "strict_before_lookups": {k: public(strict[k]) for k in ("burns_covered", "detected", "recall", "recall_ci95",
                                  "quiet_days", "false_alarms", "false_alarms_per_30d", "false_alarms_per_30d_ci95",
                                  "false_alarm_gaps")},
        "vs_nasa_header_plans": public(vs_plans),
        "aborted_attempts": public(aborted),
        "dv_vs_nasa": dv_comparison(gap_list, executed),
        "sensitivity_after_the_fact": sensitivity(gap_list, executed, every_burn),
        "gaps": [gap_row(g) for g in gap_list],
        "current_file": forward_look(files, names, noise),
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    r, p = vs_records, vs_plans
    print(f"{len(files)} distinct NASA files {doc['data']['first'][:10]} to {doc['data']['last'][:10]}; "
          f"{len(scored)} scored gaps over {doc['data']['scored_days']} days; quiet noise {noise:.3f} m/s")
    for name, s in (("records, strict", strict), ("NASA records", r), ("header plans", p)):
        lo, hi = s["recall_ci95"]
        fa = s["false_alarms_per_30d_ci95"] or (float("nan"), float("nan"))
        print(f"{name}: {s['detected']}/{s['burns_covered']} detected (95% {lo:.2f} to {hi:.2f}); "
              f"{s['false_alarms']} false alarms in {s['quiet_days']:.0f} quiet days "
              f"({(s['false_alarms_per_30d'] or 0):.2f} per 30 d, 95% {fa[0]:.2f} to {fa[1]:.2f}); "
              f"timing median {s['timing_error_h_median']} h")
    print("current file:", doc["current_file"].get("registered") or doc["current_file"].get("why_not"))


if __name__ == "__main__":
    main()
