"""Manoeuvre detection over the ISS's archived trajectory history, scored against NASA's own burn records.

Data: NASA's ISS ephemeris (CCSDS OEM) is republished every few days and only the current file is served, but the
Internet Archive kept a few hundred distinct copies (2021 to 2026). Each file starts from TOPO's best estimate of the
station's state at about the time it was made, then predicts 15 days ahead, planned burns included.

What the detector sees, and why it never reads the predictions:
- From each archived file it takes only the first orbit (the measured start, not the forecast) and turns it into one
  number: the semi-major axis averaged over exactly one orbit. Averaging removes J2's twice-per-orbit wobble (about
  10 km in the osculating value), leaving the mean orbit size to tens of metres.
- Drag only ever shrinks that number. Between two consecutive files the expected loss is the earlier file's own drag
  decay rate (the median orbit-to-orbit change over its first days: a median, so a planned burn inside that stretch
  cannot pull it). What is left over is a burn: delta-a = observed change minus expected drag loss, and for a nearly
  circular orbit delta-v = delta-a * v / (2 a).
- Timing is limited by the archive: a burn is placed somewhere between the two files, nothing finer. The reported
  time is the gap's midpoint and the honest uncertainty is half the gap.

The rule (fixed in this file before any announcement was looked at; the constants below are the whole rule):
- a gap counts as a detection when |delta-v| >= DETECT_DV_MS (both signs: some burns lower the orbit);
- only gaps up to MAX_GAP_DAYS are scored, because over longer gaps several burns can merge and the drag forecast
  error grows; longer gaps are reported as not covered, never as quiet;
- a detection matches a reference burn when the burn's ignition time falls inside the gap, widened by MATCH_PAD_DAYS
  on each side (the averaging orbit and the file's start are not at exactly the same instant).

References are kept apart from the detector: NASA's planned-burn tables in the same files' headers (the detector never
reads them) and dated public announcements, matched afterwards by `score`.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from physicslens.astro.satellites import MU_EARTH

DETECT_DV_MS = 0.2  # m/s: a fifth of a typical reboost, about 0.35 km of mean orbit size
MAX_GAP_DAYS = 7.0
MATCH_PAD_DAYS = 0.1
DRAG_RATE_DAYS = 3.0  # how much of each file sets its own drag decay rate
QUIET_UNIT_DAYS = 30.0


# --- reading one archived file -----------------------------------------------------------------------------------

@dataclass
class OemFile:
    created: datetime
    t_s: np.ndarray  # seconds since `start`
    start: datetime
    r: np.ndarray  # km
    v: np.ndarray  # km/s
    planned: list[dict[str, Any]] = field(default_factory=list)  # header burn table: reference only


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s.strip()[:23]).replace(tzinfo=UTC)


EVENT_LINE = re.compile(r"^COMMENT\s+(\S.{0,22}?)\s+(\d{3}):(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)\s+(?:\d+\s+)?(-?\d+\.\d+)")


def parse_planned(text: str, year: int, created: datetime) -> list[dict[str, Any]]:
    """The header's TRAJECTORY EVENT SUMMARY: event name, ignition time (day-of-year GMT) and delta-v in m/s.
    The table gives day of year only; a burn in early January listed by a December file belongs to the next year."""
    out = []
    for ln in text.splitlines():
        if not ln.startswith("COMMENT"):
            if out or ln[:2].isdigit():
                break
            continue
        m = EVENT_LINE.match(ln)
        if not m:
            continue
        name, doy, hh, mm, ss, dv = m.groups()
        y = year + 1 if created.month == 12 and int(doy) < 32 else year
        tig = datetime(y, 1, 1, tzinfo=UTC) + timedelta(days=int(doy) - 1, hours=int(hh), minutes=int(mm),
                                                         seconds=float(ss))
        out.append({"event": name.strip(), "tig": tig, "dv_ms": float(dv)})
    return out


def read_oem_file(text: str) -> OemFile:
    created = start = None
    rows = []
    for ln in text.splitlines():
        if ln.startswith("CREATION_DATE"):
            created = _utc(ln.split("=", 1)[1])
        elif ln.startswith("START_TIME"):
            start = _utc(ln.split("=", 1)[1])
        parts = ln.split()
        if len(parts) == 7 and parts[0][:2].isdigit() and "T" in parts[0]:
            rows.append(parts)
    if created is None or start is None or len(rows) < 50:
        raise ValueError("not a complete OEM ephemeris")
    t = np.array([(_utc(r[0]) - start).total_seconds() for r in rows])
    a = np.array([[float(x) for x in r[1:]] for r in rows])
    return OemFile(created, t, start, a[:, :3], a[:, 3:], parse_planned(text, created.year, created))


# --- the measured quantity ---------------------------------------------------------------------------------------

def osculating_a(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    return 1.0 / (2.0 / np.linalg.norm(r, axis=1) - np.sum(v * v, axis=1) / MU_EARTH)


def mean_a(t_s: np.ndarray, a: np.ndarray, t0: float) -> tuple[float, float]:
    """Osculating a averaged over exactly one orbit from t0 (the period from the mean itself, two passes), by the
    trapezoid rule on a finely interpolated curve. Returns (mean a in km, orbit period in s)."""
    period = 2 * math.pi * math.sqrt(float(np.mean(a[:30])) ** 3 / MU_EARTH)
    for _ in range(2):
        tt = np.linspace(t0, t0 + period, 721)
        m = float(np.trapezoid(np.interp(tt, t_s, a), tt) / period)
        period = 2 * math.pi * math.sqrt(m ** 3 / MU_EARTH)
    return m, period


def file_summary(f: OemFile) -> dict[str, Any]:
    """One archived file -> its measured mean orbit size at the start and its own drag decay rate (km/day)."""
    a = osculating_a(f.r, f.v)
    a0, period = mean_a(f.t_s, a, 0.0)
    n = int(DRAG_RATE_DAYS * 86400 / period)
    means = [mean_a(f.t_s, a, k * period)[0] for k in range(n + 1) if (k + 1) * period <= f.t_s[-1]]
    rate = float(np.median(np.diff(means))) * 86400 / period if len(means) > 2 else float("nan")
    speed = float(np.mean(np.linalg.norm(f.v[:30], axis=1)))
    return {"t": f.start + timedelta(seconds=period / 2), "created": f.created, "a_km": a0,
            "drag_km_per_day": rate, "speed_km_s": speed, "period_s": period}


# --- the detector ------------------------------------------------------------------------------------------------

def gaps(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Consecutive files -> one gap each, with the leftover change in orbit size turned into a delta-v."""
    s = sorted(summaries, key=lambda x: x["t"])
    out = []
    for p, q in itertools.pairwise(s):
        days = (q["t"] - p["t"]).total_seconds() / 86400
        if days <= 0.01:
            continue
        rate = p["drag_km_per_day"]
        da = q["a_km"] - p["a_km"] - (rate * days if math.isfinite(rate) else 0.0)
        dv = da * p["speed_km_s"] / (2 * p["a_km"]) * 1000.0
        out.append({"t0": p["t"], "t1": q["t"], "days": days, "observed_da_km": q["a_km"] - p["a_km"],
                    "drag_km_per_day": rate, "burn_da_km": da, "dv_ms": dv,
                    "scored": days <= MAX_GAP_DAYS and math.isfinite(rate),
                    "detected": days <= MAX_GAP_DAYS and math.isfinite(rate) and abs(dv) >= DETECT_DV_MS})
    return out


# --- scoring, after the fact -------------------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def poisson_rate_ci(k: int, exposure: float) -> tuple[float, float]:
    """Exact (Garwood) 95% interval for a Poisson count, per unit exposure."""
    from scipy.stats import chi2
    lo = 0.0 if k == 0 else chi2.ppf(0.025, 2 * k) / 2
    hi = chi2.ppf(0.975, 2 * k + 2) / 2
    return (lo / exposure, hi / exposure)


def _inside(t: datetime, g: dict[str, Any]) -> bool:
    pad = timedelta(days=MATCH_PAD_DAYS)
    return g["t0"] - pad <= t <= g["t1"] + pad


def gap_for(t: datetime, gap_list: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The one scored gap a burn belongs to: the gap that holds it outright, else one within the padding (so a burn
    near a boundary is never counted in two gaps)."""
    scored = [g for g in gap_list if g["scored"]]
    return (next((g for g in scored if g["t0"] <= t <= g["t1"]), None)
            or next((g for g in scored if _inside(t, g)), None))


def score(gap_list: list[dict[str, Any]], burns: list[dict[str, Any]],
          every_burn: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Recall over `burns` that fall in scored gaps; false alarms over gaps holding no burn of any kind
    (`every_burn`, default `burns`: a gap with a planned but unannounced burn is not quiet).
    A burn in an unscored (too long) gap or outside the archive is listed as not covered, never as a miss."""
    scored = [g for g in gap_list if g["scored"]]
    rows, hits = [], 0
    for b in burns:
        g = gap_for(b["tig"], gap_list)
        if g is None:
            rows.append({**b, "covered": False})
            continue
        mid = g["t0"] + (g["t1"] - g["t0"]) / 2
        hits += g["detected"]
        rows.append({**b, "covered": True, "detected": g["detected"], "gap_days": g["days"],
                     "estimated_dv_ms": g["dv_ms"], "timing_error_h": (mid - b["tig"]).total_seconds() / 3600,
                     "timing_bound_h": g["days"] * 12})
    n = sum(r["covered"] for r in rows)
    quiet = [g for g in scored if not any(_inside(b["tig"], g) for b in (every_burn or burns))]
    quiet_days = sum(g["days"] for g in quiet)
    fa = [g for g in quiet if g["detected"]]
    unit = quiet_days / QUIET_UNIT_DAYS
    errs = [abs(r["timing_error_h"]) for r in rows if r.get("detected")]
    return {"burns_covered": n, "detected": hits, "recall": hits / n if n else None, "recall_ci95": wilson(hits, n),
            "quiet_gaps": len(quiet), "quiet_days": quiet_days, "false_alarms": len(fa),
            "false_alarms_per_30d": len(fa) / unit if unit else None,
            "false_alarms_per_30d_ci95": poisson_rate_ci(len(fa), unit) if unit else None,
            "timing_error_h_median": float(np.median(errs)) if errs else None,
            "timing_error_h_max": float(np.max(errs)) if errs else None,
            "false_alarm_gaps": fa, "burns": rows}


SLIP_DAYS = 3.0
SLIP_DV_MS = 0.2


def slipped(e: dict[str, Any], nxt: OemFile | None) -> bool:
    """The next table moved this burn later (same size to within SLIP_DV_MS, at most SLIP_DAYS later): it is the
    same burn, taken from the newer table, not a cancelled one."""
    return nxt is not None and any(
        x["dv_ms"] > 0 and abs(x["dv_ms"] - e["dv_ms"]) <= SLIP_DV_MS
        and e["tig"] < x["tig"] <= e["tig"] + timedelta(days=SLIP_DAYS) for x in nxt.planned)


def planned_reference(files: list[OemFile]) -> list[dict[str, Any]]:
    """NASA's burn table as it stood around each burn: from every file, the burns whose ignition falls between that
    file's start and the next file's start (so a slipped burn is taken from the newest table, never twice; a file
    made just after a burn still lists it when its trajectory starts before it). A table entry is a plan, not proof:
    a burn the next table moves a little later is taken from that table (`slipped`); one cancelled after its last
    listing counts here as a miss, the conservative direction; one flown earlier than
    listed is missing here, so its gap looks quiet and a detection there counts as a false alarm until a dated
    record says otherwise."""
    fs = sorted(files, key=lambda f: f.start)
    out = []
    for f, nxt in zip(fs, [*fs[1:], None], strict=False):
        until = nxt.start if nxt else f.start + timedelta(days=15)
        # The table also lists dockings and launches with zero delta-v: only burns are references.
        out += [{**e, "listed_in": f.created} for e in f.planned
                if e["dv_ms"] > 0 and f.start < e["tig"] <= until and not slipped(e, nxt)]
    return sorted(out, key=lambda e: e["tig"])


# --- looking forward: what a new file says, and pre-registering a detection ---------------------------------------

def burns_in_prediction(f: OemFile) -> list[dict[str, Any]]:
    """Steps in a single file's predicted trajectory: TOPO models planned burns into the forecast, so a step here is
    NASA's plan read back out of the numbers, not a measurement (and never blind)."""
    a = osculating_a(f.r, f.v)
    _, period = mean_a(f.t_s, a, 0.0)
    ks = range(int(f.t_s[-1] // period))
    means = np.array([mean_a(f.t_s, a, k * period)[0] for k in ks])
    if len(means) < 3:
        return []
    # A burn inside orbit k smears across two orbit-to-orbit differences, so compare the orbits on either side of
    # it (k-1 and k+1) and keep only the strongest orbit of each step.
    drift = float(np.median(np.diff(means)))
    step = means[2:] - means[:-2] - 2 * drift
    speed = float(np.mean(np.linalg.norm(f.v, axis=1)))
    dv = step * speed / (2 * means[1:-1]) * 1000.0
    out = []
    for i, x in enumerate(dv):
        near = np.abs(dv[max(0, i - 1):i + 2])
        if abs(x) >= DETECT_DV_MS and abs(x) == near.max():
            out.append({"after": f.start + timedelta(seconds=float((i + 1) * period)), "dv_ms": float(x)})
    return out


def preregistration_body(prev: OemFile, new: OemFile, noise_ms: float, sources: dict[str, str]) -> dict[str, Any] | None:
    """A detection between two consecutive NASA files, written as a claim NASA's later records can confirm or not.
    Returns None when there is nothing to claim (no detection, or a gap too long for the rule). The body says
    whether NASA's own header already listed a burn in that window: if it did, the claim is not blind."""
    g = gaps([file_summary(prev), file_summary(new)])
    if not g or not g[0]["detected"]:
        return None
    g = g[0]
    listed = [e for f in (prev, new) for e in f.planned if e["dv_ms"] > 0 and _inside(e["tig"], g)]
    after = g["t1"] + timedelta(days=21)
    return {
        "title": f"ISS burn between {g['t0']:%Y-%m-%d %H:%M} and {g['t1']:%Y-%m-%d %H:%M} UTC",
        "question": "Did the ISS fire its engines in this window, and how hard?",
        "inputs": {"files": sources, "created": [prev.created.isoformat(), new.created.isoformat()],
                   "nasa_header_listed_burns": [{**e, "tig": e["tig"].isoformat()} for e in listed],
                   "blind": not listed},
        "prediction": {"burn_in_window": True, "window_utc": [g["t0"].isoformat(), g["t1"].isoformat()],
                       "dv_ms": round(g["dv_ms"], 3), "dv_ci95_ms": [round(g["dv_ms"] - 2 * noise_ms, 3),
                                                                     round(g["dv_ms"] + 2 * noise_ms, 3)]},
        "scoring": {"rule": "hit if a NASA record (PIMS ISS Handbook report, station blog or a later OEM header) dates "
                            "a burn inside the window; delta-v scored against the interval",
                    "detect_dv_ms": DETECT_DV_MS, "max_gap_days": MAX_GAP_DAYS,
                    "score_after_utc": after.strftime("%Y-%m-%dT%H:%M:%SZ")},
    }
