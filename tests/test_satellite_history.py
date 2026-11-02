"""The ISS history detector on synthetic histories: a known impulse is found in the right gap with the right size,
a quiet stretch raises no alarm, and the scoring and pre-registration bookkeeping behave."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta

import numpy as np
from physicslens import prereg
from physicslens.astro import sat_history as H
from physicslens.astro.satellites import MU_EARTH

T0 = datetime(2023, 3, 1, 12, tzinfo=UTC)
A0 = 6790.0  # km
DECAY = 0.08  # km/day, a typical ISS drag loss


def size(t_days: float, burns: list[tuple[float, float]]) -> float:
    """Orbit size with steady drag plus impulses (day, delta-v m/s): delta-a = 2 a delta-v / v."""
    a = A0 - DECAY * t_days
    for day, dv in burns:
        if t_days >= day:
            a += 2 * A0 * dv / 1000 / math.sqrt(MU_EARTH / A0)
    return a


def synth_file(start_day: float, burns: list[tuple[float, float]], planned: list | None = None,
               span_days: float = 4.0, step_s: float = 240.0) -> H.OemFile:
    """A circular orbit whose radius follows `size`: the osculating a is the radius, so the file carries exactly the
    history we put in (J2's wobble is absent; the averaging is tested on it separately)."""
    t = np.arange(0.0, span_days * 86400, step_s)
    a = np.array([size(start_day + s / 86400, burns) for s in t])
    n = np.sqrt(MU_EARTH / a**3)
    ph = np.cumsum(np.r_[0, n[:-1] * step_s])
    r = np.c_[a * np.cos(ph), a * np.sin(ph), np.zeros_like(a)]
    v = np.c_[-np.sin(ph), np.cos(ph), np.zeros_like(a)] * np.sqrt(MU_EARTH / a)[:, None]
    start = T0 + timedelta(days=start_day)
    return H.OemFile(created=start + timedelta(hours=3), t_s=t, start=start, r=r, v=v, planned=planned or [])


def history(burns, every=2.0, days=20.0):
    files = [synth_file(d, burns) for d in np.arange(0.0, days, every)]
    return files, H.gaps([H.file_summary(f) for f in files])


def test_known_impulse_found_in_the_right_gap_with_its_size():
    _, gl = history([(7.3, 0.5)])
    hits = [g for g in gl if g["detected"]]
    assert len(hits) == 1
    g = hits[0]
    assert g["t0"] <= T0 + timedelta(days=7.3) <= g["t1"]
    assert abs(g["dv_ms"] - 0.5) < 0.03
    assert abs(g["drag_km_per_day"] + DECAY) < 0.005


def test_quiet_stretch_raises_no_alarm():
    _, gl = history([])
    assert all(g["scored"] for g in gl)
    assert not any(g["detected"] for g in gl)
    assert max(abs(g["dv_ms"]) for g in gl) < 0.02


def test_lowering_burn_is_found_too():
    _, gl = history([(11.0, -0.4)])
    hits = [g for g in gl if g["detected"]]
    assert len(hits) == 1 and hits[0]["dv_ms"] < -0.35


def test_below_threshold_burn_is_not_claimed():
    _, gl = history([(9.0, 0.1)])
    assert not any(g["detected"] for g in gl)


def test_long_gap_is_not_scored():
    files = [synth_file(0.0, []), synth_file(10.0, [(5.0, 1.0)], span_days=4.0)]
    g = H.gaps([H.file_summary(f) for f in files])[0]
    assert g["days"] > H.MAX_GAP_DAYS and not g["scored"] and not g["detected"]


def test_mean_size_removes_a_twice_per_orbit_wobble():
    t = np.arange(0, 3 * 86400, 240.0)
    period = 2 * math.pi * math.sqrt(A0**3 / MU_EARTH)
    a = A0 + 8.0 * np.sin(4 * math.pi * t / period + 0.3)  # J2-like, 8 km amplitude
    m, p = H.mean_a(t, a, 0.0)
    assert abs(m - A0) < 0.05 and abs(p - period) < 1.0


def test_score_counts_hits_misses_and_false_alarms():
    _, gl = history([(7.3, 0.5), (15.1, 0.05)])
    burns = [{"event": "boost", "tig": T0 + timedelta(days=7.3)},
             {"event": "tiny", "tig": T0 + timedelta(days=15.1)},
             {"event": "outside", "tig": T0 + timedelta(days=90)}]
    s = H.score(gl, burns)
    assert s["burns_covered"] == 2 and s["detected"] == 1 and s["false_alarms"] == 0
    assert [b["covered"] for b in s["burns"]] == [True, True, False]
    lo, hi = s["recall_ci95"]
    assert 0.0 < lo < 0.5 < hi < 1.0
    # With no reference at all, the real burn becomes a false alarm in otherwise quiet data.
    s2 = H.score(gl, [])
    assert s2["false_alarms"] == 1 and s2["quiet_days"] > 15


def test_wilson_and_poisson_intervals():
    lo, hi = H.wilson(10, 10)
    assert hi > 0.9999 and 0.69 < lo < 0.73
    lo, hi = H.poisson_rate_ci(0, 10.0)
    assert lo == 0.0 and 0.36 < hi < 0.38  # 3.69 / 10


HEADER = """CCSDS_OEM_VERS = 2.0
CREATION_DATE  = 2022-12-30T18:44:17.495
ORIGINATOR     = NASA/JSC/FOD/TOPO
START_TIME           = 2022-12-30T12:00:00.000
COMMENT Begin sequence of events
COMMENT TRAJECTORY EVENT SUMMARY:
COMMENT |       EVENT        |       TIG        | ORB |   DV    |   HA    |   HP    |
COMMENT =============================================================================
COMMENT  GMT 365 Reboost       365:02:38:00.000             0.8     421.7     410.3
COMMENT                                                    (2.6)   (227.7)   (221.6)
COMMENT  82P Dock              003:02:51:56.000             0.0     425.8     410.2
COMMENT                                                    (0.0)   (229.9)   (221.5)
COMMENT End sequence of events
"""


def test_header_burn_table_is_parsed_with_year_rollover():
    created = datetime(2022, 12, 30, 18, 44, tzinfo=UTC)
    ev = H.parse_planned(HEADER, 2022, created)
    assert [e["event"] for e in ev] == ["GMT 365 Reboost", "82P Dock"]
    assert ev[0]["tig"] == datetime(2022, 12, 31, 2, 38, tzinfo=UTC) and ev[0]["dv_ms"] == 0.8
    assert ev[1]["tig"].year == 2023 and ev[1]["dv_ms"] == 0.0


def test_planned_reference_keeps_burns_only_and_the_newest_listing():
    tig = T0 + timedelta(days=3.5)
    a = synth_file(0.0, [], planned=[{"event": "GMT Reboost Preli", "tig": tig, "dv_ms": 0.5},
                                     {"event": "Dock", "tig": tig, "dv_ms": 0.0}])
    b = synth_file(2.0, [], planned=[{"event": "GMT Reboost", "tig": tig + timedelta(hours=1), "dv_ms": 0.6}])
    ref = H.planned_reference([a, b])
    assert len(ref) == 1 and ref[0]["event"] == "GMT Reboost" and ref[0]["dv_ms"] == 0.6


def test_a_burn_moved_later_by_the_next_table_is_one_burn_not_a_miss():
    a = synth_file(0.0, [], planned=[{"event": "GMT032 Reboost", "tig": T0 + timedelta(days=1.9), "dv_ms": 1.4}])
    b = synth_file(2.0, [], planned=[{"event": "GMT034 Reboost", "tig": T0 + timedelta(days=3.9), "dv_ms": 1.4}])
    ref = H.planned_reference([a, b])
    assert [e["event"] for e in ref] == ["GMT034 Reboost"]


def test_a_burn_near_a_boundary_belongs_to_one_gap():
    _, gl = history([])
    t = gl[2]["t1"] - timedelta(hours=1)
    assert H.gap_for(t, gl) is gl[2]


def test_burn_modelled_in_a_forecast_is_read_back():
    f = synth_file(0.0, [(2.2, 0.8)])
    found = H.burns_in_prediction(f)
    assert len(found) == 1 and abs(found[0]["dv_ms"] - 0.8) < 0.05
    assert abs((found[0]["after"] - (T0 + timedelta(days=2.2))).total_seconds()) < 2 * 5600
    assert H.burns_in_prediction(synth_file(0.0, [])) == []


def test_preregistration_only_for_a_detection_and_says_if_blind(tmp_path):
    quiet = H.preregistration_body(synth_file(0.0, []), synth_file(2.0, []), 0.04, {})
    assert quiet is None
    tig = T0 + timedelta(days=1.0)
    prev = synth_file(0.0, [(1.0, 0.7)], planned=[{"event": "Reboost", "tig": tig, "dv_ms": 0.7}])
    new = synth_file(2.0, [(1.0, 0.7)])
    body = H.preregistration_body(prev, new, 0.04, {"previous": "a", "current": "b"})
    assert body is not None and body["inputs"]["blind"] is False
    assert body["prediction"]["dv_ci95_ms"][0] < 0.7 < body["prediction"]["dv_ci95_ms"][1]
    path = prereg.register("satellites", "iss-burn-test", body, root=tmp_path)
    doc = json.loads(path.read_text())
    assert doc["case"] == "satellites" and doc["status"] == "open"
    blind = H.preregistration_body(synth_file(0.0, [(1.0, 0.7)]), new, 0.04, {})
    assert blind["inputs"]["blind"] is True
