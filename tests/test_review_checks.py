"""The checks added after the Nov 2 review: the Sun's return delay as a term, the Kepler-88 family's starting points,
and the fireball's blind landing regions. Each on small synthetic inputs with a known answer."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from physicslens.astro.ttv import spread_starts

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_return_delay_that_grows_with_latitude_is_found_by_leave_one_out():
    sun = _script("sun_demo")
    lats = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]
    rows = [{"lat": la, "error_days": 0.1 * la - 0.5} for la in lats]
    b = sun.return_bias(rows)
    assert b["late"] == 8 and b["sign_test_p"] == pytest.approx(2 / 256)
    assert b["best"] == "latitude" and b["loo_mae_days"]["latitude"] < 1e-9


def test_no_delay_is_kept_when_misses_go_both_ways():
    sun = _script("sun_demo")
    rows = [{"lat": la, "error_days": e} for la, e in zip([5, 10, 15, 20, 25, 30], [0.3, -0.3, 0.2, -0.2, 0.25, -0.25], strict=True)]
    b = sun.return_bias(rows)
    assert b["late"] == 3 and b["sign_test_p"] == 1.0
    assert b["loo_mae_days"]["none"] <= b["loo_mae_days"]["constant"]


def test_spread_starts_cover_the_period_window_and_stay_in_bounds():
    cand = {"period": 22.3}
    starts = spread_starts(cand, 10.95, 2455000.0, 40, seed=3)
    pc = np.array([s[4] for s in starts])
    assert len(starts) == 40
    assert pc.min() >= 22.3 * 0.97 and pc.max() <= 22.3 * 1.03
    assert np.ptp(pc) > 22.3 * 0.05  # stratified: the window is covered, not one corner of it
    for s in starts:
        assert math.hypot(s[2], s[3]) <= 0.2 + 1e-12 and math.hypot(s[6], s[7]) <= 0.2 + 1e-12
        assert 10 <= s[8] <= 1000


def test_blind_regions_put_a_central_find_inside_and_a_far_one_outside():
    fb = _script("fireball_demo")
    rng = np.random.default_rng(0)
    pts = [{"mass_kg": 0.1, "along": a, "across": c} for a, c in zip(rng.normal(5000, 800, 400), rng.normal(0, 60, 400), strict=True)]
    field = {"points": pts}
    finds = [{"mass_kg": 0.09, "along_m": 5000.0, "across_m": 0.0}, {"mass_kg": 0.12, "along_m": 5000.0, "across_m": 400.0}]
    orig = fb.along_cross
    fb.along_cross = lambda end, lat, lon: (lat, lon)  # points carry along/across directly in this test
    try:
        for p in pts:
            p["lat"], p["lon"] = p["along"], p["across"]
        fb.blind_contours(None, field, [0.05, 0.1, 0.2], finds)
    finally:
        fb.along_cross = orig
    assert finds[0]["blind"]["inside_68"] and finds[0]["blind"]["mass_bin_kg"] == 0.1
    assert not finds[1]["blind"]["inside_95"]


def test_status_ladder_climbs_only_on_recorded_evidence(monkeypatch):
    bc = _script("build_cases")
    from physicslens import prereg

    case = {"id": "timing", "status": "proven", "status_rule": "rule"}
    monkeypatch.setattr(prereg, "index_rows", lambda root=None: [])
    monkeypatch.setattr(bc, "_json", lambda rel: None)
    assert bc.ladder(case)["status"] == "proven"
    monkeypatch.setattr(prereg, "index_rows", lambda root=None: [{"case": "transit-timing", "id": "x"}])
    assert bc.ladder(case)["status"] == "predicted"
    files = {"outcomes/transit-timing/x.json": {"hits": 1}, "cases/timing/usage.json": {"users": [{"who": "an observatory"}]}}
    monkeypatch.setattr(bc, "_json", lambda rel: files.get(rel))
    r = bc.ladder(case)
    assert r["status"] == "solved" and r["checked"] == "proven" and [x["rung"] for x in r["ladder"]][-1] == "solved"
    # A partial case with no evidence beyond its check stays partial.
    monkeypatch.setattr(prereg, "index_rows", lambda root=None: [])
    monkeypatch.setattr(bc, "_json", lambda rel: None)
    assert bc.ladder({**case, "status": "partial"})["status"] == "partial"


def test_tie_headline_gives_a_95_percent_range_at_summary_precision():
    from physicslens.export.outcome import build_outcome

    result = {"status": "ambiguous", "family": "pendulum", "reason": "tie", "ranking": [], "agent": {},
              "agreed_parameters": [{"name": "g", "unit": "m/s^2", "value": 9.9073, "std": 0.0413, "ci95": [9.8242, 9.9862]}]}
    h = build_outcome(result)["headline"]
    assert "g = 9.91 m/s^2 (95%: 9.82 to 9.99)" in h and "±" not in h
