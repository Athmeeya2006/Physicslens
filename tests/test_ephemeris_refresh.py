"""Refreshing a stale transit ephemeris: recovers a drifted period, its held-out rule, intervals that cover truth."""
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from physicslens import prereg
from physicslens.astro import ephemeris as E
from physicslens.astro.timing import HELDOUT_CHI2

T0, P = 2458500.0, 3.5
ERR = 0.0007  # 1 minute, a TESS-like mid-time error


def tess_like(rng, truth_p, start=2459000.0, n_seasons=3, gap=700.0, per_season=7, decay=0.0):
    """Mid-times in a few 27-day seasons years apart, scattered by their stated errors."""
    rows = []
    for s in range(n_seasons):
        t_start = start + s * gap
        n = np.arange(np.ceil((t_start - T0) / truth_p), np.ceil((t_start - T0) / truth_p) + per_season)
        t = T0 + n * truth_p + 0.5 * decay * n**2 + rng.normal(0, ERR, len(n))
        rows += list(zip(n.astype(int), t, strict=True))
    ep, t = np.array(rows).T
    return ep.astype(int), t, np.full(len(t), ERR)


def test_refresh_recovers_a_drifted_period():
    rng = np.random.default_rng(1)
    truth_p = P + 2e-5  # the published period is stale by 2e-5 d: 1.7 h off after 300 orbits
    pub = E.Published(T0, 0.002, P, 1e-5)  # truth is 2 sigma from the published period
    ep, t, e = tess_like(rng, truth_p)
    assert (E.assign_epochs(t, e, pub) == ep).all()
    fit = E.refresh(ep, t, e, pub)
    s = fit.summary()
    assert abs(fit.period - truth_p) < 3 * s["period_err_days"]
    assert s["period_err_days"] < 0.1 * pub.period_err  # the new times dominate
    tf, sf = fit.predict(np.array([1000]))
    assert abs(tf[0] - (T0 + 1000 * truth_p)) < 3 * sf[0]


def test_a_contradicted_prior_is_dropped_and_said_so():
    rng = np.random.default_rng(2)
    ep, t, e = tess_like(rng, P + 2e-4)
    fit = E.refresh(ep, t, e, E.Published(T0, 0.0005, P, 1e-6))  # claims to be 200x more precise than it is
    assert not fit.used_prior and fit.prior_tension_sigma > E.PRIOR_TENSION_MAX and "not combined" in fit.prior_note


def test_heldout_passes_for_a_linear_truth_and_fails_for_a_decaying_one():
    rng = np.random.default_rng(3)
    pub = E.Published(T0, 0.002, P, 5e-5)
    ok = E.heldout(*tess_like(rng, P + 1e-5), pub)
    assert ok["passed"] and ok["chi2_per_transit"] <= HELDOUT_CHI2 and ok["n_test"] == 7
    # A period shrinking fast enough to bend the last season away from the straight line through the earlier ones.
    bad = E.heldout(*tess_like(rng, P, decay=-1e-7), pub)
    assert not bad["passed"] and bad["chi2_per_transit"] > HELDOUT_CHI2


def test_heldout_needs_two_seasons():
    rng = np.random.default_rng(4)
    r = E.heldout(*tess_like(rng, P, n_seasons=1), E.Published(T0, 0.002, P, 5e-5))
    assert not r["passed"] and "one observing season" in r["reason"]


def test_windows_cover_truth_at_their_stated_rates():
    rng = np.random.default_rng(5)
    truth_p = P + 1e-5
    pub = E.Published(T0, 0.002, P, 3e-5)
    in68 = in95 = total = 0
    for _ in range(300):
        ep, t, e = tess_like(rng, truth_p, per_season=5)
        fit = E.refresh(ep, t, e, pub)
        w = E.windows(fit, pub, 2461360.0, 2461370.0)[0]
        truth = T0 + w["epoch"] * truth_p
        in68 += w["ci68"][0] <= truth <= w["ci68"][1]
        in95 += w["ci95"][0] <= truth <= w["ci95"][1]
        total += 1
    assert 0.60 <= in68 / total <= 0.76 and 0.91 <= in95 / total <= 0.99
    assert abs(w["shift_min"] - (w["value"] - w["published_ephemeris_value"]) * 1440) < 1e-9


def test_poorly_fitted_transits_are_dropped_with_a_reason():
    rows = [{"bjd_tdb": float(i), "chi2_red": c} for i, c in enumerate([0.9, 1.0, 1.1, 0.95, 12.8])]
    keep, bad = E.well_fitted(rows)
    assert len(keep) == 4 and len(bad) == 1 and "12.8" in bad[0]["reason"]
    noisy = [{"bjd_tdb": float(i), "chi2_red": c} for i, c in enumerate([2.5, 2.8, 3.2, 2.6])]
    assert len(E.well_fitted(noisy)[0]) == 4  # uniformly noisy light curves are not thrown away


def test_a_glitched_mid_time_is_caught_within_its_own_season():
    rng = np.random.default_rng(6)
    ep, t, e = tess_like(rng, P)
    t[9] += 30 / 1440  # a box search that locked onto a systematic: half an hour off
    bad = E.season_outliers(ep, t, e)
    assert bad[9] and bad.sum() == 1
    # A real period change bends seasons apart but is smooth inside each: nothing flagged.
    assert not E.season_outliers(*tess_like(rng, P, decay=-1e-7)).any()


def test_score_transit_widens_by_the_measurement_error():
    pred = {"label": "transit N=1", "value": 100.0, "sigma_days": 0.0001, "ci95": [99.999804, 100.000196]}
    near = E.score_transit(pred, 100.001, 0.001)  # 1.4 min off, measured to 1.4 min: a hit, though outside ci95
    assert near["hit"] and not near["inside_registered_ci95"]
    assert not E.score_transit(pred, 100.01, 0.001)["hit"]


def _scorer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "score_predictions.py"
    spec = importlib.util.spec_from_file_location("score_predictions", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["score_predictions"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_scorer_scores_only_after_the_date_with_synthetic_measurements(tmp_path):
    items = [{"label": f"transit N={n}", "epoch": n, "value": 2461400.0 + n, "sigma_days": 0.0005,
              "ci68": [2461400.0 + n - 0.0005, 2461400.0 + n + 0.0005],
              "ci95": [2461400.0 + n - 0.00098, 2461400.0 + n + 0.00098], "unit": "BJD_TDB"} for n in (1, 2, 3)]
    body = {"title": "t", "question": "q", "inputs": [], "prediction": {"kind": "transit_times", "items": items},
            "scoring": {"rule": "r", "score_after_utc": "2027-01-01T00:00:00Z", "sources_to_check": []}}
    preds, outs = tmp_path / "predictions", tmp_path / "outcomes"
    prereg.register("transit-timing", "x-b-2026-11", body, root=preds, now=datetime(2026, 11, 2, tzinfo=UTC))
    sc = _scorer()
    measured = [{"bjd_tdb": 2461401.0003, "err_days": 0.0005, "source": "synthetic"},
                {"bjd_tdb": 2461402.01, "err_days": 0.0005, "source": "synthetic"}]
    early = sc.score_all(preds, outs, now=datetime(2026, 12, 1, tzinfo=UTC), fetch=lambda doc: measured)
    assert early == [] and not outs.exists()
    done = sc.score_all(preds, outs, now=datetime(2027, 2, 1, tzinfo=UTC), fetch=lambda doc: measured)
    out = json.loads(done[0].read_text())
    assert out["hits"] == 1 and out["misses"] == 1 and out["unmeasured"] == 1
    assert prereg.verify(preds) == []  # scoring never touched the prediction
    with pytest.raises(prereg.PreregError):
        prereg.score("transit-timing", "x-b-2026-11", out, root=preds, outcomes=outs, now=datetime(2026, 12, 1, tzinfo=UTC))
