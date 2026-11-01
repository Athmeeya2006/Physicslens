"""Transit timing: real WASP-12b decay against the published rate, a stable system that must come out "nothing
happening", a hidden companion's periodic signal, transit windows for observers, and per-transit mid-times."""
import csv
import math
from pathlib import Path

import numpy as np
import pytest
from physicslens.astro import timing
from physicslens.astro.transits import Shape, _batman, transit_times

ROOT = Path(__file__).resolve().parents[1]


def _synthetic(law, theta, n_events=120, err_min=0.6, seed=0, span=1500):
    rng = np.random.default_rng(seed)
    epoch = np.sort(rng.choice(np.arange(-span // 2, span // 2), n_events, replace=False))
    occ = np.zeros(n_events, bool)
    err = np.full(n_events, err_min / 1440)
    t = timing.LAWS[law].model(np.array(theta), epoch, occ) + rng.normal(0, err)
    return timing.Timings(epoch, t, err, occ)


def test_wasp12b_decays_at_the_published_rate():
    with open(ROOT / "data" / "astro" / "timing" / "wasp12b_yee2020.csv") as fh:
        rows = list(csv.DictReader(fh))
    r = timing.assess(timing.Timings.from_rows(rows))
    assert r["status"] == "accepted" and r["winner"].law.id == "decay"
    rate, sigma = timing.period_change_ms_per_year(r["winner"])
    assert abs(rate - (-29.0)) < 2 * math.hypot(sigma, 2.0)  # Yee et al. 2020: -29 +- 2 ms per year


def test_a_stable_system_is_nothing_happening():
    d = _synthetic("linear", [2458000.0, 3.5])
    r = timing.assess(d)
    assert r["status"] in ("accepted", "ambiguous") and r["winner"].law.id == "linear"


def test_a_companions_periodic_signal_is_found():
    d = _synthetic("sinusoid", [2455000.0, 10.95, 0.25, 57.5, 1.0], n_events=110, span=140)
    r = timing.assess(d, laws=("linear", "decay", "sinusoid"))
    assert r["status"] == "accepted" and r["winner"].law.id == "sinusoid"
    assert r["winner"].params["Nttv"] == pytest.approx(57.5, rel=0.05)


def test_transit_windows_for_observers():
    d = _synthetic("linear", [2458000.0, 3.5])
    fit = timing.fit_law(timing.LAWS["linear"], d, timing.linear_start(d))
    rows = timing.next_windows(fit, 2459000.0, 5)
    assert len(rows) == 5 and rows[0]["bjd_tdb"] > 2459000.0 and rows[0]["sigma_min"] > 0
    ics = timing.windows_ics("Test b", rows)
    assert ics.count("BEGIN:VEVENT") == 5 and ics.startswith("BEGIN:VCALENDAR")
    assert timing.windows_csv("Test b", rows).count("\n") == 6


def test_ephemeris_rot():
    assert timing.ephemeris_sigma_minutes(0.0, 1 / 1440, 1.0, 1e-5, 1000.0) == pytest.approx(math.hypot(1, 14.4))
    rows = [{"pl_name": "a", "pl_tranmid": "2455000", "pl_tranmiderr1": "0.001", "pl_tranmiderr2": "-0.001",
             "pl_orbper": "3", "pl_orbpererr1": "0.0001", "pl_orbpererr2": "-0.0001", "sy_vmag": "10"},
            {"pl_name": "b", "pl_tranmid": "2455000", "pl_tranmiderr1": "0.001", "pl_tranmiderr2": "-0.001",
             "pl_orbper": "3", "pl_orbpererr1": "0.000001", "pl_orbpererr2": "-0.000001", "sy_vmag": ""}]
    rot = timing.ephemeris_rot(rows, 2460000.0)
    assert [r["planet"] for r in rot] == ["a", "b"] and rot[0]["over_1_h"] and not rot[1]["over_10_min"]


def test_mid_times_from_a_light_curve_with_ttvs():
    rng = np.random.default_rng(4)
    shape = Shape(10.95, 0.035, 25.0, 89.6, 0.4, 0.3)
    t = np.arange(0, 220, 0.0204336)
    ttv = 0.15 * np.sin(2 * np.pi * np.arange(25) / 8.0)  # up to 3.6 hours early or late
    f = np.ones_like(t)
    for n in range(25):
        mid = 5.0 + n * 10.95 + ttv[n]
        near = np.abs(t - mid) < 0.5
        f[near] *= _batman(t[near], mid, shape, 0.0204336)
    f += rng.normal(0, 2e-4, len(t))
    out = transit_times(t, f, np.full(len(t), 2e-4), 5.0, 10.95, 0.25, search_half_width=0.3)
    got = {r["epoch"]: r for r in out["rows"]}
    assert len(got) >= 18
    pulls = [(got[n]["bjd_tdb"] - (5.0 + n * 10.95 + ttv[n])) / got[n]["err_days"] for n in got]
    assert np.mean(np.abs(pulls) < 4) > 0.9


def test_companion_candidates_invert_the_super_period():
    from physicslens.astro.ttv import companion_candidates, super_period
    p_b, p_ttv = 10.95, 600.0
    for c in companion_candidates(p_b, p_ttv):
        inner, outer = (p_b, c["period"]) if c["where"] == "outside" else (c["period"], p_b)
        assert super_period(inner, outer, c["j"]) == pytest.approx(p_ttv, rel=1e-9)
    assert any(c["resonance"] == "2:1" and c["where"] == "outside" and 22.0 < c["period"] < 22.5
               for c in companion_candidates(p_b, p_ttv))


def test_nbody_transits_are_linear_without_a_companion_and_wobble_with_one():
    from physicslens.astro.ttv import Setup, nbody_times
    ep = np.arange(0, 80)
    s = Setup(1.0, 0)
    quiet = nbody_times(np.array([2455000.0, 10.95, 0.0, 0.0, 22.3, 1.0, 0.0, 0.0, 1e-6]), ep, s)
    c = np.polyfit(ep, quiet, 1)
    assert np.max(np.abs(quiet - np.polyval(c, ep))) * 1440 < 0.5  # under half a minute: a clean clock
    loud = nbody_times(np.array([2455000.0, 10.95, 0.0, 0.05, 22.3, 1.0, 0.0, -0.05, 200.0]), ep, s)
    c = np.polyfit(ep, loud, 1)
    assert np.ptp(loud - np.polyval(c, ep)) * 1440 > 120  # a Saturn-mass neighbour near 2:1: hours of variation
