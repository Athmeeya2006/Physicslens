"""Physics-first machine faults: bearing defect rates, envelope diagnosis, spin-down law, machine-sound scores.

Synthetic signals with a known fault frequency; no network, no paid calls.
"""

import numpy as np
import pytest
from physicslens.physics import bearing, machine_sound, spindown

FS = 20000.0
GEOM = bearing.BearingGeometry(n_balls=8, ball_diameter_mm=6.77, pitch_diameter_mm=28.5)
SHAFT = 29.0


def test_defect_frequencies_obey_kinematic_identities():
    f = bearing.defect_frequencies(GEOM, SHAFT)
    # Every ball passes a point on the outer race once per cage turn, relative to the inner race likewise.
    assert f["outer"] + f["inner"] == pytest.approx(GEOM.n_balls * SHAFT)
    assert f["outer"] == pytest.approx(GEOM.n_balls * f["cage"])
    r = 6.77 / 28.5
    assert f["ball"] == pytest.approx(28.5 / (2 * 6.77) * SHAFT * (1 - r**2))
    assert f["outer"] == pytest.approx(88.44, abs=0.01)


@pytest.fixture(scope="module")
def rates():
    return bearing.defect_frequencies(GEOM, SHAFT)


@pytest.mark.parametrize("fault", ["outer", "inner", "ball", "cage"])
def test_envelope_diagnosis_finds_the_seeded_fault(fault, rates):
    x = bearing.synthetic_fault(FS, 6, rates[fault], shaft_hz=SHAFT, modulate_shaft=fault == "inner", seed=3)
    d = bearing.diagnose(x, FS, GEOM, SHAFT)
    assert d.fault == fault
    assert d.present[fault]


def test_healthy_signal_raises_no_alarm():
    x = bearing.synthetic_fault(FS, 6, 0.0, shaft_hz=SHAFT, impact=0.0, seed=4)
    assert bearing.diagnose(x, FS, GEOM, SHAFT).fault == "none"


def test_baseline_rule_flags_growth_and_ignores_a_series_already_in_the_baseline(rates):
    # The baseline already carries an outer-race series (another bearing of the same geometry on the shaft).
    base = bearing.synthetic_fault(FS, 6, rates["outer"], shaft_hz=SHAFT, seed=5)
    same = bearing.synthetic_fault(FS, 6, rates["outer"], shaft_hz=SHAFT, seed=6)
    d = bearing.diagnose_against_baseline(same, base, FS, GEOM, (27.0, 30.0), SHAFT)
    assert d.fault == "none"
    inner = same + bearing.synthetic_fault(FS, 6, rates["inner"], shaft_hz=SHAFT, modulate_shaft=True, noise=0.0,
                                           seed=7)
    d = bearing.diagnose_against_baseline(inner, base, FS, GEOM, (27.0, 30.0), SHAFT)
    assert d.fault == "inner"


def test_shaft_rate_is_read_from_the_vibration_not_the_tacho():
    t = np.arange(int(FS * 10)) / FS
    x = np.sin(2 * np.pi * 29.73 * t) + 0.3 * np.random.default_rng(0).standard_normal(len(t))
    assert bearing.shaft_rate_in_range(x, FS, 27.0, 30.0) == pytest.approx(29.73, abs=0.02)


def test_wilson_interval_matches_known_values():
    lo, hi = bearing.wilson(5, 5)
    assert lo == pytest.approx(0.566, abs=0.001) and hi == 1.0
    lo, hi = bearing.wilson(0, 20)
    assert lo == 0.0 and hi == pytest.approx(0.161, abs=0.001)


@pytest.mark.parametrize(("a", "b", "c", "law"), [(13.0, 0.08, 0.0, "coulomb+viscous"), (15.0, 0.0, 0.0, "coulomb")])
def test_spindown_recovers_drag_and_picks_the_needed_terms(a, b, c, law):
    t = np.arange(0, 5.0, 0.1)
    w = spindown.coast(t, 94.0, a, b, c) + np.random.default_rng(1).normal(0, 0.05, len(t))
    keep = w > 5
    fits = spindown.fit_spindown(t[keep], w[keep])
    chosen = spindown.select_law(fits)
    assert chosen.model == law
    assert chosen.params["a"] == pytest.approx(a, rel=0.03)
    if b:
        assert chosen.params["b"] == pytest.approx(b, rel=0.15)


def _machine(f0, unbalance=1.0, seed=0, fs=16000.0, seconds=4.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * seconds)) / fs
    x = 0.5 * rng.standard_normal(len(t))
    for k in range(1, 9):
        x += (unbalance if k == 1 else 1.0) * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 6.28)) / k
    return fs, x


def test_machine_sound_tracks_rotation_and_scores_physical_change_higher():
    refs = [machine_sound.spectrum(x, fs) for fs, x in (_machine(24.0, seed=s) for s in range(8))]
    f_ref = machine_sound.reference_rotation(refs, step_hz=0.1)
    assert f_ref == pytest.approx(24.0, abs=0.2)
    ref = machine_sound.robust_reference([machine_sound.clip_features(s, f_ref) for s in refs])
    normal = [machine_sound.anomaly_score(machine_sound.clip_features(machine_sound.spectrum(x, fs), f_ref), ref)
              for fs, x in (_machine(24.0, seed=s) for s in range(20, 26))]
    shifted = [machine_sound.anomaly_score(machine_sound.clip_features(machine_sound.spectrum(x, fs), f_ref), ref)
               for fs, x in (_machine(24.8, unbalance=4.0, seed=s) for s in range(30, 36))]
    auc, lo, hi = machine_sound.auc_delong(np.array(shifted), np.array(normal))
    assert auc == 1.0 and lo <= auc <= hi


def test_auc_delong_is_half_for_identical_scores():
    auc, _, _ = machine_sound.auc_delong(np.ones(10), np.ones(10))
    assert auc == 0.5
