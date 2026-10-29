"""Laws across recordings: known scalings recovered from noisy synthetic points, honest ties and predictions."""

import math

import numpy as np
import pytest
from physicslens.physics.scaling import assess_scaling, check_prediction, next_setting

G = 9.81


def periods(lengths, rel_noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    t = [2 * math.pi * math.sqrt(L / G) for L in lengths]
    return [v * (1 + rng.normal(0, rel_noise)) for v in t], [v * rel_noise for v in t]


def test_pendulum_period_scales_with_root_length_and_predicts_the_next_length():
    lengths = [0.25, 0.4, 0.6, 0.9]
    y, s = periods(lengths)
    v = assess_scaling(lengths, y, s)
    assert v.status == "accepted" and v.winner.law == "power_0.5", v.reason
    a = v.winner.params[0]
    assert a == pytest.approx(2 * math.pi / math.sqrt(G), rel=0.02)  # the prefactor gives g back
    pred, std = v.winner.predict(1.2)
    truth = 2 * math.pi * math.sqrt(1.2 / G)
    assert abs(pred - truth) < Z95_STD * std + 0.02 * truth
    assert check_prediction(pred, std, truth, 0.01 * truth)["within_95"]


Z95_STD = 1.96


def test_spring_mass_sweep_finds_root_mass():
    masses = [0.1, 0.2, 0.35, 0.5]
    k = 40.0
    t = [2 * math.pi * math.sqrt(m / k) for m in masses]
    v = assess_scaling(masses, t, [0.01 * x for x in t])
    assert v.status == "accepted" and v.winner.law == "power_0.5"


def test_constant_quantity_is_found_as_no_dependence():
    # g measured at four lengths should not depend on length at all.
    v = assess_scaling([0.25, 0.4, 0.6, 0.9], [9.80, 9.83, 9.79, 9.82], [0.03] * 4)
    assert v.status == "accepted" and v.winner.law == "constant"


def test_two_points_are_too_few_and_a_narrow_range_is_a_tie():
    assert assess_scaling([0.4, 0.5], [1.27, 1.42], [0.01, 0.01]).status == "too_few"
    # Over a tiny range a square-root law and a straight line both predict every point.
    lengths = [0.50, 0.51, 0.52]
    y, s = periods(lengths, rel_noise=0.02)
    v = assess_scaling(lengths, y, s)
    assert v.status == "ambiguous"
    x_next = next_setting(v, lengths)
    assert x_next is not None and (x_next < 0.3 or x_next > 0.8)  # it asks for a setting far from the cluster


def test_no_law_fits_scattered_points():
    v = assess_scaling([0.2, 0.4, 0.6, 0.8, 1.0], [1.0, 3.0, 0.5, 4.0, 0.2], [0.01] * 5)
    assert v.status == "no_fit" and v.winner is None


def test_prediction_check_flags_a_miss():
    assert not check_prediction(2.0, 0.02, 2.2, 0.02)["within_95"]
