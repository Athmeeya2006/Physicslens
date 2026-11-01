"""Occultations: edge times from integrated frames (calibrated errors), the fundamental plane, and outlines from
chords with misses as constraints."""
import math

import numpy as np
import pytest
from physicslens.astro import occultation as O


def test_edges_from_integrated_frames_with_honest_errors():
    rng = np.random.default_rng(0)
    pulls = []
    for _ in range(20):
        td, tr = rng.uniform(10, 11), rng.uniform(15, 18)
        t = np.arange(0, 30, 0.13)
        f = O._model(t, 0.13, td, tr, 1.0, 0.1) + rng.normal(0, 0.08, len(t))
        e = O.edge_times(t, f, 0.13, 0.08)
        pulls += [(e["disappear"] - td) / e["disappear_sigma"], (e["reappear"] - tr) / e["reappear_sigma"]]
    assert 0.6 < np.std(pulls) < 1.6 and np.mean(np.abs(pulls) <= 2) > 0.85


def test_a_long_occultation_is_still_found():
    # The star hidden for most of the recording: the bright level is not the median.
    t = np.arange(0, 60, 0.5)
    f = O._model(t, 0.5, 5.2, 50.7, 1.0, 0.1)
    e = O.edge_times(t, f, 0.5, 0.02)
    assert e["disappear"] == pytest.approx(5.2, abs=0.05) and e["reappear"] == pytest.approx(50.7, abs=0.05)


def test_no_drop_is_a_negative_chord():
    with pytest.raises(ValueError):
        O.edge_times(np.arange(0, 10, 0.1), np.ones(100) + np.random.default_rng(1).normal(0, 0.01, 100), 0.1, 0.01)


def test_plane_axes_are_orthogonal_to_the_star():
    f, g = O.plane_axes(37.6, 14.3)
    a, d = math.radians(37.6), math.radians(14.3)
    s = np.array([math.cos(d) * math.cos(a), math.cos(d) * math.sin(a), math.sin(d)])
    assert abs(f @ s) < 1e-12 and abs(g @ s) < 1e-12 and abs(f @ g) < 1e-12 and g[2] > 0


def _chords(R=100.0, a=None, b=None, phi=0.0, n=9, seed=0):
    rng = np.random.default_rng(seed)
    a, b = a or R, b or R
    out = []
    for y in np.linspace(-0.85, 0.85, n) * b:
        # Horizontal chords across an ellipse centred at (5, -3).
        c, s = math.cos(phi), math.sin(phi)
        xs = np.linspace(-2 * a, 2 * a, 4001)
        X, Y = xs * c - y * s, xs * s + y * c  # rotate the chord line back into the ellipse's frame
        inside = (X / a) ** 2 + (Y / b) ** 2 <= 1
        if inside.any():
            x0, x1 = xs[inside][[0, -1]]
            pts = np.array([[x0, y], [x1, y]]) + [5, -3] + rng.normal(0, 0.5, (2, 2))
            out.append(O.Chord("p", True, pts, np.array([0.5, 0.5])))
    return out


def test_circle_radius_and_centre():
    f = O.fit_outline(_chords(R=100.0), "circle")
    assert f["params"]["radius"] == pytest.approx(100.0, abs=1.0)
    assert f["params"]["fc"] == pytest.approx(5.0, abs=1.0) and f["params"]["gc"] == pytest.approx(-3.0, abs=1.0)


def test_ellipse_axes():
    f = O.fit_outline(_chords(a=120.0, b=80.0, phi=0.0, n=15), "ellipse")
    assert f["params"]["a"] == pytest.approx(120.0, rel=0.03) and f["params"]["b"] == pytest.approx(80.0, rel=0.03)


def test_a_miss_bounds_the_outline():
    # Chords only across the lower half leave the top free; an observer just above the true edge saw nothing.
    rng = np.random.default_rng(3)
    chords = []
    for y in (-70.0, -50.0, -30.0):
        x = math.sqrt(100.0 ** 2 - y ** 2)
        pts = np.array([[-x, y], [x, y]]) + rng.normal(0, 2.0, (2, 2))
        chords.append(O.Chord("p", True, pts, np.array([2.0, 2.0])))
    miss = O.Chord("miss", False, np.column_stack([np.linspace(-200, 200, 80), np.full(80, 102.0)]), np.zeros(80))
    free = O.fit_outline(chords, "circle")
    bound = O.fit_outline([*chords, miss], "circle")
    assert bound["misses_crossed"] == 0 and bound["params"]["gc"] + bound["params"]["radius"] <= 102.0 + 3.0
    assert free["params"]["radius"] == pytest.approx(100.0, abs=25.0)


def test_a_short_event_in_a_long_recording():
    # A few seconds inside two minutes of video, the normal case for an asteroid: the dip must be found where it is.
    rng = np.random.default_rng(0)
    t = np.arange(0, 120, 0.1)
    f = O._model(t, 0.1, 60.03, 62.07, 1.0, 0.1) + rng.normal(0, 0.08, len(t))
    for sigma in (0.08, None):
        e = O.edge_times(t, f, 0.1, sigma)
        assert e["disappear"] == pytest.approx(60.03, abs=0.05) and e["reappear"] == pytest.approx(62.07, abs=0.05)
