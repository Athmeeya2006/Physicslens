"""Scenarios, the shared renderer, detectability, and free direct-track discovery (offline, from data/sky/)."""
import numpy as np
import pytest
from physicslens.sky import scenarios as S
from physicslens.sky.discover import SOURCE, discover
from physicslens.sky.ephem import sun_gm
from physicslens.sky.render import Drawn, Frame, detectability, project, render_frames


def test_every_scenario_loads_and_validates():
    assert len(S.names()) >= 11
    for name in S.names():
        sc = S.load(name)
        S.validate(sc)
        assert sc.description and (sc.baseline is None or sc.baseline in S.names())


def test_bad_scenarios_are_refused():
    with pytest.raises(S.ScenarioError, match="No scenario"):
        S.load("../../etc/passwd")
    sc = S.load("jupiter_x10")
    sc.mass_scale = {"Jupiter": 5000}
    with pytest.raises(S.ScenarioError, match="mass_scale"):
        S.validate(sc)
    sc = S.load("jupiter_x10")
    sc.clip_days = 400 * 365.25
    with pytest.raises(S.ScenarioError, match="300 years"):
        S.validate(sc)


def test_the_renderer_puts_each_dot_where_the_projection_says():
    basis = np.eye(3)
    pos = np.column_stack([np.linspace(-0.5, 0.5, 6), np.linspace(0.3, -0.2, 6), np.zeros(6)])
    px = project(pos, basis, 300.0, (200.0, 200.0))
    frames = render_frames([Drawn("dot", px, 8, (255, 255, 255))], Frame(400, 30, [""] * 6))
    assert len(frames) == 6
    for img, want in zip(frames, px, strict=True):
        g = img[:, :, 0].astype(float) - 18  # the background's blue channel
        g[g < 0] = 0
        g[:60] = 0  # the caption area
        ys, xs = np.indices(g.shape)
        # Pixel i covers [i, i+1), so a centroid in index units is half a pixel below the continuous coordinate.
        got = np.array([(g * xs).sum() / g.sum(), (g * ys).sum() / g.sum()]) + 0.5
        # Measured: 1/16 px, half a supersample (the drawn centre is rounded down at 8x). The plan asked for 0.5 px.
        assert np.linalg.norm(got - want) < 0.1


def test_detectability_says_what_a_video_can_and_cannot_see():
    sun_only = S.run(S.load("apophis_sun_only"))
    sc = S.load("apophis_sun_only")
    sc.bodies = ["Jupiter"]
    with_jupiter = S.run(sc)
    px = lambda r: (r.run.of("Apophis")[0][np.isin(r.run.jd, r.frames_jd)] @ sun_only.basis[:2].T) * 440.0
    real = detectability(px(sun_only), px(with_jupiter))
    assert not real.visible and "cannot see" in real.message
    heavy = S.run(S.load("jupiter_x100"))
    assert heavy.detectability.visible and "can see" in heavy.detectability.message


@pytest.mark.slow
def test_direct_track_discovery_recovers_the_suns_gm():
    sr = S.run(S.load("apophis_sun_only"))
    exact = discover(sr, noise_px=0.0)
    assert exact["source"] == SOURCE and exact["status"] == "accepted"
    mu = next(p for p in exact["parameters"] if p["name"] == "mu")
    assert abs(mu["value"] / sun_gm() - 1) < 1e-4
    noisy = discover(sr, noise_px=2.0)
    mu = next(p for p in noisy["parameters"] if p["name"] == "mu")
    assert mu["ci95"][0] <= sun_gm() <= mu["ci95"][1]


def test_scenarios_start_from_the_real_state_whatever_their_physics():
    # Sun-only and all-planet Apophis begin at the same JPL-seeded state: only the forces differ.
    a, b = S.build(S.load("apophis_sun_only")), S.build(S.load("apophis_with_planets"))
    state = lambda built: next(x.state for x in built[1] if x.name == "Apophis")
    np.testing.assert_allclose(state(a), state(b))
