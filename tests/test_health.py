"""Machine-health drift between two recordings: real changes flagged, re-recording noise never."""

import functools
import json

import numpy as np
import pytest
from physicslens.physics import health as health_module
from physicslens.physics.health import compare

# Fewer bootstrap samples than the app (20) keep the suite fast; the flagging logic is unchanged.
health_metrics = functools.partial(health_module.health_metrics, reps=8)


@pytest.fixture(autouse=True)
def _fast_endpoint_bootstrap(monkeypatch):
    from physicslens import app as app_module
    monkeypatch.setattr(app_module, "health_metrics", health_metrics)

from .test_regime import FPS, pendulum


@pytest.fixture(scope="module")
def baseline():
    return health_metrics(pendulum([(10, 0.6, 0.05)], seed=11), "pendulum")


def drift(baseline, segments, theta0=0.6, seed=12):
    return compare(baseline, health_metrics(pendulum(segments, theta0=theta0, seed=seed), "pendulum"))


@pytest.mark.parametrize("theta0", [0.6, 0.35])
def test_rerecording_the_same_system_raises_no_alarm(baseline, theta0):
    assert not drift(baseline, [(10, 0.6, 0.05)], theta0=theta0)["drift"]


def test_more_damping_is_flagged(baseline):
    r = drift(baseline, [(10, 0.6, 0.075)])  # +50%: an early sign of wear
    decay = next(m for m in r["metrics"] if m["metric"] == "decay_rate_per_s")
    assert r["drift"] and decay["flagged"] and decay["change_pct"] == pytest.approx(50, abs=5)
    assert "damping up" in r["summary"]


def test_frequency_shift_is_flagged(baseline):
    r = drift(baseline, [(10, 0.624, 0.05)])  # 4% longer: frequency -2%
    freq = next(m for m in r["metrics"] if m["metric"] == "natural_freq_hz")
    assert r["drift"] and freq["flagged"] and freq["change_pct"] == pytest.approx(-1.94, abs=0.2)


def test_changes_inside_repeatability_are_not_flagged(baseline):
    assert not drift(baseline, [(10, 0.606, 0.05)])["drift"]  # frequency -0.5%: within re-recording spread


def stored_run(app_module, traj, family="pendulum") -> str:
    """A finished run on disk, shaped like a real analysis result (pixel track, fps, scale)."""
    ppm = 800.0
    px = np.column_stack([400 + traj.xy[:, 0] * ppm, 100 - (traj.xy[:, 1] - traj.xy[0, 1]) * ppm])
    run_dir = app_module.runs.new_run_dir()
    result = {"status": "accepted", "family": family, "clip": {"fps": FPS, "frames": len(traj)},
              "calibration": {"px_per_m": ppm}, "scene": {"moving_object": "metal ball"},
              "real_track_px": [[int(f), float(x), float(y)] for f, (x, y) in zip(traj.frames, px, strict=True)]}
    (run_dir / "result.json").write_text(json.dumps(result))
    return run_dir.name


def test_compare_endpoint():
    from fastapi.testclient import TestClient
    from physicslens import app as app_module

    a = stored_run(app_module, pendulum([(8, 0.6, 0.05)], seed=21))
    b = stored_run(app_module, pendulum([(8, 0.6, 0.15)], seed=22))
    r = TestClient(app_module.app).post("/api/compare", json={"baseline": a, "current": b})
    body = r.json()
    assert r.status_code == 200 and body["drift"] and body["runs"]["baseline"]["id"] == a


def test_trend_catches_slowly_rising_damping():
    from physicslens.physics.health import trend
    series = [health_metrics(pendulum([(8, 0.6, g)], seed=30 + i), "pendulum") for i, g in enumerate([0.05, 0.06, 0.07, 0.08, 0.09])]
    out = trend(series)
    d = out["metrics"]["decay_rate_per_s"]
    assert out["drift"] and d["steady_change"] and d["slope_per_recording"] > 0
    assert not out["metrics"]["natural_freq_hz"]["steady_change"]
    assert "damping rising" in out["summary"]


def test_no_trend_when_nothing_changes():
    from physicslens.physics.health import trend
    series = [health_metrics(pendulum([(8, 0.6, 0.05)], seed=40 + i), "pendulum") for i in range(4)]
    assert not trend(series)["drift"]


def test_trend_endpoint_caches_metrics():
    from fastapi.testclient import TestClient
    from physicslens import app as app_module

    ids = [stored_run(app_module, pendulum([(6, 0.6, g)], seed=50 + i)) for i, g in enumerate([0.05, 0.08, 0.12])]
    client = TestClient(app_module.app)
    body = client.post("/api/health/trend", json={"runs": ids}).json()
    assert [r["id"] for r in body["recordings"]] == ids
    assert body["recordings"][0]["flagged"] == {"natural_freq_hz": False, "decay_rate_per_s": False}
    assert body["recordings"][2]["flagged"]["decay_rate_per_s"]
    assert all((app_module.runs.run_dir(i) / "health.json").is_file() for i in ids)
    assert client.post("/api/health/trend", json={"runs": ids[:2]}).status_code == 400
