"""The Solar System Lab's API: free, bounded, cached, and never a paid call (except /render, an upload)."""
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def lab(monkeypatch):
    from physicslens import app as app_module
    from physicslens import sky_api
    sky_api.CACHE.items.clear()
    sky_api.WINDOWS = sky_api._windows()  # a fresh minute for each test
    # Another test's upload may still be analysing in the background: let it finish before counting paid calls.
    deadline = time.monotonic() + 120
    while any(r.status in ("queued", "running") for r in list(app_module.runs._runs.values())):
        if time.monotonic() > deadline:
            break
        time.sleep(0.2)
    paid = []
    for name in ("meta_text", "open_weight_chat", "sam_video"):
        if hasattr(app_module.client, name):
            monkeypatch.setattr(app_module.client, name, lambda *a, _n=name, **k: paid.append(_n))
    yield TestClient(app_module.app), sky_api, paid
    assert paid == []  # nothing in the lab spends money


def test_scenarios_and_bodies_are_listed(lab):
    client, _, _ = lab
    names = {s["name"] for s in client.get("/api/sky/scenarios").json()}
    assert {"apophis_with_planets", "uranus_neptune_hidden", "mercury_gr_exaggerated"} <= names
    bodies = {b["name"]: b for b in client.get("/api/sky/bodies").json()}
    assert bodies["Jupiter"]["gm_km3_s2"] > 1.2e8 and "JPL" in bodies["Jupiter"]["gm_source"]


def test_simulate_returns_paths_elements_and_detectability_and_caches(lab):
    client, _, _ = lab
    req = {"scenario": "mercury_gr_exaggerated"}
    first = client.post("/api/sky/simulate", json=req).json()
    assert first["cached"] is False and first["source"] == "simulation, no pixels"
    assert all(len(b["points"]) <= 2000 for b in first["bodies"]) and first["elements"]
    assert first["detectability"]["visible"]
    assert first["precession_arcsec_per_century"] == pytest.approx(20000 * 43, rel=0.05)
    assert client.post("/api/sky/simulate", json=req).json()["cached"] is True


def test_a_change_makes_it_fictional_and_is_compared_with_the_original(lab):
    client, _, _ = lab
    out = client.post("/api/sky/simulate", json={"scenario": "real_inner_planets", "mass_scale": {"Jupiter": 100},
                                                 "years": 5}).json()
    assert out["scenario"]["fictional"] and out["detectability"] is not None


@pytest.mark.parametrize(("body", "status"), [
    ({"scenario": "jupiter_x10", "mass_scale": {"Jupiter": 5000}}, 422),
    ({"scenario": "jupiter_x10", "mass_scale": {"Jupiter": "NaN"}}, 422),
    ({"scenario": "jupiter_x10", "years": 400}, 422),
    ({"scenario": "rogue_star_flyby", "star": {"gm_suns": 0.5, "closest_au": 200, "speed_km_s": 0, "closest": "2100-01-01"}}, 422),
    ({"scenario": "jupiter_x10", "mass_scale": {"Pluto": 2}}, 400),
    ({"scenario": "jupiter_x10", "disabled": ["Mars"]}, 400),
    ({"scenario": "no_such_thing"}, 404),
    ({"scenario": "jupiter_x10", "surprise": 1}, 422),
])
def test_bad_requests_are_refused_with_a_reason(lab, body, status):
    client, _, _ = lab
    r = client.post("/api/sky/simulate", json=body)
    assert r.status_code == status and r.json()["detail"]


def test_each_address_is_rate_limited(lab):
    client, sky_api, _ = lab
    sky_api.configure({"per_ip_per_minute": 2})
    try:
        replies = [client.post("/api/sky/simulate", json={"scenario": "mercury_newton", "years": 1},
                               headers={"x-forwarded-for": "8.8.8.8"}) for _ in range(3)]
    finally:
        sky_api.configure({"per_ip_per_minute": 20})
    assert [r.status_code for r in replies] == [200, 200, 429]
    assert 1 <= int(replies[2].headers["retry-after"]) <= 60


@pytest.mark.slow
def test_discover_is_free_and_labelled(lab):
    client, _, _ = lab
    out = client.post("/api/sky/discover", json={"scenario": "apophis_sun_only", "noise_px": 2.0}).json()
    assert out["source"] == "simulation, no pixels" and out["status"] == "accepted"
    # Every candidate carries its verdict and the reason, for the lab's "how the law was chosen" table.
    verdicts = {r["status"] for r in out["ranking"]}
    assert "accepted" in verdicts and all(r["why"] for r in out["ranking"])
