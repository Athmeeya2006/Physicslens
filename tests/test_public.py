"""Public visitors: a few free analyses per address per day, optional bot check, a free gallery."""

import json

import pytest
from physicslens.public import PublicQuota, client_address, turnstile_ok


def test_quota_per_address_per_day(tmp_path):
    q = PublicQuota(tmp_path / "public.db", per_day=2)
    assert q.take("1.1.1.1") and q.take("1.1.1.1") and not q.take("1.1.1.1")
    assert q.take("2.2.2.2")  # another visitor is unaffected


def test_forwarded_address_only_trusted_behind_a_proxy(monkeypatch):
    monkeypatch.delenv("TRUST_PROXY", raising=False)
    assert client_address("10.0.0.1", "9.9.9.9") == "10.0.0.1"
    monkeypatch.setenv("TRUST_PROXY", "1")
    assert client_address("10.0.0.1", "9.9.9.9, 10.0.0.1") == "9.9.9.9"


def test_turnstile_off_unless_configured(monkeypatch):
    monkeypatch.delenv("TURNSTILE_SECRET_KEY", raising=False)
    assert turnstile_ok(None, "1.1.1.1")
    monkeypatch.setenv("TURNSTILE_SECRET_KEY", "test-secret")
    assert not turnstile_ok(None, "1.1.1.1")  # configured: a missing token is refused without any network call


@pytest.fixture
def public_app(monkeypatch, tmp_path):
    from physicslens import app as app_module
    monkeypatch.setattr(app_module, "audience_for", lambda code: "public")
    monkeypatch.setattr(app_module, "public_quota", PublicQuota(tmp_path / "q.db", per_day=2))
    monkeypatch.delenv("TURNSTILE_SECRET_KEY", raising=False)
    return app_module


def test_public_uploads_closed_by_default(public_app, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("PUBLIC_TRY_ENABLED", "0")
    r = TestClient(public_app.app).post("/api/runs", files={"video": ("a.mp4", b"x", "video/mp4")})
    assert r.status_code == 403 and "gallery" in r.json()["detail"]


def test_public_uploads_limited_per_day(public_app, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("PUBLIC_TRY_ENABLED", "1")
    client = TestClient(public_app.app)
    codes = [client.post("/api/runs", files={"video": ("a.mp4", b"x", "video/mp4")}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    health = client.get("/api/health").json()
    assert health["audience"] == "public" and not health["can_go_live"]
    assert "budget" not in json.dumps(health) and "quota" not in json.dumps(health)


def test_gallery_lists_curated_results(public_app, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    item = tmp_path / "gallery" / "gallery-test"
    item.mkdir(parents=True)
    (item / "meta.json").write_text(json.dumps({"title": "Pendulum", "description": "d", "credit": "c", "order": 1}))
    (item / "result.json").write_text(json.dumps({"status": "accepted", "family": "pendulum", "scene": {"moving_object": "ball"}}))
    monkeypatch.setattr(public_app.runs, "gallery_root", tmp_path / "gallery")
    client = TestClient(public_app.app)
    items = client.get("/api/gallery").json()
    assert items[0]["id"] == "gallery-test" and items[0]["title"] == "Pendulum" and items[0]["object"] == "ball"
    assert client.get("/api/runs/gallery-test").json()["status"] == "done"  # opens like any finished run
