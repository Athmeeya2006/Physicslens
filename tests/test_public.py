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


def test_cloudflare_address_wins_behind_the_tunnel(monkeypatch):
    """Through a Cloudflare tunnel every request comes from 127.0.0.1; the visitor is in CF-Connecting-IP."""
    monkeypatch.setenv("TRUST_PROXY", "1")
    assert client_address("127.0.0.1", "6.6.6.6", "8.8.8.8") == "8.8.8.8"
    monkeypatch.delenv("TRUST_PROXY")
    assert client_address("127.0.0.1", "6.6.6.6", "8.8.8.8") == "127.0.0.1"


def test_visitor_allowance_is_for_the_whole_period_per_kind(tmp_path):
    from physicslens.public import VisitorAllowance
    v = VisitorAllowance(tmp_path / "v.db", {"analysis": 2, "live": 1})
    assert v.take("1.1.1.1", "analysis") and v.take("1.1.1.1", "analysis") and not v.take("1.1.1.1", "analysis")
    assert v.take("1.1.1.1", "live") and not v.take("1.1.1.1", "live")
    assert v.take("2.2.2.2", "analysis")
    assert not v.take("1.1.1.1", "unknown kind")  # anything unlisted is closed


@pytest.fixture
def open_app(monkeypatch, tmp_path):
    import dataclasses

    from physicslens import app as app_module
    from physicslens.public import VisitorAllowance
    monkeypatch.setattr(app_module, "settings", dataclasses.replace(app_module.settings, app_env="demo"))
    monkeypatch.setattr(app_module, "visitors", VisitorAllowance(tmp_path / "v.db", {"analysis": 2}))
    monkeypatch.setenv("OPEN_ACCESS", "1")
    monkeypatch.setenv("TRUST_PROXY", "1")
    monkeypatch.setenv("DEMO_ACCESS_CODE", "judge-code")
    return app_module


def test_open_access_needs_no_code_and_limits_each_visitor(open_app):
    from fastapi.testclient import TestClient
    client = TestClient(open_app.app)
    health = client.get("/api/health").json()
    assert health["audience"] == "judge" and health["can_upload"] and health["can_go_live"] and health["open_access"]
    post = lambda ip, **h: client.post("/api/runs", files={"video": ("a.mp4", b"x", "video/mp4")},
                                       headers={"cf-connecting-ip": ip, **h}).status_code
    assert [post("8.8.8.8") for _ in range(3)] == [200, 200, 429]
    assert post("9.9.9.9") == 200                               # another visitor
    assert post("8.8.8.8", **{"x-access-code": "judge-code"}) == 200  # the code still lifts the per-visitor limit
