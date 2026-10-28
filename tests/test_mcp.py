"""The MCP server: tools an AI agent calls, backed by the PhysicsLens HTTP API (faked here, no network)."""

import json

import anyio
import httpx
import pytest
from mcp.server.mcpserver.exceptions import ToolError
from physicslens.mcp_server import build

RESULT = {"status": "accepted", "reason": "Pendulum forecasts held-out data within 0.6%", "family": "pendulum",
          "equation": {"title": "Pendulum", "latex": r"\ddot\theta = -\frac{g}{L}\sin\theta"},
          "parameters": [{"name": "g", "value": 9.79, "std": 0.01, "ci_low": 9.77, "ci_high": 9.81, "unit": "m/s^2"}],
          "ranking": [{"title": "Pendulum", "heldout_nrmse": 0.006, "bic": -900.0, "rationale": "accepted"}],
          "trail": [{"step": "scene", "model": "muse-glimmer-30b", "text": "A pendulum"}],
          "audit": {"verdict": "consistent", "headline": "Obeys physics", "reasons": []}}


def fake_api(seen: list[httpx.Request]) -> httpx.MockTransport:
    polls = {"n": 0}

    def handle(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.method == "POST" and req.url.path in ("/api/runs", "/api/audit"):
            return httpx.Response(200, json={"run_id": "r1", "status": "queued", "progress": []})
        if req.url.path == "/api/runs/r1":
            polls["n"] += 1
            done = polls["n"] > 1  # first poll: still running
            return httpx.Response(200, json={"run_id": "r1", "status": "done" if done else "running",
                                             "result": RESULT if done else None})
        if req.url.path == "/api/runs/r1/export/mjcf":
            return httpx.Response(200, text="<mujoco/>", headers={"content-type": "application/xml"})
        return httpx.Response(404, json={"detail": "Unknown run."})

    return httpx.MockTransport(handle)


@pytest.fixture
def server(tmp_path):
    seen: list[httpx.Request] = []
    mcp = build(lambda: httpx.Client(base_url="http://physicslens.test", transport=fake_api(seen),
                                     headers={"x-access-code": "judge"}), poll_s=0.0)
    video = tmp_path / "swing.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    return mcp, seen, video


def call(mcp, name: str, args: dict):
    async def go():
        return await mcp.call_tool(name, args)
    return anyio.run(go)


def payload(result) -> dict:
    if result.structured_content is not None:
        content = result.structured_content
        return content.get("result", content) if set(content) == {"result"} else content
    return json.loads(result.content[0].text)


def test_tools_are_listed():
    names = {t.name for t in anyio.run(build(lambda: None).list_tools)}
    assert {"measure_physics", "audit_physics", "get_result", "export", "what_if"} <= names


def test_measure_uploads_waits_and_summarizes(server):
    mcp, seen, video = server
    out = payload(call(mcp, "measure_physics", {"video_path": str(video), "pendulum_length_m": 0.5}))
    assert out["run_id"] == "r1" and out["status"] == "accepted" and out["parameters"][0]["value"] == 9.79
    assert out["decision_trail"][0]["model"] == "muse-glimmer-30b" and out["audit"]["verdict"] == "consistent"
    upload = seen[0]
    assert upload.headers["x-access-code"] == "judge" and b"pendulum_length_m" in upload.content


def test_only_local_video_files_are_accepted(server):
    mcp, seen, _ = server
    with pytest.raises(ToolError, match="MP4 or MOV file on this machine"):
        call(mcp, "measure_physics", {"video_path": "https://example.com/clip.mp4"})
    assert seen == []  # nothing fetched, nothing sent


def test_export_kind_is_checked_and_returns_the_file(server):
    mcp, _, _ = server
    assert "<mujoco/>" in str(payload(call(mcp, "export", {"run_id": "r1", "kind": "mjcf"})))
    with pytest.raises(ToolError, match="kind must be one of"):
        call(mcp, "export", {"run_id": "r1", "kind": "exe"})
