"""PhysicsLens as an MCP server: an AI agent can measure the physics in a video and get verified equations back.

A thin client of the PhysicsLens HTTP API, run on the user's own machine over stdio:

    PHYSICSLENS_URL=https://... PHYSICSLENS_ACCESS_CODE=... python -m physicslens.mcp_server

It reads only video files on that machine and talks only to the configured PhysicsLens server, so it can't be used to
fetch arbitrary URLs, and every paid model call still happens on the server, behind its cost guard and budgets.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from mcp.server import MCPServer

# ToolError's message reaches the calling agent; any other exception is treated as a crash and its text withheld.
from mcp.server.mcpserver.exceptions import ToolError

VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v")
EXPORT_KINDS = ("json", "py", "mjcf", "contact", "report")
POLL_S = 3.0
WAIT_LIMIT_S = 1800.0  # an analysis on the open-weight model takes minutes, not seconds

ClientFactory = Callable[[], httpx.Client]


def _default_client() -> httpx.Client:
    base = os.environ.get("PHYSICSLENS_URL", "").rstrip("/")
    if not base:
        raise ToolError("Set PHYSICSLENS_URL to the PhysicsLens server's address.")
    code = os.environ.get("PHYSICSLENS_ACCESS_CODE", "")
    return httpx.Client(base_url=base, headers={"x-access-code": code} if code else {}, timeout=60.0)


def _check(r: httpx.Response) -> Any:
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise ToolError(f"PhysicsLens said {r.status_code}: {detail}")
    return r.json() if "json" in r.headers.get("content-type", "") else r.text


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    """What an agent needs: the verdict, the law, numbers with intervals, and why the rivals lost."""
    eq = result.get("equation") or {}
    return {
        "status": result.get("status"), "reason": result.get("reason"), "family": result.get("family"),
        "equation": {"title": eq.get("title"), "latex": eq.get("latex")} if eq else None,
        "parameters": [{k: p.get(k) for k in ("name", "value", "std", "ci_low", "ci_high", "unit", "uncalibrated")}
                       for p in result.get("parameters", [])],
        "verification": [{k: r.get(k) for k in ("title", "heldout_nrmse", "bic", "rationale")}
                         for r in result.get("ranking", [])],
        "scale_basis": result.get("scale_basis") or (result.get("calibration") or {}).get("source"),
        "audit": {k: (result.get("audit") or {}).get(k) for k in ("verdict", "headline", "reasons")} if result.get("audit") else None,
        "decision_trail": [{"step": t.get("step"), "model": t.get("model"), "text": t.get("text")} for t in result.get("trail", [])],
    }


def build(client_factory: ClientFactory = _default_client, poll_s: float = POLL_S) -> MCPServer:
    mcp = MCPServer("PhysicsLens", instructions=(
        "Measure the physics of a moving object from a video. The equation is accepted only if it forecasts held-out "
        "motion; otherwise the answer says no clean equation was found. Analyses take several minutes."))

    def _wait(c: httpx.Client, run_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + WAIT_LIMIT_S
        while True:
            state = _check(c.get(f"/api/runs/{run_id}"))
            if state["status"] == "done":
                return state["result"]
            if state["status"] == "error":
                raise ToolError(state.get("error") or "The analysis failed.")
            if time.monotonic() > deadline:
                raise ToolError(f"Still running after {WAIT_LIMIT_S / 60:.0f} minutes; ask again with get_result('{run_id}').")
            time.sleep(poll_s)

    def _upload(path: str, endpoint: str, options: dict[str, Any]) -> dict[str, Any]:
        p = Path(path).expanduser()
        if p.suffix.lower() not in VIDEO_SUFFIXES or not p.is_file():
            raise ToolError(f"Give the path of an MP4 or MOV file on this machine (got {path!r}).")
        form = {k: str(v) for k, v in options.items() if v is not None}
        with client_factory() as c, p.open("rb") as f:
            state = _check(c.post(endpoint, files={"video": (p.name, f, "video/mp4")}, data=form))
            result = _wait(c, state["run_id"])
        return {"run_id": state["run_id"], **summarize(result)}

    @mcp.tool()
    def measure_physics(video_path: str, pendulum_length_m: float | None = None, object_size_m: float | None = None,
                        slow_motion: float = 1.0, object_hint: str | None = None) -> dict[str, Any]:
        """Find and verify the equation of motion in a video on this machine.

        Optional: the pendulum's length or the moving object's size in meters (sets the scale), the slow-motion
        factor if the clip was filmed in slow motion, and a short hint of what to track ("the red ball").
        """
        return _upload(video_path, "/api/runs", {"pendulum_length_m": pendulum_length_m, "object_size_m": object_size_m,
                                                "slow_motion": slow_motion, "object_hint": object_hint})

    @mcp.tool()
    def audit_physics(video_path: str, object_size_m: float | None = None) -> dict[str, Any]:
        """Check whether the motion in a video obeys physics (for AI-generated video): consistent, violates physics
        with each violation in numbers, or can't judge."""
        return _upload(video_path, "/api/audit", {"object_size_m": object_size_m})

    @mcp.tool()
    def get_result(run_id: str) -> dict[str, Any]:
        """The result of an earlier analysis, waiting for it to finish if it is still running."""
        with client_factory() as c:
            return {"run_id": run_id, **summarize(_wait(c, run_id))}

    @mcp.tool()
    def export(run_id: str, kind: str) -> str:
        """Export a verified result: json (parameters), py (standalone simulation), mjcf (MuJoCo model),
        contact (MuJoCo bounce contact), report (one-page Markdown)."""
        if kind not in EXPORT_KINDS:
            raise ToolError(f"kind must be one of {', '.join(EXPORT_KINDS)}.")
        with client_factory() as c:
            return str(_check(c.get(f"/api/runs/{run_id}/export/{kind}")))

    @mcp.tool()
    def what_if(run_id: str, question: str) -> dict[str, Any]:
        """Re-simulate a verified result with a changed parameter, asked in plain words ("what if it were on the Moon?")."""
        with client_factory() as c:
            return _check(c.post(f"/api/runs/{run_id}/whatif", json={"question": question}))

    return mcp


def main() -> None:
    build().run("stdio")


if __name__ == "__main__":
    main()
