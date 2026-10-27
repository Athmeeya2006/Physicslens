"""FastAPI entry point."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Annotated

from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .costguard import CostGuardClient
from .export.mp4 import ExportError, render_share_video
from .live import LiveConfig, LiveSession
from .pipeline import RunOptions
from .runs import RunManager
from .settings import load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Raises before the server accepts a request if MOCK_MODELS=1 in demo/prod or budget.yaml is missing.
settings = load_settings()
client = CostGuardClient(settings)
runs = RunManager(settings, client)

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_LIVE_FRAMES = 25 * 60 * 10  # 10 minutes at 25 fps; budget caps normally end a session long before
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}

app = FastAPI(title="PhysicsLens")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.environ.get("FRONTEND_ORIGINS", "http://localhost:3000").split(",") if o],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def audience_for(access_code: str | None) -> str:
    if settings.app_env == "dev":
        return "dev"
    expected = os.environ.get("DEMO_ACCESS_CODE", "")
    if expected and access_code and hmac.compare_digest(access_code, expected):
        return "judge"
    return "public"


@app.get("/api/health")
def health() -> dict[str, object]:
    # mock_mode drives the red "MOCK MODE" banner in the UI.
    return {"status": "ok", "app_env": settings.app_env, "mock_mode": settings.mock_models}


@app.get("/api/budget")
def budget() -> dict[str, object]:
    # Dev UI header only: judges and public visitors must never see quota numbers.
    if settings.app_env != "dev":
        raise HTTPException(status_code=404)
    return client.guard.summary()


@app.post("/api/runs")
def create_run(
    video: Annotated[UploadFile, File()],
    pendulum_length_m: Annotated[float | None, Form()] = None,
    x_access_code: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    audience = audience_for(x_access_code)
    if audience == "public" and os.environ.get("PUBLIC_TRY_ENABLED", "0") != "1":
        raise HTTPException(status_code=403, detail="Uploads are closed right now; please browse the gallery.")
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Please upload a video file (MP4 or MOV).")
    run_dir = runs.new_run_dir()
    dest = run_dir / f"input{suffix}"
    written = 0
    with dest.open("wb") as out:
        while chunk := video.file.read(1 << 20):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                break
            out.write(chunk)
    if written > MAX_UPLOAD_BYTES:
        shutil.rmtree(run_dir)
        raise HTTPException(status_code=413, detail="That video is too large; trim it to under 30 seconds.")
    state = runs.start(dest, RunOptions(audience=audience, pendulum_length_m=pendulum_length_m))
    return runs.public_view(state)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, object]:
    state = runs.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown run.")
    return runs.public_view(state)


def _run_file(run_id: str, name: str) -> Path:
    if not run_id.replace("-", "").isalnum():
        raise HTTPException(status_code=404)
    return runs.run_dir(run_id) / name


@app.get("/api/runs/{run_id}/clip.mp4")
def get_clip(run_id: str) -> FileResponse:
    path = _run_file(run_id, "clip.mp4")
    if not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="video/mp4")


@app.post("/api/runs/{run_id}/share")
def make_share_video(run_id: str) -> dict[str, str]:
    """Render (once) the clip with the trail, ghost and equation burned in. Local only: costs nothing."""
    state = runs.get(run_id)
    if state is None or state.status != "done" or state.result is None:
        raise HTTPException(status_code=404, detail="That analysis is not finished.")
    out = _run_file(run_id, "share.mp4")
    result_file = _run_file(run_id, "result.json")
    stale = not out.is_file() or (result_file.is_file() and result_file.stat().st_mtime > out.stat().st_mtime)
    if stale:
        try:
            render_share_video(state.result, _run_file(run_id, "clip.mp4"), out)
        except ExportError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
    return {"url": f"/api/runs/{run_id}/share.mp4"}


@app.get("/api/runs/{run_id}/share.mp4")
def get_share_video(run_id: str) -> FileResponse:
    path = _run_file(run_id, "share.mp4")
    if not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="video/mp4", filename=f"physicslens-{run_id}.mp4")


def live_cap() -> float:
    cap = client.guard.budget.get("live_session_cap_usd")
    if cap is None:
        logging.getLogger("physicslens").warning("live_session_cap_usd missing from config/budget.yaml; using per_run_cap_usd.")
        return float(client.guard.budget["per_run_cap_usd"])
    return float(cap)


@app.websocket("/ws/live")
async def live(ws: WebSocket) -> None:
    """First message: JSON {fps, pendulum_length_m?, access_code?}. Then one binary JPEG per camera frame."""
    await ws.accept()
    init = await ws.receive_json()
    audience = audience_for(init.get("access_code"))
    if audience == "public":
        await ws.send_json({"type": "notice", "level": "error", "message": "Live mode needs the access code; the gallery is open to everyone."})
        await ws.close()
        return
    loop = asyncio.get_running_loop()

    def send(msg: dict) -> None:
        # Called from worker threads; the socket belongs to the event loop.
        asyncio.run_coroutine_threadsafe(ws.send_json(msg), loop)

    run_dir = runs.new_run_dir()
    fps = min(float(init.get("fps") or 25), 30.0)
    length = init.get("pendulum_length_m")
    session = LiveSession(client, run_dir, f"live-{run_dir.name}",
                          LiveConfig(fps=fps, pendulum_length_m=float(length) if length else None, audience=audience,
                                     run_cap_usd=live_cap()), send)
    await ws.send_json({"type": "ready", "run_id": run_dir.name})
    try:
        frames = 0
        while frames < MAX_LIVE_FRAMES:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                if len(message["bytes"]) <= MAX_FRAME_BYTES:
                    session.add_frame(message["bytes"])
                    frames += 1
            elif message.get("text"):
                cmd = json.loads(message["text"])
                if cmd.get("type") == "select":
                    session.select(float(cmd["x"]), float(cmd["y"]))
        else:
            await ws.send_json({"type": "notice", "level": "info", "message": "Session length limit reached."})
    except WebSocketDisconnect:
        pass
    except ValueError as e:
        await ws.send_json({"type": "notice", "level": "error", "message": str(e)})
    finally:
        session.close()
