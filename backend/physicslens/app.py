"""FastAPI entry point."""

from __future__ import annotations

import hmac
import logging
import os
import shutil
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .costguard import CostGuardClient
from .pipeline import RunOptions
from .runs import RunManager
from .settings import load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Raises before the server accepts a request if MOCK_MODELS=1 in demo/prod or budget.yaml is missing.
settings = load_settings()
client = CostGuardClient(settings)
runs = RunManager(settings, client)

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
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


@app.get("/api/runs/{run_id}/clip.mp4")
def get_clip(run_id: str) -> FileResponse:
    if not run_id.isalnum():
        raise HTTPException(status_code=404)
    path = runs.run_dir(run_id) / "clip.mp4"
    if not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="video/mp4")
