"""FastAPI entry point."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import math
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    Body,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from .agent.llm import LLMError as StudyLLMError
from .agent.llm import reasoning_provider
from .agent.notes import interpret_notes
from .agent.schemas import ControlTask
from .agent.setup_guide import GUIDE_FRAME, GUIDE_JPG, SetupGuideError, make_setup_guide
from .agent.whatif import run_whatif
from .control.harness import isolation as sandbox_isolation
from .control.manager import ControlManager
from .control.system import ControlUnsupported
from .costguard import CostGuardClient, CostGuardError
from .costguard.limiter import RequestWindows
from .export.mjcf import bounce_mjcf, build_mjcf
from .export.mp4 import ExportError, render_share_video
from .export.og import render as render_og
from .export.report import report_markdown
from .export.robot import RobotExportError, sdf, urdf
from .export.script import parameters_json, python_script
from .export.sim2real import Sim2RealError, sim2real
from .inputs import MAX_INPUT_BYTES, InputError, parse_inputs_csv
from .live import LiveConfig, LiveSession
from .perception import media
from .perception.lens import Lens, LensError, calibrate_video, checkerboard_svg
from .physics.health import (
    HealthError,
    HealthMetrics,
    compare,
    health_metrics,
    metrics_from_dict,
    metrics_to_dict,
    trend,
)
from .pipeline import MIN_CLIP_FRAMES, MIN_CLIP_SECONDS, AnalysisError, RunOptions, trajectory_from_result
from .public import PublicQuota, VisitorAllowance, client_address, turnstile_ok
from .runs import RunManager
from .settings import OPEN_WEIGHT_MODELS, cors_origins, load_settings
from .sky.nbody import SkyTooSlow
from .sky.scenarios import FRAME_NOTE as SKY_FRAME_NOTE
from .sky.scenarios import render_scenario
from .sky.scenarios import run as run_scenario
from .sky_api import SimulateRequest
from .sky_api import configure as configure_sky
from .sky_api import router as sky_router
from .sky_api import scenario_for as sky_scenario
from .studies import Studies, StudyError
from .voice.events import final_turn, partial_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Raises before the server accepts a request if MOCK_MODELS=1 in demo/prod or budget.yaml is missing.
log = logging.getLogger("physicslens.app")
settings = load_settings()
client = CostGuardClient(settings)
runs = RunManager(settings, client)
controls = ControlManager(client, runs.root)
studies = Studies(settings.data_dir / "studies", client)
public_quota = PublicQuota(settings.data_dir / "public.db", int(client.guard.budget.get("public_per_ip_daily_analyses", 3)))

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_LIVE_FRAMES = 25 * 60 * 10  # 10 minutes at 25 fps; budget caps normally end a session long before
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}

app = FastAPI(title="PhysicsLens")
# Probe the controller sandbox's OS isolation now, off the request path, so a missing wall is in the log at startup.
threading.Thread(target=sandbox_isolation, name="sandbox-probe", daemon=True).start()
app.include_router(sky_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(settings.app_env, os.environ.get("FRONTEND_ORIGINS")),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def has_code(access_code: str | None) -> bool:
    expected = os.environ.get("DEMO_ACCESS_CODE", "")
    return bool(expected and access_code and hmac.compare_digest(access_code, expected))


def open_access() -> bool:
    """Open judging: everyone gets what a judge gets, with no code to paste (the judges' total allowance still caps
    the money, and VisitorAllowance keeps one visitor from using it all)."""
    return os.environ.get("OPEN_ACCESS", "0") == "1"


def audience_for(access_code: str | None) -> str:
    if settings.app_env == "dev":
        return "dev"
    if has_code(access_code) or open_access():
        return "judge"
    return "public"


# Per visitor, for the whole period, under open access: generous for a judge trying everything twice.
# "question": a typed or spoken what-if, or spoken settings: one short text call each, so many more than analyses.
OPEN_ACCESS_DEFAULTS = {"analysis": 15, "live": 5, "picture": 5, "controller": 3, "study": 3, "question": 40,
                       "lens": 10}  # lens: free local math, limited only because it is CPU-heavy
configure_sky(client.limits.get("sky"))
visitors = VisitorAllowance(settings.data_dir / "public.db",
                            {**OPEN_ACCESS_DEFAULTS, **(client.guard.budget.get("open_access_per_visitor") or {})})


# Incoming requests per route group over a sliding minute, per visitor address and for everyone: a flood is refused
# before any work starts. Generous for people; config/limits.yaml (requests:) can change them.
REQUEST_DEFAULTS = {
    "uploads": {"per_ip_per_minute": 6, "global_per_minute": 30},   # /api/runs, /api/audit, /api/sky/render
    "studies": {"per_ip_per_minute": 10, "global_per_minute": 60},  # creating studies, adding recordings, reviews
    "live": {"per_ip_per_minute": 6, "global_per_minute": 20},      # /ws/live sessions
    "voice": {"per_ip_per_minute": 12, "global_per_minute": 60},    # /ws/voice, /ws/voice/notes
    "lens": {"per_ip_per_minute": 4, "global_per_minute": 12},      # /api/lens/calibrate (CPU-heavy)
}
request_windows = RequestWindows({**REQUEST_DEFAULTS, **(client.limits.get("requests") or {})})


def _address(remote: str | None, headers: Any) -> str:
    return client_address(remote, headers.get("x-forwarded-for"), headers.get("cf-connecting-ip"))


def limit_requests(group: str, request: Request) -> None:
    """429 with Retry-After when this address, or everyone together, sent too many of these in the last minute."""
    wait = request_windows.check(group, _address(request.client.host if request.client else None, request.headers))
    if wait is not None:
        raise HTTPException(status_code=429, detail="Too many requests in a minute; please wait a moment and try again.",
                            headers={"Retry-After": str(math.ceil(wait))})


async def ws_allowed(group: str, ws: WebSocket) -> bool:
    """The socket form of limit_requests: a notice, then close code 1013 (try again later)."""
    wait = request_windows.check(group, _address(ws.client.host if ws.client else None, ws.headers))
    if wait is None:
        return True
    await ws.send_json({"type": "notice", "level": "error",
                        "message": f"Too many sessions in a minute; please try again in {math.ceil(wait)} s."})
    await ws.close(code=1013)
    return False


def allow_visitor(kind: str, access_code: str | None, headers: Any, remote: str | None) -> None:
    """Under open access, one more paid action of this kind for this visitor, or a polite refusal (never a count)."""
    if settings.app_env == "dev" or not open_access() or has_code(access_code):
        return
    addr = client_address(remote, headers.get("x-forwarded-for"), headers.get("cf-connecting-ip"))
    if not visitors.take(addr, kind):
        raise HTTPException(status_code=429, detail="You've tried this a lot already; every result in the gallery "
                                                    "shows the full analysis. Please come back later.")


def public_try_enabled() -> bool:
    return os.environ.get("PUBLIC_TRY_ENABLED", "0") == "1"


@app.get("/api/health")
def health(x_access_code: Annotated[str | None, Header()] = None) -> dict[str, object]:
    # mock_mode drives the red "MOCK MODE" banner; audience decides what the page offers (never any quota numbers).
    audience = audience_for(x_access_code)
    provider = reasoning_provider()
    return {"status": "ok", "app_env": settings.app_env, "mock_mode": settings.mock_models, "audience": audience,
            "reasoning": {"provider": provider, "model": OPEN_WEIGHT_MODELS[provider]} if provider else None,
            "can_upload": audience != "public" or public_try_enabled(), "can_go_live": audience != "public",
            "open_access": open_access(),
            # The OS wall around generated controller code: "bubblewrap", "unshare" (network only) or "none".
            "sandbox_isolation": sandbox_isolation(),
            "turnstile_site_key": os.environ.get("TURNSTILE_SITE_KEY", "") if audience == "public" else ""}


@app.get("/api/gallery")
def gallery() -> list[dict]:
    return runs.gallery()


@app.get("/api/budget")
def budget() -> dict[str, object]:
    # Dev UI header only: judges and public visitors must never see quota numbers.
    if settings.app_env != "dev":
        raise HTTPException(status_code=404)
    return client.guard.summary()


@app.post("/api/runs")
# An audit is a normal run: every result carries result["audit"] (local math, no extra model calls), so the audit
# endpoint shares the upload handling, access rules and limits of /api/runs exactly.
@app.post("/api/audit")
def create_run(
    request: Request,
    video: Annotated[UploadFile, File()],
    pendulum_length_m: Annotated[float | None, Form()] = None,
    object_size_m: Annotated[float | None, Form()] = None,
    slow_motion: Annotated[float, Form()] = 1.0,
    object_hint: Annotated[str | None, Form(max_length=60)] = None,
    turnstile_token: Annotated[str | None, Form()] = None,
    x_access_code: Annotated[str | None, Header()] = None,
    x_forwarded_for: Annotated[str | None, Header()] = None,
    blade_count: Annotated[int | None, Form(ge=1, le=64)] = None,
    depth_3d: Annotated[bool, Form()] = False,
    focal_px: Annotated[float | None, Form()] = None,
    gravity_up: Annotated[str | None, Form(max_length=80)] = None,
    lens: Annotated[str | None, Form(max_length=600)] = None,
    inputs_csv: Annotated[UploadFile | None, File()] = None,
    known_mover: Annotated[str | None, Form(max_length=60)] = None,
) -> dict[str, object]:
    limit_requests("uploads", request)
    audience = audience_for(x_access_code)
    inputs = None
    if inputs_csv is not None and inputs_csv.filename:
        # Known signals a law may use (a logged motor command, a second body's path): checked before the video is kept.
        try:
            inputs = parse_inputs_csv(inputs_csv.file.read(MAX_INPUT_BYTES + 1))
        except InputError as e:
            raise HTTPException(status_code=400, detail=f"Inputs file: {e}") from e
    if focal_px is not None and not 50 <= focal_px <= 50000:
        raise HTTPException(status_code=400, detail="The focal length should be between 50 and 50000 pixels.")
    up = parse_gravity_up(gravity_up)
    lens_dict = parse_lens(lens)
    for name, value in (("pendulum length", pendulum_length_m), ("object size", object_size_m)):
        if value is not None and not 0.005 <= value <= 100:
            raise HTTPException(status_code=400, detail=f"The {name} should be between 0.5 cm and 100 m.")
    if not 1 <= slow_motion <= 1000:
        raise HTTPException(status_code=400, detail="The slow-motion factor should be between 1 (normal speed) and 1000.")
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Please upload a video file (MP4 or MOV).")
    allow_visitor("analysis", x_access_code, request.headers, request.client.host if request.client else None)
    # Public checks come after the cheap request checks, so a wrong file never uses up a visitor's daily analysis.
    if audience == "public":
        if not public_try_enabled():
            raise HTTPException(status_code=403, detail="Uploads are closed right now; please browse the gallery.")
        addr = client_address(request.client.host if request.client else None, x_forwarded_for)
        if not turnstile_ok(turnstile_token, addr):
            raise HTTPException(status_code=403, detail="Please complete the quick check above the upload button.")
        if not public_quota.take(addr):
            raise HTTPException(status_code=429, detail="You've used today's free analyses. Browse the gallery, or come back tomorrow.")
    run_dir = runs.new_run_dir()
    dest = run_dir / f"input{suffix}"
    try:
        save_upload(video, dest, "That video is too large; trim it to under 30 seconds.")
    except HTTPException:
        shutil.rmtree(run_dir)
        raise
    # Read it now (ffprobe: length, frames, rotation), so a file that cannot work is refused with a reason before it
    # is queued and before anything is spent on it.
    try:
        info = media.probe(dest)
    except media.MediaError as e:
        shutil.rmtree(run_dir)
        raise HTTPException(status_code=400, detail=f"That file could not be read as a video ({e}).") from e
    if info.fps <= 0 or info.frames / info.fps < MIN_CLIP_SECONDS or info.frames < MIN_CLIP_FRAMES:
        shutil.rmtree(run_dir)
        raise HTTPException(status_code=400, detail=f"The clip is too short: at least {MIN_CLIP_SECONDS:g} seconds and "
                                                    f"{MIN_CLIP_FRAMES} frames are needed (this one has {info.frames} "
                                                    f"frames at {info.fps:.3g} fps).")
    state = runs.start(dest, RunOptions(audience=audience, pendulum_length_m=pendulum_length_m, object_size_m=object_size_m,
                                         slow_motion=slow_motion, object_hint=object_hint, blade_count=blade_count,
                                         depth_3d=depth_3d, focal_px=focal_px, gravity_up=up, lens=lens_dict, inputs=inputs,
                                         known_mover=known_mover))
    return runs.public_view(state)


def save_upload(video: UploadFile, dest: Path, too_large: str) -> None:
    """Stream an upload to dest (a name we chose, never the client's), refusing a non-video before writing anything.

    The first chunk must start like an MP4, MOV or WebM file; past MAX_UPLOAD_BYTES the copy stops. The caller removes
    dest's directory when this raises.
    """
    head = video.file.read(1 << 20)
    if not media.looks_like_video(head):
        raise HTTPException(status_code=400, detail="That file could not be read as a video (it does not start like "
                                                    "an MP4, MOV or WebM file).")
    written = 0
    with dest.open("wb") as out:
        chunk = head
        while chunk:
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=too_large)
            out.write(chunk)
            chunk = video.file.read(1 << 20)


def parse_gravity_up(text: str | None) -> tuple[float, float, float] | None:
    """The phone's accelerometer 'up' at capture, "x,y,z" in camera axes; any length (only the direction is used)."""
    if not text:
        return None
    try:
        x, y, z = (float(v) for v in text.split(","))
    except ValueError as e:
        raise HTTPException(status_code=400, detail="The gravity reading should be three numbers: x,y,z.") from e
    if not all(math.isfinite(v) for v in (x, y, z)) or math.hypot(x, y, z) < 1e-3:
        raise HTTPException(status_code=400, detail="The gravity reading is empty or not a number.")
    return (x, y, z)


def parse_lens(text: str | None) -> dict[str, object] | None:
    """A lens calibration sent with an upload (JSON from /api/lens/calibrate, kept in the visitor's browser)."""
    if not text:
        return None
    try:
        return Lens.from_dict(json.loads(text)).as_dict()
    except (ValueError, LensError) as e:
        raise HTTPException(status_code=400, detail=f"Lens calibration: {e}") from e


@app.post("/api/lens/calibrate")
def calibrate_lens(request: Request, video: Annotated[UploadFile, File()],
                   x_access_code: Annotated[str | None, Header()] = None) -> dict[str, object]:
    """Focal length and distortion from a short video of the printed board. Local math only: no model calls."""
    limit_requests("lens", request)
    allow_visitor("lens", x_access_code, request.headers, request.client.host if request.client else None)
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Please upload a video file (MP4 or MOV).")
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / f"board{suffix}"
        save_upload(video, dest, "That video is too large; 10 to 20 seconds is plenty.")
        try:
            lens_cal = calibrate_video(dest)
        except LensError as e:
            raise HTTPException(status_code=400, detail=f"Calibration failed: {e}") from e
    return {"lens": lens_cal.as_dict(), "focal_px": (lens_cal.fx + lens_cal.fy) / 2}


@app.get("/api/lens/checkerboard.svg")
def lens_checkerboard() -> Response:
    return Response(checkerboard_svg(), media_type="image/svg+xml")


@app.post("/api/sky/render")
def sky_render(
    request: Request,
    req: Annotated[SimulateRequest, Body()],
    x_access_code: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    """Film a lab scenario: render its video and run the normal (paid) pipeline on it, exactly like an upload, behind
    the same access rules and per-visitor allowance. The lab's own simulate and discover endpoints are free."""
    limit_requests("uploads", request)
    audience = audience_for(x_access_code)
    if audience == "public" and not public_try_enabled():
        raise HTTPException(status_code=403, detail="Filming a scenario is paused right now; the lab itself is open.")
    sc = sky_scenario(req)
    allow_visitor("analysis", x_access_code, request.headers, request.client.host if request.client else None)
    run_dir = runs.new_run_dir()
    dest = run_dir / "input.mp4"
    try:
        meta = render_scenario(run_scenario(sc), dest)
    except SkyTooSlow as e:
        shutil.rmtree(run_dir)
        raise HTTPException(status_code=503, detail=f"That scenario is too long to film: {e}.") from e
    state = runs.start(dest, RunOptions(
        audience=audience, object_hint=f"the white dot ({sc.target})", known_px_per_unit=meta["px_per_au"],
        origin_px=tuple(meta["sun_px"]), units={"length": "AU", "time": "day"}, frame_note=SKY_FRAME_NOTE,
        slow_motion=1.0 / (meta["step_days"] * meta["fps"])))
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


@app.get("/api/runs/{run_id}/poster.jpg")
def get_poster(run_id: str) -> FileResponse:
    path = _run_file(run_id, "poster.jpg")
    if not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/runs/{run_id}/og.jpg")
def get_share_image(run_id: str) -> FileResponse:
    """The link-preview picture for a finished result (drawn once, locally: costs nothing)."""
    state = runs.get(run_id)
    if state is None or state.status != "done" or state.result is None:
        raise HTTPException(status_code=404, detail="That analysis is not finished.")
    out = _run_file(run_id, "og.jpg")
    if not out.is_file():
        render_og(state.result, _run_file(run_id, "poster.jpg"), out)
    return FileResponse(out, media_type="image/jpeg")


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


FIRST_MESSAGE_SECONDS = 15


async def _first_message(ws: WebSocket) -> dict | None:
    """The opening JSON of a socket, or None (socket closed): a client that never says hello must not hold it open."""
    try:
        init = await asyncio.wait_for(ws.receive_json(), FIRST_MESSAGE_SECONDS)
    except (TimeoutError, ValueError, KeyError):
        await ws.close(code=1008)
        return None
    if not isinstance(init, dict):
        await ws.close(code=1008)
        return None
    return init


@app.websocket("/ws/live")
async def live(ws: WebSocket) -> None:
    """First message: JSON {fps, pendulum_length_m?, object_size_m?, access_code?}. Then one binary JPEG per camera frame."""
    await ws.accept()
    if not await ws_allowed("live", ws):
        return
    init = await _first_message(ws)
    if init is None:
        return
    audience = audience_for(init.get("access_code"))
    if audience == "public":
        await ws.send_json({"type": "notice", "level": "error", "message": "Live mode is paused right now; the gallery is open to everyone."})
        await ws.close()
        return
    try:
        allow_visitor("live", init.get("access_code"), ws.headers, ws.client.host if ws.client else None)
    except HTTPException as e:
        await ws.send_json({"type": "notice", "level": "error", "message": e.detail})
        await ws.close()
        return
    loop = asyncio.get_running_loop()

    def send(msg: dict) -> None:
        # Called from worker threads; the socket belongs to the event loop.
        asyncio.run_coroutine_threadsafe(ws.send_json(msg), loop)

    run_dir = runs.new_run_dir()
    try:
        fps = float(init.get("fps") or 25)
    except (TypeError, ValueError):
        fps = 25.0
    fps = min(fps, 30.0) if math.isfinite(fps) and fps >= 1 else 25.0  # NaN would pass min() untouched
    length, size = init.get("pendulum_length_m"), init.get("object_size_m")

    def positive(v) -> float | None:
        try:
            return float(v) if v is not None and 0 < float(v) < 1000 else None
        except (TypeError, ValueError):
            return None
    session = LiveSession(client, run_dir, f"live-{run_dir.name}",
                          LiveConfig(fps=fps, pendulum_length_m=positive(length), object_size_m=positive(size),
                                     audience=audience, run_cap_usd=live_cap()), send)
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
                elif cmd.get("type") == "family":
                    session.choose_family_now(str(cmd.get("family", "")))
                elif cmd.get("type") == "control":
                    session.set_control(str(cmd.get("goal", "off")), float(cmd.get("target_deg", 30)))
                elif cmd.get("type") == "length" and 0.005 <= float(cmd.get("meters", 0)) <= 100:
                    session.set_length(float(cmd["meters"]))
                elif cmd.get("type") == "save":
                    # The session becomes a normal result (a ?run= link, exports, what-ifs); the video only if asked.
                    try:
                        await asyncio.to_thread(session.save, run_dir, bool(cmd.get("keep_video")))
                        await ws.send_json({"type": "saved", "run_id": run_dir.name})
                    except ValueError as e:
                        await ws.send_json({"type": "notice", "level": "error", "message": str(e)})
        else:
            await ws.send_json({"type": "notice", "level": "info", "message": "Session length limit reached."})
    except WebSocketDisconnect:
        pass
    except ValueError as e:
        await ws.send_json({"type": "notice", "level": "error", "message": str(e)})
    finally:
        session.close()


VOICE_MAX_SECONDS = 60.0
VOICE_KEYWORDS = ["pendulum", "string", "length", "damping", "gravity", "spring", "twice", "half", "drag", "Mars", "Moon"]


def _finished_result(run_id: str) -> dict:
    state = runs.get(run_id)
    if state is None or state.status != "done" or state.result is None:
        raise HTTPException(status_code=404, detail="That analysis is not finished.")
    return state.result


@app.post("/api/runs/{run_id}/whatif")
def whatif(
    run_id: str,
    request: Request,
    question: Annotated[str, Body(embed=True)],
    x_access_code: Annotated[str | None, Header()] = None,
) -> dict:
    """Typed what-if (the voice path ends here too). Judges and development only, like voice."""
    audience = audience_for(x_access_code)
    if audience == "public":
        raise HTTPException(status_code=403, detail="What-if questions are paused right now.")
    result = _finished_result(run_id)
    allow_visitor("question", x_access_code, request.headers, request.client.host if request.client else None)
    try:
        # One budget "run" per question: a shared id would fill the lifetime per-run cap and lock everyone out.
        return run_whatif(client, question[:500], result, f"whatif-{run_id}-{uuid.uuid4().hex[:8]}", audience)
    except CostGuardError as e:
        detail = str(e) if settings.app_env == "dev" else "Temporarily busy, please retry later."
        raise HTTPException(status_code=503, detail=detail) from e


@app.post("/api/runs/{run_id}/setup-guide")
def setup_guide(
    run_id: str,
    request: Request,
    step_ref: Annotated[int, Body(embed=True)],
    x_access_code: Annotated[str | None, Header()] = None,
) -> dict:
    """The requested experiment drawn on this clip's own frame. Paid (Muse Image): judges and development only."""
    audience = audience_for(x_access_code)
    if audience == "public":
        raise HTTPException(status_code=403, detail="Picture guides are paused right now.")
    allow_visitor("picture", x_access_code, request.headers, request.client.host if request.client else None)
    result = _finished_result(run_id)
    run_dir = _run_file(run_id, "")
    try:
        # Its own budget run per request, like what-ifs; the image call also has its own daily cap.
        guide = make_setup_guide(client, run_dir, result, step_ref, f"guide-{run_id}-{uuid.uuid4().hex[:8]}", audience)
    except SetupGuideError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    item = guide["trail_item"]
    trail = [s for s in result.get("trail", []) if s.get("step") != "setup_guide"]
    result["trail"] = [*trail, item]
    saved = run_dir / "result.json"
    if saved.is_file() and guide["status"] == "picture":
        saved.write_text(json.dumps(result, default=str))
    out = {k: guide.get(k) for k in ("status", "instruction", "caption", "reason", "provider", "model", "planned_by",
                                     "step_ref", "trail_item")}
    if guide["status"] == "picture":
        out["image_url"] = f"/api/runs/{run_id}/setup-guide.jpg"
        out["frame_url"] = f"/api/runs/{run_id}/setup-guide-frame.jpg"
    return out


@app.get("/api/runs/{run_id}/setup-guide.jpg")
def get_setup_guide(run_id: str) -> FileResponse:
    path = _run_file(run_id, GUIDE_JPG)
    if not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/runs/{run_id}/setup-guide-frame.jpg")
def get_setup_guide_frame(run_id: str) -> FileResponse:
    path = _run_file(run_id, GUIDE_FRAME)
    if not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="image/jpeg")


def _visitor_may_ask(ws: WebSocket, init: dict) -> bool:
    """The per-visitor allowance for one spoken question; refusals reach the page as a notice, not a closed socket."""
    try:
        allow_visitor("question", init.get("access_code"), ws.headers, ws.client.host if ws.client else None)
    except HTTPException as e:
        asyncio.get_running_loop().create_task(ws.send_json({"type": "notice", "level": "error", "message": e.detail}))
        return False
    return True


@app.websocket("/ws/voice")
async def voice(ws: WebSocket) -> None:
    """First message: JSON {run_id, access_code?}. Then binary 24 kHz s16le mono PCM; text {"type": "stop"} ends it."""
    await ws.accept()
    if not await ws_allowed("voice", ws):
        return
    init = await _first_message(ws)
    if init is None:
        return
    run_id = str(init.get("run_id", ""))
    audience = audience_for(init.get("access_code"))
    state = runs.get(run_id) if run_id.replace("-", "").isalnum() else None
    if audience == "public" or state is None or state.result is None:
        await ws.send_json({"type": "notice", "level": "error",
                            "message": "Voice questions need a finished analysis, and are paused right now."})
        await ws.close()
        return
    result = state.result
    chunks: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def audio():
        while (chunk := await chunks.get()) is not None:
            yield chunk

    async def on_event(event: dict) -> None:
        turn = final_turn(event)
        if turn is not None:
            _, text = turn
            await ws.send_json({"type": "transcript", "text": text, "final": True})
            if not _visitor_may_ask(ws, init):
                return
            answer = await asyncio.to_thread(run_whatif, client, text, result,
                                             f"voice-{run_id}-{uuid.uuid4().hex[:8]}", audience)
            await ws.send_json({"type": "whatif", **answer})
            return
        partial = partial_text(event)
        if partial:
            await ws.send_json({"type": "transcript", "text": partial, "final": False})

    async def reader() -> None:
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await chunks.put(message["bytes"])
                elif message.get("text") and json.loads(message["text"]).get("type") == "stop":
                    break
        finally:
            await chunks.put(None)

    read_task = asyncio.create_task(reader())
    try:
        usage = await client.voice_transcribe(audio=audio(), on_event=on_event, run_id=f"voice-{run_id}",
                                              max_seconds=VOICE_MAX_SECONDS, audience=audience, keywords=VOICE_KEYWORDS)
        await ws.send_json({"type": "voice_done", "audio_seconds": round(usage.audio_seconds, 1)})
    except CostGuardError as e:
        await ws.send_json({"type": "notice", "level": "warning",
                            "message": str(e) if audience == "dev" else "Voice is temporarily busy, please type your question."})
    except WebSocketDisconnect:
        pass
    finally:
        read_task.cancel()


EXPORTS = {
    "json": ("application/json", "json"),
    "py": ("text/x-python", "py"),
    "mjcf": ("application/xml", "xml"),
    "report": ("text/markdown", "md"),
    "contact": ("application/xml", "xml"),
    "urdf": ("application/xml", "urdf"),
    "sdf": ("application/xml", "sdf"),
}


@app.get("/api/runs/{run_id}/export/{kind}")
def export(run_id: str, kind: str) -> Response:
    """Parameters (JSON), a standalone Python simulation, or a MuJoCo model of the fitted system."""
    if kind not in EXPORTS:
        raise HTTPException(status_code=404)
    result = _finished_result(run_id)
    if kind == "report":  # every verdict gets a report, including "no clean equation"
        body = report_markdown(result)
    elif kind == "contact":
        c = (result.get("bounces") or {}).get("mujoco_contact")
        if not c:
            raise HTTPException(status_code=409, detail="No verified bounces to export as a contact model.")
        body = bounce_mjcf(result["bounces"]["g"], result["bounces"]["e_mean"], c["radius_m"], c["dampratio"])
    elif "fit_state" not in result:
        raise HTTPException(status_code=409, detail="No fitted model to export for this clip.")
    elif kind == "json":
        body = parameters_json(result)
    elif kind == "py":
        body = python_script(result)
    elif kind in ("urdf", "sdf"):
        try:
            body = (urdf if kind == "urdf" else sdf)(result["fit_state"])
        except RobotExportError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
    else:
        xml, notes = build_mjcf(result["fit_state"])
        if result.get("scale_note"):
            notes = [*notes, "No scale reference in the clip: lengths in this model are in arbitrary units."]
        body = xml if not notes else xml.replace("<worldbody>", "<!-- " + " ".join(notes).replace("--", "-") + " -->\n  <worldbody>", 1)
    media, ext = EXPORTS[kind]
    return Response(body, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="physicslens-{run_id}.{ext}"'})


SIM2REAL_VERSION = 1  # bump when the computation changes, so cached copies are recomputed


@app.get("/api/runs/{run_id}/sim2real")
def sim_to_real(run_id: str) -> dict:
    """Textbook physics against the identified model on the real video, plus a simulated control study.

    Local math and MuJoCo only (no model calls); computed on first request and cached next to the result.
    """
    result = _finished_result(run_id)
    result_file = _run_file(run_id, "result.json")
    # Gallery entries live in the tracked repo; their cache goes to the runtime directory instead.
    cache_dir = runs.root / run_id if result_file.parent.parent == runs.gallery_root else result_file.parent
    cache = cache_dir / "sim2real.json"
    if cache.is_file() and not (result_file.is_file() and result_file.stat().st_mtime > cache.stat().st_mtime):
        cached = json.loads(cache.read_text())
        if cached.get("version") == SIM2REAL_VERSION:
            return cached
    try:
        out = {"version": SIM2REAL_VERSION, **sim2real(result)}
    except Sim2RealError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


@app.post("/api/runs/{run_id}/control")
def start_control(run_id: str, task: ControlTask, request: Request,
                  x_access_code: Annotated[str | None, Header()] = None) -> dict:
    """Muse Wonder writes a controller for the identified system; MuJoCo decides whether it works. Judges and dev only."""
    audience = audience_for(x_access_code)
    if audience == "public":
        raise HTTPException(status_code=403, detail="Controller design is paused right now.")
    allow_visitor("controller", x_access_code, request.headers, request.client.host if request.client else None)
    _run_file(run_id, "result.json")  # rejects malformed ids
    result = _finished_result(run_id)
    try:
        return controls.start(run_id, result, task, audience).view()
    except ControlUnsupported as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@app.get("/api/runs/{run_id}/control/{task_id}")
def get_control(run_id: str, task_id: str, x_access_code: Annotated[str | None, Header()] = None) -> dict:
    if audience_for(x_access_code) == "public":
        raise HTTPException(status_code=403, detail="Controller design is paused right now.")
    _run_file(run_id, "result.json")
    state = controls.get(run_id, task_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown controller task.")
    return state.view()


@app.post("/api/compare")
def compare_runs(
    baseline: Annotated[str, Body()],
    current: Annotated[str, Body()],
) -> dict:
    """Machine-health drift between two finished analyses of the same system. Local math only: free."""
    a, b = _finished_result(baseline), _finished_result(current)
    if a.get("family") != b.get("family"):
        raise HTTPException(status_code=409, detail=f"Can't compare a {a.get('family')} with a {b.get('family')}.")
    out = compare(_health(baseline), _health(current))
    out["runs"] = {"baseline": {"id": baseline, "object": a.get("scene", {}).get("moving_object")},
                   "current": {"id": current, "object": b.get("scene", {}).get("moving_object")}}
    return out


def _health(run_id: str) -> HealthMetrics:
    """Health metrics of one finished run, computed once (about 7 s) and cached beside its result."""
    cache = _run_file(run_id, "health.json")
    if cache.is_file():
        return metrics_from_dict(json.loads(cache.read_text()))
    result = _finished_result(run_id)
    try:
        m = health_metrics(trajectory_from_result(result), result["family"])
    except (HealthError, AnalysisError) as e:
        raise HTTPException(status_code=409, detail=f"{run_id}: {e}") from e
    cache.write_text(json.dumps(metrics_to_dict(m)))
    return m


@app.post("/api/health/trend")
def health_trend(runs: Annotated[list[str], Body(embed=True)]) -> dict:
    """Health metrics across 3 to 12 recordings in the given order, each checked against the first."""
    if not 3 <= len(runs) <= 12:
        raise HTTPException(status_code=400, detail="Give between 3 and 12 recordings, oldest first.")
    series = [_health(r) for r in runs]
    try:
        out = trend(series)
    except HealthError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    out["recordings"] = []
    for run_id, m in zip(runs, series, strict=True):
        vs = compare(series[0], m)
        flags = {row["metric"]: row["flagged"] for row in vs["metrics"]}
        out["recordings"].append({
            "id": run_id,
            "natural_freq_hz": [m.natural_freq_hz.value, m.natural_freq_hz.std],
            "decay_rate_per_s": [m.decay_rate_per_s.value, m.decay_rate_per_s.std],
            "flagged": flags,
        })
    return out


# ---------- goal-driven studies: several recordings toward one deliverable ----------

def _study_audience(x_access_code: str | None) -> str:
    audience = audience_for(x_access_code)
    if audience == "public":
        raise HTTPException(status_code=403, detail="Studies are paused right now.")
    return audience


def _result_or_none(run_id: str) -> dict | None:
    state = runs.get(run_id)
    if state is None or state.status in ("queued", "running"):
        return None
    if state.status == "error":
        return {"status": "error", "reason": state.error or "The analysis failed."}
    return state.result


def _health_series(run_ids: list[str]) -> dict | None:
    """Wear verdict across a study's recordings: two are compared directly, three or more get a trend."""
    try:
        series = [_health(r) for r in run_ids]
        if len(series) == 2:
            return compare(series[0], series[1])
        return trend(series[:12])
    except (HTTPException, HealthError) as e:
        log.warning("Study health check skipped: %s", e)
        return None


def _study_call(fn, *args):
    try:
        return fn(*args)
    except StudyError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except CostGuardError as e:
        detail = str(e) if settings.app_env == "dev" else "Temporarily busy, please retry later."
        raise HTTPException(status_code=503, detail=detail) from e
    except StudyLLMError as e:
        raise HTTPException(status_code=502, detail="The reasoning model gave no usable answer; please try again.") from e


@app.post("/api/studies")
def create_study(goal: Annotated[str, Body(embed=True)], request: Request,
                 x_access_code: Annotated[str | None, Header()] = None) -> dict:
    limit_requests("studies", request)
    audience = _study_audience(x_access_code)
    allow_visitor("study", x_access_code, request.headers, request.client.host if request.client else None)
    return _study_call(studies.create, goal, audience)


@app.get("/api/studies/{study_id}")
def get_study(study_id: str) -> dict:
    return _study_call(studies.get, study_id)


@app.post("/api/studies/{study_id}/runs")
def add_study_run(
    study_id: str,
    request: Request,
    video: Annotated[UploadFile, File()],
    label: Annotated[str, Form(max_length=60)] = "",
    pendulum_length_m: Annotated[float | None, Form()] = None,
    object_size_m: Annotated[float | None, Form()] = None,
    slow_motion: Annotated[float, Form()] = 1.0,
    object_hint: Annotated[str | None, Form(max_length=60)] = None,
    setting: Annotated[float | None, Form(gt=0)] = None,
    x_access_code: Annotated[str | None, Header()] = None,
) -> dict:
    limit_requests("studies", request)
    _study_audience(x_access_code)
    _study_call(studies.get, study_id)  # fail before accepting the upload
    run = create_run(request, video, pendulum_length_m, object_size_m, slow_motion, object_hint, None, x_access_code, None)
    study = _study_call(studies.attach, study_id, run["run_id"], label, setting)
    return {"study": study, "run": run}


@app.post("/api/studies/{study_id}/review")
def review_study(study_id: str, request: Request, x_access_code: Annotated[str | None, Header()] = None) -> dict:
    limit_requests("studies", request)
    return _study_call(studies.review, study_id, _result_or_none, _health_series, _study_audience(x_access_code))


@app.get("/api/studies/{study_id}/report")
def study_report(study_id: str) -> Response:
    body = _study_call(studies.report, study_id, _result_or_none)
    return Response(body, media_type="text/markdown",
                    headers={"Content-Disposition": f'attachment; filename="physicslens-study-{study_id}.md"'})


@app.get("/api/studies/{study_id}/mjcf")
def study_mjcf(study_id: str) -> Response:
    body = _study_call(studies.mjcf, study_id, _result_or_none)
    return Response(body, media_type="application/xml",
                    headers={"Content-Disposition": f'attachment; filename="physicslens-study-{study_id}.xml"'})


NOTES_KEYWORDS = ["string", "length", "centimeters", "meters", "inches", "feet", "slow motion", "front", "left", "right"]


@app.websocket("/ws/voice/notes")
async def voice_notes(ws: WebSocket) -> None:
    """Spoken settings while filming. First message: JSON {access_code?}; then 24 kHz s16le mono PCM; {"type": "stop"}."""
    await ws.accept()
    if not await ws_allowed("voice", ws):
        return
    init = await _first_message(ws)
    if init is None:
        return
    audience = audience_for(init.get("access_code"))
    if audience == "public":
        await ws.send_json({"type": "notice", "level": "error", "message": "Voice is paused right now."})
        await ws.close()
        return
    chunks: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def audio():
        while (chunk := await chunks.get()) is not None:
            yield chunk

    async def on_event(event: dict) -> None:
        turn = final_turn(event)
        if turn is not None:
            _, text = turn
            await ws.send_json({"type": "transcript", "text": text, "final": True})
            if not _visitor_may_ask(ws, init):
                return
            notes = await asyncio.to_thread(interpret_notes, client, text, f"notes-{uuid.uuid4().hex[:8]}", audience)
            await ws.send_json({"type": "notes", **notes})
            return
        partial = partial_text(event)
        if partial:
            await ws.send_json({"type": "transcript", "text": partial, "final": False})

    async def reader() -> None:
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await chunks.put(message["bytes"])
                elif message.get("text") and json.loads(message["text"]).get("type") == "stop":
                    break
        finally:
            await chunks.put(None)

    read_task = asyncio.create_task(reader())
    try:
        await client.voice_transcribe(audio=audio(), on_event=on_event, run_id=f"notes-{uuid.uuid4().hex[:8]}",
                                      max_seconds=VOICE_MAX_SECONDS, audience=audience, keywords=NOTES_KEYWORDS)
        await ws.send_json({"type": "voice_done"})
    except CostGuardError as e:
        await ws.send_json({"type": "notice", "level": "warning",
                            "message": str(e) if audience == "dev" else "Voice is temporarily busy; please type the values."})
    except WebSocketDisconnect:
        pass
    finally:
        read_task.cancel()
