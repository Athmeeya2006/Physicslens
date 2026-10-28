"""FastAPI entry point."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Annotated

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
from .agent.notes import interpret_notes
from .agent.setup_guide import GUIDE_FRAME, GUIDE_JPG, SetupGuideError, make_setup_guide
from .agent.whatif import run_whatif
from .costguard import CostGuardClient, CostGuardError
from .export.mjcf import bounce_mjcf, build_mjcf
from .export.mp4 import ExportError, render_share_video
from .export.report import report_markdown
from .export.script import parameters_json, python_script
from .live import LiveConfig, LiveSession
from .physics.health import (
    HealthError,
    HealthMetrics,
    compare,
    health_metrics,
    metrics_from_dict,
    metrics_to_dict,
    trend,
)
from .pipeline import AnalysisError, RunOptions, trajectory_from_result
from .public import PublicQuota, client_address, turnstile_ok
from .runs import RunManager
from .settings import load_settings
from .studies import Studies, StudyError
from .voice.events import final_turn, partial_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Raises before the server accepts a request if MOCK_MODELS=1 in demo/prod or budget.yaml is missing.
log = logging.getLogger("physicslens.app")
settings = load_settings()
client = CostGuardClient(settings)
runs = RunManager(settings, client)
studies = Studies(settings.data_dir / "studies", client)
public_quota = PublicQuota(settings.data_dir / "public.db", int(client.guard.budget.get("public_per_ip_daily_analyses", 3)))

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


def public_try_enabled() -> bool:
    return os.environ.get("PUBLIC_TRY_ENABLED", "0") == "1"


@app.get("/api/health")
def health(x_access_code: Annotated[str | None, Header()] = None) -> dict[str, object]:
    # mock_mode drives the red "MOCK MODE" banner; audience decides what the page offers (never any quota numbers).
    audience = audience_for(x_access_code)
    return {"status": "ok", "app_env": settings.app_env, "mock_mode": settings.mock_models, "audience": audience,
            "can_upload": audience != "public" or public_try_enabled(), "can_go_live": audience != "public",
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
) -> dict[str, object]:
    audience = audience_for(x_access_code)
    for name, value in (("pendulum length", pendulum_length_m), ("object size", object_size_m)):
        if value is not None and not 0.005 <= value <= 100:
            raise HTTPException(status_code=400, detail=f"The {name} should be between 0.5 cm and 100 m.")
    if not 1 <= slow_motion <= 1000:
        raise HTTPException(status_code=400, detail="The slow-motion factor should be between 1 (normal speed) and 1000.")
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Please upload a video file (MP4 or MOV).")
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
    state = runs.start(dest, RunOptions(audience=audience, pendulum_length_m=pendulum_length_m, object_size_m=object_size_m,
                                         slow_motion=slow_motion, object_hint=object_hint))
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
                elif cmd.get("type") == "family":
                    session.choose_family_now(str(cmd.get("family", "")))
                elif cmd.get("type") == "length" and 0.005 <= float(cmd.get("meters", 0)) <= 100:
                    session.set_length(float(cmd["meters"]))
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
    question: Annotated[str, Body(embed=True)],
    x_access_code: Annotated[str | None, Header()] = None,
) -> dict:
    """Typed what-if (the voice path ends here too). Judges and development only, like voice."""
    audience = audience_for(x_access_code)
    if audience == "public":
        raise HTTPException(status_code=403, detail="What-if questions need the access code.")
    result = _finished_result(run_id)
    try:
        # One budget "run" per question: a shared id would fill the lifetime per-run cap and lock everyone out.
        return run_whatif(client, question[:500], result, f"whatif-{run_id}-{uuid.uuid4().hex[:8]}", audience)
    except CostGuardError as e:
        detail = str(e) if settings.app_env == "dev" else "Temporarily busy, please retry later."
        raise HTTPException(status_code=503, detail=detail) from e


@app.post("/api/runs/{run_id}/setup-guide")
def setup_guide(
    run_id: str,
    step_ref: Annotated[int, Body(embed=True)],
    x_access_code: Annotated[str | None, Header()] = None,
) -> dict:
    """The requested experiment drawn on this clip's own frame. Paid (Muse Image): judges and development only."""
    audience = audience_for(x_access_code)
    if audience == "public":
        raise HTTPException(status_code=403, detail="Picture guides need the access code.")
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


@app.websocket("/ws/voice")
async def voice(ws: WebSocket) -> None:
    """First message: JSON {run_id, access_code?}. Then binary 24 kHz s16le mono PCM; text {"type": "stop"} ends it."""
    await ws.accept()
    init = await ws.receive_json()
    run_id = str(init.get("run_id", ""))
    audience = audience_for(init.get("access_code"))
    state = runs.get(run_id) if run_id.replace("-", "").isalnum() else None
    if audience == "public" or state is None or state.result is None:
        await ws.send_json({"type": "notice", "level": "error",
                            "message": "Voice questions need a finished analysis and the access code."})
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
    else:
        xml, notes = build_mjcf(result["fit_state"])
        if result.get("scale_note"):
            notes = [*notes, "No scale reference in the clip: lengths in this model are in arbitrary units."]
        body = xml if not notes else xml.replace("<worldbody>", "<!-- " + " ".join(notes).replace("--", "-") + " -->\n  <worldbody>", 1)
    media, ext = EXPORTS[kind]
    return Response(body, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="physicslens-{run_id}.{ext}"'})


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
        raise HTTPException(status_code=403, detail="Studies need the access code.")
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
def create_study(goal: Annotated[str, Body(embed=True)], x_access_code: Annotated[str | None, Header()] = None) -> dict:
    return _study_call(studies.create, goal, _study_audience(x_access_code))


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
    x_access_code: Annotated[str | None, Header()] = None,
) -> dict:
    _study_audience(x_access_code)
    _study_call(studies.get, study_id)  # fail before accepting the upload
    run = create_run(request, video, pendulum_length_m, object_size_m, slow_motion, object_hint, None, x_access_code, None)
    study = _study_call(studies.attach, study_id, run["run_id"], label)
    return {"study": study, "run": run}


@app.post("/api/studies/{study_id}/review")
def review_study(study_id: str, x_access_code: Annotated[str | None, Header()] = None) -> dict:
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
    init = await ws.receive_json()
    audience = audience_for(init.get("access_code"))
    if audience == "public":
        await ws.send_json({"type": "notice", "level": "error", "message": "Voice needs the access code."})
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
