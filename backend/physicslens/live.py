"""Live mode: JPEG frames in, SAM chunks out, rolling refits, ghost predictions sent back to the phone."""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import meta_sam_parser as msp
import numpy as np

from .agent.reasoning import propose_hypotheses, understand_scene
from .agent.schemas import Scene
from .costguard import CostGuardClient, CostGuardError
from .costguard.cache import file_digest
from .perception.calibration import (
    Calibration,
    CalibrationError,
    from_known_length,
    from_reference,
)
from .perception.sam_track import ObjectTrack, tracks_from_text
from .physics.coords import fit_circle, to_coordinates
from .physics.fit import FitResult, fit_model, simulate
from .physics.models import MODELS, models_for
from .physics.verify import assess
from .pipeline import MOVING_ID, REFERENCE_ID, SAM_METADATA
from .sigproc.cleanup import fill_short_gaps
from .sigproc.summary import summarize
from .sigproc.trajectory import from_pixels

log = logging.getLogger("physicslens.live")

WINDOW_S = 10.0  # rolling buffer the equation is fitted on
MIN_FIT_S = 2.5  # don't fit before this much track exists
PREDICT_S = 3.0  # ghost extends this far past the newest tracked frame, covering SAM's latency
HISTORY = 6  # recent fits used to show how well the parameters have converged

Send = Callable[[dict[str, Any]], None]
SamFn = Callable[[Path, str, int], str]  # (chunk mp4, prompt, frames) -> SAM output text


@dataclass
class LiveConfig:
    fps: float = 25.0
    pendulum_length_m: float | None = None
    audience: str = "dev"
    chunk_frames: int = 50
    run_cap_usd: float | None = None  # live sessions have their own cap (budget.yaml live_session_cap_usd)


@dataclass
class _Track:
    frames: list[int] = field(default_factory=list)
    centers: list[tuple[float, float]] = field(default_factory=list)
    sizes: list[float] = field(default_factory=list)


class LiveSession:
    def __init__(self, client: CostGuardClient, run_dir: Path, run_id: str, config: LiveConfig, send: Send,
                 sam_fn: SamFn | None = None, synchronous: bool = False):
        self.client = client
        self.run_dir = run_dir
        self.run_id = run_id
        self.cfg = config
        self.send = send
        self.sam_fn = sam_fn or self._sam_via_client
        self.synchronous = synchronous
        run_dir.mkdir(parents=True, exist_ok=True)

        self._pending: list[bytes] = []
        self._pending_start = 0
        self._next_index = 0
        self.size: tuple[int, int] | None = None
        self.scene: Scene | None = None
        self.calibration: Calibration | None = None
        self.track = _Track()
        self.last_box: msp.PromptBox | None = None
        self.family: str | None = None
        self.candidates: list[str] = []
        self.warm: dict[str, dict[str, float]] = {}
        self.history: list[dict[str, float]] = []
        self.paused_reason: str | None = None
        self._chunks: queue.Queue[tuple[int, list[bytes]] | None] = queue.Queue()
        self._fit_lock = threading.Lock()
        self._worker = None if synchronous else threading.Thread(target=self._chunk_loop, daemon=True)
        if self._worker:
            self._worker.start()

    # ---------- input ----------

    def add_frame(self, jpeg: bytes) -> int:
        """Frames are numbered in arrival order; the phone counts the same way, so indices line up."""
        idx = self._next_index
        self._next_index += 1
        if self.size is None:
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("First frame is not a JPEG.")
            self.size = (img.shape[1] - img.shape[1] % 2, img.shape[0] - img.shape[0] % 2)
        if self.paused_reason:
            return idx
        if not self._pending:
            self._pending_start = idx
        self._pending.append(jpeg)
        if len(self._pending) >= self.cfg.chunk_frames:
            chunk, start = self._pending, self._pending_start
            self._pending = []
            if self.synchronous:
                self._process_chunk(start, chunk)
            else:
                self._chunks.put((start, chunk))
        return idx

    def close(self) -> None:
        self._chunks.put(None)

    # ---------- chunk processing (one worker thread, chunks in order) ----------

    def _chunk_loop(self) -> None:
        while (item := self._chunks.get()) is not None:
            self._process_chunk(*item)

    def _process_chunk(self, start: int, jpegs: list[bytes]) -> None:
        try:
            if self.scene is None:
                self._understand(jpegs)
            path = self._encode(start, jpegs)
            prompt = self._prompt()
            text = self.sam_fn(path, prompt, len(jpegs))
            tracks = tracks_from_text(text) if text.strip() else {}
            self._ingest(start, len(jpegs), tracks)
            self._maybe_refit()
        except CostGuardError as e:
            # Budget or rate limit mid-session: stop paid calls, keep the session (and the UI) alive.
            self.paused_reason = str(e)
            self.send({"type": "notice", "level": "warning",
                       "message": "Tracking paused: " + (str(e) if self.cfg.audience == "dev" else "temporarily busy, please retry later.")})
        except Exception as e:
            log.exception("Live chunk at frame %d failed", start)
            self.send({"type": "notice", "level": "warning", "message": f"Skipped a chunk: {e}"})

    def _understand(self, jpegs: list[bytes]) -> None:
        picks = [jpegs[0], *jpegs[len(jpegs) // 3::max(1, len(jpegs) // 3)]][:4]
        keyframes = [_downscale(j, 768) for j in picks]
        self.scene, call = understand_scene(self.client, keyframes, self.run_id, self.cfg.audience, self.cfg.run_cap_usd)
        assert self.size is not None
        self.last_box = _to_box(self.scene.moving_object_box, *self.size)
        self.send({"type": "scene", "object": self.scene.moving_object, "family": self.scene.motion_family,
                   "reference": self.scene.reference_kind, "model": call.model, "provider": call.provider})

    def _prompt(self) -> str:
        assert self.size is not None and self.last_box is not None and self.scene is not None
        objects = [msp.PromptObject(id=int(MOVING_ID), prompts=[msp.FramePrompt(frame=0, box=self.last_box)])]
        if self.calibration is None and self.scene.reference_box is not None:
            objects.append(msp.PromptObject(id=int(REFERENCE_ID), prompts=[
                msp.FramePrompt(frame=0, box=_to_box(self.scene.reference_box, *self.size))]))
        return msp.build_object_prompt(size=self.size, objects=objects)

    def _encode(self, start: int, jpegs: list[bytes]) -> Path:
        assert self.size is not None
        out = self.run_dir / f"chunk_{start:06d}.mp4"
        w, h = self.size
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", f"{self.cfg.fps:g}", "-i", "-",
             "-vf", f"crop={w}:{h}:0:0", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(out)],
            input=b"".join(jpegs), check=True, capture_output=True,
        )
        return out

    def _sam_via_client(self, path: Path, prompt: str, frames: int) -> str:
        r = self.client.sam_video(prompt=prompt, video_path=path, video_hash=file_digest(path), est_frames=frames,
                                  metadata=SAM_METADATA, run_id=self.run_id, audience=self.cfg.audience,
                                  run_cap_usd=self.cfg.run_cap_usd)
        return r.text

    def _ingest(self, start: int, n: int, tracks: dict[str, ObjectTrack]) -> None:
        if self.calibration is None:
            self.calibration = self._calibrate(tracks.get(REFERENCE_ID), tracks.get(MOVING_ID))
        moving = tracks.get(MOVING_ID)
        if moving is None:
            self.send({"type": "notice", "level": "info", "message": "Lost the object in this chunk."})
            return
        keep = moving.frames < n  # chunk-local indices only; anything else would be misplaced in time
        points = []
        for f, (x, y), size in zip(moving.frames[keep], moving.centers_px[keep], moving.long_px[keep], strict=True):
            self.track.frames.append(start + int(f))
            self.track.centers.append((float(x), float(y)))
            self.track.sizes.append(float(size))
            points.append([start + int(f), round(float(x), 2), round(float(y), 2)])
        # Carry identity into the next chunk: its frame 0 is prompted with this chunk's final box.
        last = int(np.argmax(moving.frames[keep])) if keep.any() else None
        if last is not None:
            cx, cy = moving.centers_px[keep][last]
            half = moving.long_px[keep][last] / 2
            w, h = self.size or (0, 0)
            self.last_box = msp.PromptBox(max(0, int(cx - half)), max(0, int(cy - half)),
                                          min(w, int(cx + half) + 1), min(h, int(cy + half) + 1))
        self.send({"type": "track", "points": points,
                   "object_size_px": float(np.median(self.track.sizes)) if self.track.sizes else None})

    def _calibrate(self, reference: ObjectTrack | None, moving: ObjectTrack | None) -> Calibration | None:
        if reference is not None and self.scene is not None and self.scene.reference_kind != "none":
            try:
                return from_reference(reference, self.scene.reference_kind)
            except CalibrationError as e:
                self.send({"type": "notice", "level": "warning", "message": str(e)})
        return None  # pendulum length calibration needs more swing; done at fit time

    # ---------- fitting ----------

    def _maybe_refit(self) -> None:
        if self.synchronous:
            self.refit()
        elif self._fit_lock.acquire(blocking=False):
            # Refits run off the chunk thread; if one is still running, this chunk's data waits for the next.
            def run() -> None:
                try:
                    self.refit()
                finally:
                    self._fit_lock.release()
            threading.Thread(target=run, daemon=True).start()

    def refit(self) -> dict[str, Any] | None:
        fps = self.cfg.fps
        if len(self.track.frames) < MIN_FIT_S * fps:
            return None
        frames = np.array(self.track.frames)
        centers = np.array(self.track.centers)
        recent = frames >= frames[-1] - WINDOW_S * fps
        segments = fill_short_gaps(frames[recent], centers[recent], max_gap=3)
        seg_frames, seg_centers = max(segments, key=lambda s: len(s[0]))
        if len(seg_frames) < MIN_FIT_S * fps:
            return None

        ppm = self._scale(seg_centers)
        if ppm is None:
            self.send({"type": "notice", "level": "warning",
                       "message": "No scale yet: show an A4 sheet in the swing plane, or enter the pendulum length."})
            return None
        traj = from_pixels(seg_frames, seg_centers, fps, ppm)
        origin = seg_centers[0]

        if self.family is None:
            self._choose_family(traj)
        try:
            coords = to_coordinates(traj, self.family)  # type: ignore[arg-type]
        except ValueError as e:
            self.send({"type": "fit", "status": "no_fit", "reason": str(e)})
            return None
        fits = [fit_model(MODELS[m], coords, warm=self.warm.get(m)) for m in self.candidates]
        for f in fits:
            if f.success:
                self.warm[f.model_id] = f.params
        verdict = assess(coords, fits)
        lead: FitResult | None = verdict.winner or (verdict.contenders[0] if verdict.contenders else None)
        msg: dict[str, Any] = {"type": "fit", "status": verdict.status, "reason": verdict.reason,
                               "samples": len(traj), "window_s": round(float(traj.t[-1]), 2)}
        if lead is not None:
            spec = MODELS[lead.model_id]
            self.history = [*self.history, lead.params][-HISTORY:]
            same = [h for h in self.history if set(h) == set(lead.params)]
            msg["equation"] = {"model_id": spec.id, "title": spec.title, "latex": spec.latex}
            msg["parameters"] = [
                {"name": p.name, "latex": p.latex, "unit": p.unit, "value": lead.params[p.name],
                 "spread": float(np.std([h[p.name] for h in same])) if len(same) >= 3 else None}
                for p in spec.params
            ]
            msg["ghost"] = self._prediction(spec, lead, coords, seg_frames, origin, ppm)
        self.send(msg)
        return msg

    def _scale(self, centers: np.ndarray) -> float | None:
        if self.calibration is not None:
            return self.calibration.px_per_m
        if self.cfg.pendulum_length_m:
            radius_px = fit_circle(centers * np.array([1.0, -1.0]))[2]
            # Recomputed while the arc is short; frozen once it is measured over a whole swing.
            cal = from_known_length(radius_px, self.cfg.pendulum_length_m)
            if len(centers) >= 2 * MIN_FIT_S * self.cfg.fps:
                self.calibration = cal
            return cal.px_per_m
        return None

    def _choose_family(self, traj) -> None:
        """One LLM call per session (hypotheses); afterwards refits are local math only."""
        assert self.scene is not None
        try:
            hyp, call = propose_hypotheses(self.client, summarize(traj), self.scene, self.run_id, self.cfg.audience,
                                           run_cap_usd=self.cfg.run_cap_usd)
            self.family = hyp.family
            ids = [h.model_id for h in hyp.candidates]
            self.send({"type": "hypotheses", "family": hyp.family, "model": call.model, "provider": call.provider,
                       "items": [{"model_id": h.model_id, "title": MODELS[h.model_id].title, "why": h.rationale}
                                 for h in hyp.candidates]})
        except CostGuardError as e:
            # No LLM budget: fall back to the scene's family and its whole model library, and say so.
            self.family = self.scene.motion_family if self.scene.motion_family != "other" else "pendulum"
            ids = []
            self.send({"type": "notice", "level": "warning", "message": f"Hypotheses unavailable ({e}); testing every {self.family} model."})
        simplest = min(models_for(self.family), key=lambda m: len(m.params)).id
        self.candidates = list(dict.fromkeys([simplest, *(ids or [m.id for m in models_for(self.family)])]))

    def _prediction(self, spec, lead: FitResult, coords, frames: np.ndarray, origin: np.ndarray, ppm: float) -> list[list[float]]:
        fps = self.cfg.fps
        extra = int(PREDICT_S * fps)
        t = np.arange(len(frames) + extra) / fps
        q = simulate(spec, lead.params, lead.y0, t, coords.ctx)
        if q is None:
            return []
        xy = coords.to_xy(q)
        f0 = int(frames[0])
        px = origin[0] + xy[:, 0] * ppm
        py = origin[1] - xy[:, 1] * ppm
        return [[f0 + i, round(float(x), 2), round(float(y), 2)] for i, (x, y) in enumerate(zip(px, py, strict=True))]


def _to_box(box, width: int, height: int) -> msp.PromptBox:
    x1, y1, x2, y2 = box.to_pixels(width, height)
    return msp.PromptBox(x1, y1, max(x2, x1 + 1), max(y2, y1 + 1))


def _downscale(jpeg: bytes, long_side: int) -> bytes:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    s = min(1.0, long_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else jpeg
