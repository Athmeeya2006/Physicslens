"""Live mode: instant local tracking for the overlay, SAM 3.1 refining it in parallel chunks, rolling refits.

Measured on the first real session: SAM answers a 2 s chunk about 12 to 16 s later (~5 frames/s of
throughput), so it cannot drive a live overlay on its own. A template tracker follows the object on every
frame immediately; SAM runs on every second frame, several chunks at once, and its positions replace the
local ones as they arrive. If SAM shows the local tracker drifted, the tracker is re-anchored.
"""

from __future__ import annotations

import base64
import logging
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import meta_sam_parser as msp
import numpy as np

from .agent.llm import LLMError
from .agent.reasoning import propose_hypotheses, understand_scene
from .agent.schemas import Scene
from .costguard import CostGuardClient, CostGuardError, ProviderCallFailed
from .costguard.cache import file_digest
from .perception.calibration import (
    Calibration,
    CalibrationError,
    from_known_length,
    from_reference,
)
from .perception.local_track import TemplateTracker, box_around_tap, to_gray
from .perception.sam_track import ObjectTrack, tracks_from_text
from .physics.coords import fit_circle, to_coordinates
from .physics.fit import FitResult, fit_model, simulate
from .physics.models import MODELS, models_for
from .physics.regime import detect_change
from .physics.verify import assess
from .pipeline import MOVING_ID, REFERENCE_ID, SAM_METADATA
from .sigproc.cleanup import fill_short_gaps
from .sigproc.summary import summarize
from .sigproc.trajectory import from_pixels

log = logging.getLogger("physicslens.live")

WINDOW_S = 10.0  # rolling buffer the equation is fitted on
MIN_FIT_S = 2.5  # don't fit before this much track exists
PREDICT_S = 3.0  # ghost extends this far past the newest tracked frame
HISTORY = 6  # recent fits used to show how well the parameters have converged
SCENE_AFTER_FRAMES = 25  # ask the scene model after ~1 s of video
REFIT_EVERY = 12  # frames of new local track between refits (~0.5 s)
LOCAL_BATCH = 5  # local points are sent in small batches
BUFFER_S = 40.0  # frames kept for chunk encoding and tracker re-anchoring
DRIFT_FRACTION = 0.5  # re-anchor when local and SAM disagree by more than this share of the object size
SAM_WORKERS = 4
LLM_MIN_INTERVAL_S = 5.0  # live mode never calls the language model more often than this

Send = Callable[[dict[str, Any]], None]
SamFn = Callable[[Path, str, list[int]], str]  # (chunk mp4, prompt, global frame indices) -> SAM output text


@dataclass
class LiveConfig:
    fps: float = 25.0
    pendulum_length_m: float | None = None
    audience: str = "dev"
    chunk_frames: int = 50  # frames per SAM request (its billing minimum)
    sam_stride: int = 2  # SAM sees every n-th frame: halves cost, plenty of samples for the fit
    run_cap_usd: float | None = None


class LiveSession:
    def __init__(self, client: CostGuardClient, run_dir: Path, run_id: str, config: LiveConfig, send: Send,
                 sam_fn: SamFn | None = None, synchronous: bool = False):
        self.client = client
        self.run_dir = run_dir
        self.run_id = run_id
        self.cfg = config
        self.send = send
        self.sam_fn = sam_fn or self._sam_via_client
        self.sync = synchronous
        run_dir.mkdir(parents=True, exist_ok=True)

        self.lock = threading.RLock()
        self.frames: dict[int, bytes] = {}
        self.latest = -1
        self.size: tuple[int, int] | None = None
        self.scene: Scene | None = None
        self.calibration: Calibration | None = None
        self.tracker: TemplateTracker | None = None
        self.local_next = 0  # next frame index the local tracker will process
        self.local: dict[int, tuple[float, float]] = {}
        self.sam: dict[int, tuple[float, float]] = {}
        self.object_size: float | None = None
        self.next_chunk = 0
        self.chunk_span = config.chunk_frames * config.sam_stride
        self.reanchor: tuple[int, tuple[int, int, int, int]] | None = None
        self.use_data_url = True
        self.family: str | None = None
        self.candidates: list[str] = []
        self.warm: dict[str, dict[str, float]] = {}
        self.history: list[dict[str, float]] = []
        self.paused_reason: str | None = None
        self.regime_start = 0  # frames before this belong to an earlier regime and are not fitted
        self._last_llm = 0.0
        self._scene_started = False
        self.selected_by_tap = False
        self._since_refit = 0
        self._pending_local: list[list[float]] = []
        self._refit_busy = threading.Lock()
        self._wake = threading.Event()
        self._closed = False
        self._pool = None if synchronous else ThreadPoolExecutor(max_workers=SAM_WORKERS)
        self._local_thread = None if synchronous else threading.Thread(target=self._local_loop, daemon=True)
        if self._local_thread:
            self._local_thread.start()

    # ---------- input ----------

    def add_frame(self, jpeg: bytes) -> int:
        """Frames are numbered in arrival order; the phone counts the same way, so indices line up."""
        if self.size is None:
            gray = to_gray(jpeg)
            self.size = (gray.shape[1] - gray.shape[1] % 2, gray.shape[0] - gray.shape[0] % 2)
        with self.lock:
            self.latest += 1
            idx = self.latest
            self.frames[idx] = jpeg
            for old in [k for k in self.frames if k < idx - BUFFER_S * self.cfg.fps]:
                del self.frames[old]
        if not self._scene_started and idx + 1 >= SCENE_AFTER_FRAMES:
            self._scene_started = True
            self._run(self._understand)
        if self.sync:
            self._local_step()
        else:
            self._wake.set()
        return idx

    def select(self, x: float, y: float) -> None:
        """The user tapped the object: start tracking at the newest frame now, without waiting for the scene model."""
        with self.lock:
            idx = self.latest
            jpeg = self.frames.get(idx)
            if jpeg is None:
                return
            gray = to_gray(jpeg)
            box = box_around_tap(gray, x, y)
            self.tracker = TemplateTracker(gray, box)
            self.object_size = float(max(box[2] - box[0], box[3] - box[1]))
            # A new selection starts a new measurement: earlier tracks and fits belonged to whatever was tracked before.
            self.local = {idx: (self.tracker.cx, self.tracker.cy)}
            self.sam.clear()
            self.warm.clear()
            self.history.clear()
            self.local_next = idx + 1
            self.next_chunk = idx
            self.selected_by_tap = True
        self.send({"type": "selected", "frame": idx, "box": list(box)})
        if self.sync:
            self._local_step()
        else:
            self._wake.set()

    def close(self) -> None:
        self._closed = True
        self._wake.set()
        if self._pool:
            self._pool.shutdown(wait=False, cancel_futures=True)

    def _run(self, fn: Callable[[], None]) -> None:
        if self.sync:
            self._guarded(fn)()
        else:
            threading.Thread(target=self._guarded(fn), daemon=True).start()

    def _guarded(self, fn: Callable[[], Any]) -> Callable[[], None]:
        def run() -> None:
            try:
                fn()
            except CostGuardError as e:
                self._pause(e)
            except Exception as e:
                log.exception("Live step failed")
                self.send({"type": "notice", "level": "warning", "message": f"A background step failed: {e}"})
        return run

    def _pause(self, e: Exception) -> None:
        # Budget or rate limit: stop paid calls, keep local tracking and fitting alive.
        if self.paused_reason is None:
            self.paused_reason = str(e)
            text = str(e) if self.cfg.audience == "dev" else "temporarily busy, please retry later."
            self.send({"type": "notice", "level": "warning", "message": f"SAM refinement paused: {text} Local tracking continues."})

    # ---------- scene (one LLM call) ----------

    def _understand(self) -> None:
        with self.lock:
            first = self.frames.get(0)
            ordered = [self.frames[i] for i in sorted(self.frames)]
        if first is None:
            raise RuntimeError("First frame left the buffer before the scene was read.")
        step = max(1, len(ordered) // 4)
        keyframes = [_downscale(j, 768) for j in [first, *ordered[step::step]][:4]]
        self.scene, call = understand_scene(self.client, keyframes, self.run_id, self.cfg.audience, self.cfg.run_cap_usd)
        assert self.size is not None
        box = self.scene.moving_object_box.to_pixels(*self.size)
        with self.lock:
            if not self.selected_by_tap:  # a tap is the user's explicit choice and wins over the model's box
                self.tracker = TemplateTracker(to_gray(first), box)
                self.object_size = float(max(box[2] - box[0], box[3] - box[1]))
                self.local[0] = (self.tracker.cx, self.tracker.cy)
                self.local_next = 1
        self.send({"type": "scene", "object": self.scene.moving_object, "family": self.scene.motion_family,
                   "reference": self.scene.reference_kind, "model": call.model, "provider": call.provider})
        if self.sync:
            self._local_step()
        else:
            self._wake.set()

    # ---------- local tracking (every frame, immediately) ----------

    def _local_loop(self) -> None:
        while not self._closed:
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            self._guarded(self._local_step)()

    def _local_step(self) -> None:
        while True:
            with self.lock:
                if self.reanchor is not None:
                    self._apply_reanchor()
                if self.tracker is None or self.local_next > self.latest:
                    break
                idx = self.local_next
                jpeg = self.frames.get(idx)
                self.local_next += 1
            if jpeg is None:
                continue
            m = self.tracker.update(to_gray(jpeg))
            if m is not None:
                with self.lock:
                    self.local[idx] = (m.x, m.y)
                self._pending_local.append([idx, round(m.x, 2), round(m.y, 2)])
                self._since_refit += 1
            if len(self._pending_local) >= LOCAL_BATCH:
                self.send({"type": "track", "source": "local", "points": self._pending_local,
                           "object_size_px": self.object_size})
                self._pending_local = []
            self._schedule_chunks()
            if self._since_refit >= REFIT_EVERY:
                self._since_refit = 0
                self._maybe_refit()

    def _apply_reanchor(self) -> None:
        frame, box = self.reanchor  # type: ignore[misc]
        self.reanchor = None
        jpeg = self.frames.get(frame)
        if jpeg is None or self.tracker is None:
            return
        self.tracker.reset(to_gray(jpeg), box)
        # Replay everything after the anchor; those local positions are recomputed.
        for k in [k for k in self.local if k > frame]:
            del self.local[k]
        self.local_next = frame + 1
        self.send({"type": "notice", "level": "info", "message": "Re-anchored the tracker to SAM."})

    # ---------- SAM refinement (parallel chunks) ----------

    def _schedule_chunks(self) -> None:
        while self.paused_reason is None and (self.scene is not None or self.selected_by_tap):
            with self.lock:
                start = self.next_chunk
                indices = list(range(start, start + self.chunk_span, self.cfg.sam_stride))
                if indices[-1] > self.latest or start not in self.local:
                    return
                jpegs = [self.frames[i] for i in indices if i in self.frames]
                self.next_chunk += self.chunk_span
                if len(jpegs) != len(indices):
                    continue  # fell out of the buffer; skip rather than send a chunk with a gap
                cx, cy = self.local[start]
            half = (self.object_size or 20) / 2
            box = (cx - half, cy - half, cx + half, cy + half)
            job = self._guarded(lambda i=indices, j=jpegs, b=box: self._refine(i, j, b))
            if self.sync:
                job()
            else:
                assert self._pool is not None
                self._pool.submit(job)

    def _refine(self, indices: list[int], jpegs: list[bytes], box: tuple[float, ...]) -> None:
        if self.paused_reason is not None:
            return
        path = self._encode(indices[0], jpegs)
        first = self.calibration is None and self.scene is not None
        text = self.sam_fn(path, self._prompt(box, first_chunk=first), indices)
        tracks = tracks_from_text(text) if text.strip() else {}
        if first:
            self.calibration = self._calibrate(tracks.get(REFERENCE_ID))
        moving = tracks.get(MOVING_ID)
        if moving is None:
            self.send({"type": "notice", "level": "info", "message": "SAM lost the object in one chunk; local tracking continues."})
            return
        keep = moving.frames < len(indices)  # chunk-local frames only
        points = []
        with self.lock:
            for k, (x, y) in zip(moving.frames[keep], moving.centers_px[keep], strict=True):
                g = indices[int(k)]
                self.sam[g] = (float(x), float(y))
                points.append([g, round(float(x), 2), round(float(y), 2)])
            if keep.any():
                self.object_size = float(np.median(moving.long_px[keep]))
            self._check_drift(points)
        self.send({"type": "track", "source": "sam", "points": points, "object_size_px": self.object_size})

    def _check_drift(self, points: list[list[float]]) -> None:
        if not points or self.object_size is None:
            return
        g, x, y = points[-1]
        local = self.local.get(int(g))
        if local is not None and np.hypot(local[0] - x, local[1] - y) > DRIFT_FRACTION * self.object_size:
            half = self.object_size / 2
            self.reanchor = (int(g), (int(x - half), int(y - half), int(x + half) + 1, int(y + half) + 1))

    def _prompt(self, box: tuple[float, ...], first_chunk: bool) -> str:
        assert self.size is not None
        w, h = self.size
        b = msp.PromptBox(max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2]) + 1), min(h, int(box[3]) + 1))
        objects = [msp.PromptObject(id=int(MOVING_ID), prompts=[msp.FramePrompt(frame=0, box=b)])]
        if first_chunk and self.scene is not None and self.scene.reference_box is not None:
            rx1, ry1, rx2, ry2 = self.scene.reference_box.to_pixels(w, h)
            objects.append(msp.PromptObject(id=int(REFERENCE_ID), prompts=[
                msp.FramePrompt(frame=0, box=msp.PromptBox(rx1, ry1, max(rx2, rx1 + 1), max(ry2, ry1 + 1)))]))
        return msp.build_object_prompt(size=self.size, objects=objects)

    def _encode(self, start: int, jpegs: list[bytes]) -> Path:
        assert self.size is not None
        out = self.run_dir / f"chunk_{start:06d}.mp4"
        w, h = self.size
        rate = self.cfg.fps / self.cfg.sam_stride
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", f"{rate:g}", "-i", "-",
             "-vf", f"crop={w}:{h}:0:0", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(out)],
            input=b"".join(jpegs), check=True, capture_output=True,
        )
        return out

    def _sam_via_client(self, path: Path, prompt: str, indices: list[int]) -> str:
        common = {"prompt": prompt, "video_hash": file_digest(path), "est_frames": len(indices), "metadata": SAM_METADATA,
                  "run_id": self.run_id, "audience": self.cfg.audience, "run_cap_usd": self.cfg.run_cap_usd}
        if self.use_data_url:
            # Inline base64 saves an upload and a delete round trip per chunk (seconds from India).
            url = "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode()
            try:
                return self.client.sam_video(video_url=url, **common).text
            except ProviderCallFailed as e:
                if e.status_code != 400:
                    raise
                log.warning("SAM rejected an inline video (%s); uploading chunks instead.", e)
                self.use_data_url = False
        return self.client.sam_video(video_path=path, **common).text

    def _calibrate(self, reference: ObjectTrack | None) -> Calibration | None:
        if reference is not None and self.scene is not None and self.scene.reference_kind != "none":
            try:
                return from_reference(reference, self.scene.reference_kind)
            except CalibrationError as e:
                self.send({"type": "notice", "level": "warning", "message": str(e)})
        return None

    # ---------- fitting ----------

    def _merged(self) -> tuple[np.ndarray, np.ndarray]:
        """SAM where it has answered, the local tracker everywhere else."""
        with self.lock:
            merged = {**self.local, **self.sam}
        frames = np.array(sorted(merged), dtype=int)
        return frames, np.array([merged[f] for f in frames]) if len(frames) else np.empty((0, 2))

    def _maybe_refit(self) -> None:
        if self.sync:
            self._guarded(self.refit)()
        elif self._refit_busy.acquire(blocking=False):
            def run() -> None:
                try:
                    self._guarded(self.refit)()
                finally:
                    self._refit_busy.release()
            threading.Thread(target=run, daemon=True).start()

    def refit(self) -> dict[str, Any] | None:
        fps = self.cfg.fps
        frames, centers = self._merged()
        if len(frames) < MIN_FIT_S * fps:
            return None
        recent = (frames >= frames[-1] - WINDOW_S * fps) & (frames >= self.regime_start)
        seg_frames, seg_centers = max(fill_short_gaps(frames[recent], centers[recent], max_gap=3), key=lambda s: len(s[0]))
        if len(seg_frames) < MIN_FIT_S * fps:
            return None
        ppm = self._scale(seg_centers)
        if ppm is None:
            self.send({"type": "notice", "level": "warning",
                       "message": "No scale yet: show an A4 sheet in the swing plane, or enter the pendulum length."})
            return None
        traj = from_pixels(seg_frames, seg_centers, fps, ppm)
        if self.family is None:
            if self.scene is None:
                return None  # tracking started from a tap; the scene model hasn't answered yet
            self._choose_family(traj)
        try:
            coords = to_coordinates(traj, self.family)  # type: ignore[arg-type]
        except ValueError as e:
            msg = {"type": "fit", "status": "no_fit", "reason": str(e)}
            self.send(msg)
            return msg
        fits = [fit_model(MODELS[m], coords, warm=self.warm.get(m)) for m in self.candidates]
        for f in fits:
            if f.success:
                self.warm[f.model_id] = f.params
        verdict = assess(coords, fits)
        change = detect_change(traj, self.family, max(self.candidates, key=lambda m: len(MODELS[m].params)))  # type: ignore[arg-type]
        if change is not None:
            self._new_regime(int(seg_frames[change.index]), change.t_estimate + float(seg_frames[0]) / fps, change.error_ratio)
            return None
        lead: FitResult | None = verdict.winner or (verdict.contenders[0] if verdict.contenders else None)
        with self.lock:
            sam_share = sum(1 for f in seg_frames if int(f) in self.sam) / len(seg_frames)
        msg: dict[str, Any] = {"type": "fit", "status": verdict.status, "reason": verdict.reason, "samples": len(traj),
                               "window_s": round(float(traj.t[-1]), 2), "sam_share": round(sam_share, 2)}
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
            msg["ghost"] = self._prediction(spec, lead, coords, seg_frames, seg_centers[0], ppm)
        self.send(msg)
        return msg

    def _new_regime(self, start_frame: int, at_s: float, ratio: float) -> None:
        """The fitted model stopped predicting the motion: fit only what comes after, from scratch."""
        with self.lock:
            self.regime_start = start_frame
            self.warm.clear()
            self.history.clear()
        if time.monotonic() - self._last_llm >= LLM_MIN_INTERVAL_S:
            self.family = None  # one fresh hypotheses call once the new regime has enough data
        self.send({"type": "regime_change", "frame": start_frame, "at_s": round(at_s, 2), "error_ratio": round(ratio, 1),
                   "message": f"Dynamics changed at about {at_s:.1f} s (prediction error {ratio:.0f}x normal). Refitting on the new motion."})

    def _scale(self, centers: np.ndarray) -> float | None:
        if self.calibration is not None:
            return self.calibration.px_per_m
        if self.cfg.pendulum_length_m:
            radius_px = fit_circle(centers * np.array([1.0, -1.0]))[2]
            cal = from_known_length(radius_px, self.cfg.pendulum_length_m)
            if len(centers) >= 2 * MIN_FIT_S * self.cfg.fps:
                self.calibration = cal  # frozen once the arc is measured over a long stretch
            return cal.px_per_m
        return None

    def _choose_family(self, traj) -> None:
        """One LLM call per session; afterwards refits are local math only."""
        assert self.scene is not None
        ids: list[str] = []
        self._last_llm = time.monotonic()
        try:
            hyp, call = propose_hypotheses(self.client, summarize(traj), self.scene, self.run_id, self.cfg.audience,
                                           run_cap_usd=self.cfg.run_cap_usd)
            self.family = hyp.family
            ids = [h.model_id for h in hyp.candidates]
            if self.family in ("double_pendulum", "rotation"):
                # Live mode follows one object's position; both bobs, or a spin angle, need recorded mode.
                what = "Double pendulum" if self.family == "double_pendulum" else "Spinning object"
                self.send({"type": "notice", "level": "info",
                           "message": f"{what} detected: record a clip for the full analysis (live mode follows one position)."})
                self.family, ids = ("pendulum" if self.family == "double_pendulum" else self.scene.motion_family
                                    if self.scene.motion_family in ("pendulum", "oscillator", "projectile") else "pendulum"), []
            self.send({"type": "hypotheses", "family": hyp.family, "model": call.model, "provider": call.provider,
                       "items": [{"model_id": h.model_id, "title": MODELS[h.model_id].title, "why": h.rationale}
                                 for h in hyp.candidates]})
        except (CostGuardError, LLMError) as e:
            # No usable hypotheses (budget, or invalid output twice): test the whole library of the scene's family.
            self.family = self.scene.motion_family if self.scene.motion_family != "other" else "pendulum"
            self.send({"type": "notice", "level": "warning", "message": f"Hypotheses unavailable ({e}); testing every {self.family} model."})
        simplest = min(models_for(self.family), key=lambda m: len(m.params)).id
        self.candidates = list(dict.fromkeys([simplest, *(ids or [m.id for m in models_for(self.family)])]))

    def _prediction(self, spec, lead: FitResult, coords, frames: np.ndarray, origin: np.ndarray, ppm: float) -> list[list[float]]:
        fps = self.cfg.fps
        t = np.arange(len(frames) + int(PREDICT_S * fps)) / fps
        q = simulate(spec, lead.params, lead.y0, t, coords.ctx)
        if q is None:
            return []
        xy = coords.to_xy(q)
        f0 = int(frames[0])
        px = origin[0] + xy[:, 0] * ppm
        py = origin[1] - xy[:, 1] * ppm
        return [[f0 + i, round(float(x), 2), round(float(y), 2)] for i, (x, y) in enumerate(zip(px, py, strict=True))]


def _downscale(jpeg: bytes, long_side: int) -> bytes:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    s = min(1.0, long_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else jpeg
