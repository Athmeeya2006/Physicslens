"""Live mode: instant local tracking for the overlay, SAM 3.1 refining it in parallel chunks, rolling refits.

Measured on the first real session: SAM answers a 2 s chunk about 12 to 16 s later (~5 frames/s of
throughput), so it cannot drive a live overlay on its own. A template tracker follows the object on every
frame immediately; SAM runs on every second frame, several chunks at once, and its positions replace the
local ones as they arrive. If SAM shows the local tracker drifted, the tracker is re-anchored.
"""

from __future__ import annotations

import base64
import json
import logging
import struct
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
from .physics.coords import Coordinates, fit_circle, to_coordinates
from .physics.cueing import learned_lead, next_cue, push_delay
from .physics.fit import FitResult, fit_model, simulate
from .physics.forecast import FAMILIES as FORECAST_FAMILIES
from .physics.forecast import Prediction, judge, observed_crossing, predict_crossing
from .physics.models import MODELS, models_for
from .physics.regime import detect_change
from .physics.uncertainty import bootstrap_samples
from .physics.verify import assess
from .pipeline import (
    MOVING_ID,
    NOMINAL_PX_PER_M,
    REFERENCE_ID,
    SAM_METADATA,
    SCALE_NOTE,
    _candidates,
    _hypothesis_items,
    _with_control,
)
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
BUFFER_S = 25.0  # frames kept for chunk encoding and tracker re-anchoring (fit window plus SAM's lag)
DRIFT_FRACTION = 0.5  # re-anchor when local and SAM disagree by more than this share of the object size
SAM_WORKERS = 4
LLM_MIN_INTERVAL_S = 5.0  # live mode never calls the language model more often than this
# Predict-the-future: try this far ahead first (shorter if the swing will have died out), with a small bootstrap
# for the timing window, once the law is accepted and its parameters have settled.
FORECAST_HORIZONS_S = (20.0, 10.0, 5.0)
FORECAST_REPS = 12
FORECAST_MIN_HISTORY = 3

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
        # Capture time of every frame, from the phone's clock: phones rarely hold the nominal rate, and a fit on
        # arrival order would compress time (15 fps taken as 25 inflates g almost threefold).
        self.times: dict[int, float] = {}
        self.uncalibrated_noted = False
        self.t0: float | None = None  # capture time of the first frame: session times are relative to it
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
        self.family_from_user = False  # tapped "what is it?": fit now, let the agent's hypotheses refine it later
        self._agent_consulted = False
        self.candidates: list[str] = []
        self.warm: dict[str, dict[str, float]] = {}
        self.history: list[dict[str, float]] = []
        self.paused_reason: str | None = None
        self.regime_start = 0  # frames before this belong to an earlier regime and are not fitted
        # Each prediction is stored before its event happens, with wall-clock time, so "predicted before it
        # happened" can be audited from predictions.json.
        self.forecast: dict[str, Any] | None = None
        self.forecasts: list[dict[str, Any]] = []
        self._forecast_busy = False
        # Control by hand: the user pushes, the law says when. Fits use only motion since the last push (a push is
        # outside the law), and each push's real time, found on the track, teaches the app this person's lag.
        self.control: dict[str, Any] | None = None
        self.push_delays: list[float] = []
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

    def add_frame(self, data: bytes) -> int:
        """Frames are numbered in arrival order; the phone counts the same way, so indices line up."""
        jpeg, captured = split_frame(data)
        if self.size is None:
            gray = to_gray(jpeg)
            self.size = (gray.shape[1] - gray.shape[1] % 2, gray.shape[0] - gray.shape[0] % 2)
        with self.lock:
            self.latest += 1
            idx = self.latest
            self.frames[idx] = jpeg
            self.times[idx] = captured if captured is not None else idx / self.cfg.fps
            if self.t0 is None:
                self.t0 = self.times[idx]
            for old in [k for k in self.frames if k < idx - BUFFER_S * self.cfg.fps]:
                del self.frames[old]
                self.times.pop(old, None)
        if not self._scene_started and idx + 1 >= SCENE_AFTER_FRAMES:
            self._scene_started = True
            self._run(self._understand)
        if self.sync:
            self._local_step()
        else:
            self._wake.set()
        return idx

    def choose_family_now(self, family: str) -> None:
        """The user says what it is, so fitting starts at once instead of after the reasoning model's first answers."""
        if family not in USER_FAMILIES or self.family is not None:
            return
        with self.lock:
            self.family = family
            self.family_from_user = True
            self.candidates = _with_control(family, [m.id for m in models_for(family)])
        self.send({"type": "notice", "level": "info",
                   "message": f"Fitting {USER_FAMILIES[family]} now; the agent's own hypotheses will refine them when ready."})

    def _consult_agent(self, traj) -> None:
        """After a tapped family: the agent still proposes; its candidates join if it agrees, and it says so if not."""
        assert self.scene is not None
        hyp, call = propose_hypotheses(self.client, summarize(traj, self.calibration is None and not self.cfg.pendulum_length_m),
                                       self.scene, self.run_id, self.cfg.audience, run_cap_usd=self.cfg.run_cap_usd)
        self.send({"type": "hypotheses", "family": hyp.family, "model": call.model, "provider": call.provider,
                   "items": _hypothesis_items(hyp)})
        if hyp.family == self.family:
            ids, _ = _candidates(hyp)
            with self.lock:
                self.candidates = list(dict.fromkeys([*self.candidates, *ids]))
        else:
            self.send({"type": "notice", "level": "info",
                       "message": f"The agent reads this as {hyp.family} motion, not {self.family}. Restart and pick that if it agrees with what you see."})

    def set_control(self, goal: str, target_deg: float) -> None:
        if goal not in ("pump", "stop"):
            self.control = None
            self.send({"type": "cue", "action": "off"})
            return
        self.control = {"goal": goal, "target": float(np.radians(np.clip(target_deg, 3.0, 80.0))), "after_t": None,
                        "pending": None}
        self.send({"type": "notice", "level": "info", "message": (
            "Control on: listen for the ticks and push on the beep." if self.family == "pendulum"
            else "Control works on a swinging pendulum; it starts once the motion is identified as one.")})

    def set_length(self, meters: float) -> None:
        """A length said out loud mid-session. It replaces an earlier given length, never a reference object's scale."""
        with self.lock:
            self.cfg.pendulum_length_m = meters
            if self.calibration is not None and self.calibration.source == "user":
                self.calibration = None
        self.send({"type": "notice", "level": "info", "message": f"Length set to {meters * 100:.0f} cm."})

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
        try:
            text = self.sam_fn(path, self._prompt(box, first_chunk=first), indices)
        finally:
            path.unlink(missing_ok=True)  # chunks are only an upload format; the cache keys on their hash
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
        if self.control is not None and self.control["after_t"] is not None:
            with self.lock:
                recent &= np.array([self.times.get(int(f), int(f) / fps) > self.control["after_t"] for f in frames])
        seg_frames, seg_centers = max(fill_short_gaps(frames[recent], centers[recent], max_gap=3), key=lambda s: len(s[0]))
        if len(seg_frames) < MIN_FIT_S * fps:
            return None
        ppm = self._scale(seg_centers)
        uncalibrated = ppm is None
        if uncalibrated:
            # Like recorded mode: keep going on a nominal scale; frequency and damping don't need one.
            ppm = NOMINAL_PX_PER_M
            if not self.uncalibrated_noted:
                self.uncalibrated_noted = True
                self.send({"type": "notice", "level": "info", "message": SCALE_NOTE})
        with self.lock:
            ts = np.array([self.times.get(int(f), int(f) / fps) for f in seg_frames])
        ts = ts + np.arange(len(ts)) * 1e-6  # identical stamps (a clock tick) must still be strictly increasing
        ts = np.maximum.accumulate(ts)
        if ts[-1] - ts[0] <= 0:
            return None
        # Resample onto an even grid at the rate the phone really managed: the fitter assumes even sampling.
        rate = (len(ts) - 1) / (ts[-1] - ts[0])
        grid = ts[0] + np.arange(len(ts)) / rate
        even = np.column_stack([np.interp(grid, ts, seg_centers[:, 0]), np.interp(grid, ts, seg_centers[:, 1])])
        traj = from_pixels(np.arange(len(grid)), even, rate, ppm)
        if self.family is None:
            if self.scene is None:
                return None  # tracking started from a tap; the scene model hasn't answered yet
            self._choose_family(traj)
        elif self.family_from_user and self.scene is not None and not self._agent_consulted:
            self._agent_consulted = True
            self._run(lambda t=traj: self._consult_agent(t))  # in the background: refits keep running meanwhile
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
            at = grid[min(change.index, len(grid) - 1)]
            frame = int(seg_frames[min(np.searchsorted(ts, at), len(seg_frames) - 1)])
            self._new_regime(frame, change.t_estimate + float(ts[0] - (self.t0 or 0.0)), change.error_ratio)
            return None
        lead: FitResult | None = verdict.winner or (verdict.contenders[0] if verdict.contenders else None)
        with self.lock:
            sam_share = sum(1 for f in seg_frames if int(f) in self.sam) / len(seg_frames)
        msg: dict[str, Any] = {"type": "fit", "status": verdict.status, "reason": verdict.reason, "samples": len(traj),
                               "window_s": round(float(traj.t[-1]), 2), "sam_share": round(sam_share, 2),
                               "measured_fps": round(rate, 1), "uncalibrated": uncalibrated}
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
            msg["ghost"] = self._prediction(spec, lead, coords, seg_frames, ts, rate, seg_centers[0], ppm)
            self._forecast_step(spec, lead, coords, float(ts[0]), verdict.status == "accepted")
            if self.control is not None and self.family == "pendulum" and verdict.status == "accepted":
                self._control_step(spec, lead, coords, float(ts[0]))
        self.send(msg)
        return msg

    # ---------- predict the future, then prove it ----------

    def _forecast_step(self, spec, lead: FitResult, coords, t_abs0: float, accepted: bool) -> None:
        """Judge a pending prediction once its window has passed; start a new one when none is pending."""
        f = self.forecast
        if f is not None:
            p = Prediction(f["t_abs"] - t_abs0, f["window_s"], float(np.median(coords.q[:, 0])), f["direction"], 0.0)
            outcome = judge(p, observed_crossing(coords.t, coords.q[:, 0], p))
            if outcome["status"] != "unmeasured":
                f.update(outcome, judged_wall=time.time(), observed_s=round(outcome["observed"] + t_abs0 - (self.t0 or 0.0), 3))
                self.forecast = None
                self._save_forecasts()
                self.send({"type": "prediction", **_public(f)})
            elif coords.t[-1] + t_abs0 > f["t_abs"] + f["window_s"] + 2.0:
                f.update(status="unmeasured", judged_wall=time.time())  # tracking lost it; say so rather than guess
                self.forecast = None
                self._save_forecasts()
                self.send({"type": "prediction", **_public(f)})
            return
        # While the user is pushing the swing, it is not a free system: no predictions about it.
        if (not accepted or self._forecast_busy or self.control is not None or self.family not in FORECAST_FAMILIES
                or len(self.history) < FORECAST_MIN_HISTORY):
            return
        self._forecast_busy = True
        self._run(lambda: self._make_forecast(spec, lead, coords, t_abs0))

    def _make_forecast(self, spec, lead: FitResult, coords, t_abs0: float) -> None:
        try:
            samples, _ = bootstrap_samples(spec, coords, lead, reps=FORECAST_REPS)
            for horizon in FORECAST_HORIZONS_S:
                p = predict_crossing(spec, lead, coords, horizon, samples)
                if p is not None:
                    t_rel = (self.t0 or 0.0)
                    self.forecast = {"id": len(self.forecasts) + 1, "status": "pending", "model_id": spec.id,
                                     "made_wall": time.time(), "made_s": round(p.made_at + t_abs0 - t_rel, 3),
                                     "target_s": round(p.t_pred + t_abs0 - t_rel, 3), "window_s": round(p.window_s, 3),
                                     "horizon_s": horizon, "direction": p.direction, "t_abs": p.t_pred + t_abs0}
                    self.forecasts.append(self.forecast)
                    self._save_forecasts()
                    self.send({"type": "prediction", **_public(self.forecast)})
                    return
        finally:
            self._forecast_busy = False

    def _control_step(self, spec, lead: FitResult, coords: Coordinates, t_abs0: float) -> None:
        ctl = self.control
        assert ctl is not None
        now = float(coords.t[-1]) + t_abs0
        pend = ctl["pending"]
        if pend is not None:
            if now < pend["pushes_abs"][-1] + 0.8:
                return  # a push train is still under way: don't reschedule over it
            # Find when the first push really landed, on the law that was fitted before it.
            shift = t_abs0 - pend["t_abs0"]
            d = push_delay(pend["spec"], pend["fit"], pend["coords"], coords.t + shift, coords.q[:, 0],
                           pend["pushes_abs"][0] - pend["t_abs0"])
            if d is not None:
                self.push_delays.append(d + pend["lead"])
            ctl["after_t"] = pend["pushes_abs"][-1] + 0.1
            ctl["pending"] = None
            return  # refit on the motion after the pushes first
        lead_s = learned_lead(self.push_delays)
        cue = next_cue(spec, lead, coords, ctl["goal"], ctl["target"], lead_s=lead_s)
        if cue is None:
            return
        t0 = self.t0 or 0.0
        msg = {"type": "cue", "action": cue.action, "goal": ctl["goal"], "period_s": round(cue.period_s, 3),
               "direction": "right" if cue.direction > 0 else "left", "amplitude_deg": round(float(np.degrees(cue.amplitude)), 1),
               "target_deg": round(float(np.degrees(ctl["target"])), 1), "lead_s": round(lead_s, 3),
               "bottom_speed": round(cue.bottom_speed, 3), "reason": cue.reason}
        if cue.action == "push":
            pushes_abs = [p + t_abs0 for p in cue.pushes]
            ctl["pending"] = {"pushes_abs": pushes_abs, "t_abs0": t_abs0, "spec": spec, "fit": lead, "coords": coords,
                              "lead": lead_s}
            msg["pushes_s"] = [round(p - t0, 3) for p in pushes_abs]
        self.send(msg)

    def _save_forecasts(self) -> None:
        (self.run_dir / "predictions.json").write_text(json.dumps([_public(f) for f in self.forecasts], indent=1))

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
            hyp, call = propose_hypotheses(self.client, summarize(traj, self.calibration is None and not self.cfg.pendulum_length_m), self.scene, self.run_id, self.cfg.audience,
                                           run_cap_usd=self.cfg.run_cap_usd)
            self.family = hyp.family
            ids, _ = _candidates(hyp)
            if self.family in ("double_pendulum", "rotation"):
                # Live mode follows one object's position; both bobs, or a spin angle, need recorded mode.
                what = "Double pendulum" if self.family == "double_pendulum" else "Spinning object"
                self.send({"type": "notice", "level": "info",
                           "message": f"{what} detected: record a clip for the full analysis (live mode follows one position)."})
                self.family, ids = ("pendulum" if self.family == "double_pendulum" else self.scene.motion_family
                                    if self.scene.motion_family in ("pendulum", "oscillator", "projectile") else "pendulum"), []
            self.send({"type": "hypotheses", "family": hyp.family, "model": call.model, "provider": call.provider,
                       "items": _hypothesis_items(hyp)})
        except (CostGuardError, LLMError) as e:
            # No usable hypotheses (budget, or invalid output twice): test the whole library of the scene's family.
            self.family = self.scene.motion_family if self.scene.motion_family != "other" else "pendulum"
            self.send({"type": "notice", "level": "warning", "message": f"Hypotheses unavailable ({e}); testing every {self.family} model."})
        self.candidates = _with_control(self.family, ids or [m.id for m in models_for(self.family)])

    def _prediction(self, spec, lead: FitResult, coords, frames: np.ndarray, ts: np.ndarray, rate: float,
                    origin: np.ndarray, ppm: float) -> list[list[float]]:
        """Ghost points for the frames shown so far (at their real capture times) and PREDICT_S ahead."""
        ahead = round(PREDICT_S * rate)
        rel = np.concatenate([ts - ts[0], (ts[-1] - ts[0]) + np.arange(1, ahead + 1) / rate])
        ids = np.concatenate([frames, frames[-1] + np.arange(1, ahead + 1)])
        q = simulate(spec, lead.params, lead.y0, rel, coords.ctx)
        if q is None:
            return []
        xy = coords.to_xy(q)
        px = origin[0] + xy[:, 0] * ppm
        py = origin[1] - xy[:, 1] * ppm
        return [[int(f), round(float(x), 2), round(float(y), 2)] for f, x, y in zip(ids, px, py, strict=True)]


USER_FAMILIES = {"pendulum": "pendulum models", "oscillator": "spring models", "projectile": "thrown-object models",
                 "line": "rolling and sliding models"}

FRAME_HEADER = b"PLT1"  # magic, then the capture time in seconds as little-endian float64, then the JPEG


def _public(f: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in f.items() if k != "t_abs"}


def split_frame(data: bytes) -> tuple[bytes, float | None]:
    """A bare JPEG (starts FF D8) from an older client has no timestamp; the nominal rate is assumed then."""
    if data[:4] == FRAME_HEADER and len(data) > 12:
        (t,) = struct.unpack("<d", data[4:12])
        return data[12:], float(t) if np.isfinite(t) else None
    return data, None


def _downscale(jpeg: bytes, long_side: int) -> bytes:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    s = min(1.0, long_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else jpeg
