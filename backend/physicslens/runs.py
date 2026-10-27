"""Background analysis runs: one worker thread per run, state kept in memory and results written to var/runs/."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .agent.llm import LLMError
from .costguard import CostGuardClient, CostGuardError
from .costguard.guard import BUSY_MESSAGE
from .perception.media import MediaError
from .pipeline import AnalysisError, RunOptions, analyze
from .settings import Settings

log = logging.getLogger("physicslens.runs")

PUBLIC_PAUSED_MESSAGE = "Free analyses are paused for today. Browse the gallery above, or come back tomorrow."


@dataclass
class RunState:
    run_id: str
    status: str = "queued"  # queued | running | done | error
    progress: list[str] = field(default_factory=list)
    error: str | None = None
    result: dict[str, Any] | None = None
    started: float = field(default_factory=time.time)


class RunManager:
    def __init__(self, settings: Settings, client: CostGuardClient, workers: int = 2):
        self.settings = settings
        self.client = client
        self.root = settings.data_dir / "runs"
        self.gallery_root = settings.repo_root / "gallery"
        self._runs: dict[str, RunState] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers)

    def run_dir(self, run_id: str) -> Path:
        # Curated gallery entries (tracked in the repo) are looked up first; they are read-only showcases.
        gallery = self.gallery_root / run_id
        return gallery if gallery.is_dir() else self.root / run_id

    def start(self, video: Path, options: RunOptions) -> RunState:
        state = RunState(run_id=video.parent.name)
        with self._lock:
            self._runs[state.run_id] = state
        self._pool.submit(self._work, state, video, options)
        return state

    def new_run_dir(self) -> Path:
        d = self.root / uuid.uuid4().hex[:12]
        d.mkdir(parents=True)
        return d

    def get(self, run_id: str) -> RunState | None:
        with self._lock:
            state = self._runs.get(run_id)
        if state is None and run_id.replace("-", "").isalnum():
            # Finished runs survive a server restart (and can be shared by link).
            saved = self.run_dir(run_id) / "result.json"
            if saved.is_file():
                state = RunState(run_id=run_id, status="done", progress=["Done"], result=json.loads(saved.read_text()))
        return state

    def _work(self, state: RunState, video: Path, options: RunOptions) -> None:
        state.status = "running"
        try:
            state.result = analyze(self.client, video, video.parent, state.run_id, options, progress=state.progress.append)
            state.status = "done"
            (video.parent / "result.json").write_text(json.dumps(state.result, default=str))
        except (AnalysisError, LLMError) as e:
            state.status, state.error = "error", str(e)
        except MediaError:
            state.status, state.error = "error", "Couldn't read that video. Please upload an MP4 or MOV file."
        except CostGuardError as e:
            # Judges and public visitors never see budget wording; public visitors are pointed to the gallery.
            state.status = "error"
            state.error = {"dev": str(e), "judge": BUSY_MESSAGE}.get(options.audience, PUBLIC_PAUSED_MESSAGE)
            log.error("Run %s stopped by costguard: %s", state.run_id, e)
        except Exception:
            log.exception("Run %s crashed", state.run_id)
            state.status, state.error = "error", "Something went wrong while analyzing this clip. Please try another recording."

    def public_view(self, state: RunState) -> dict[str, Any]:
        return {k: v for k, v in asdict(state).items() if k != "started"}


    def gallery(self) -> list[dict[str, Any]]:
        """Curated, precomputed results anyone can open for free."""
        items = []
        for d in sorted(self.gallery_root.glob("*/meta.json")):
            meta = json.loads(d.read_text())
            result_file = d.parent / "result.json"
            result = json.loads(result_file.read_text()) if result_file.is_file() else {}
            eq = result.get("equation") or {}
            items.append({"id": d.parent.name, **meta, "status": result.get("status"), "family": result.get("family"),
                          "equation_title": eq.get("title"), "object": (result.get("scene") or {}).get("moving_object")})
        return sorted(items, key=lambda i: i.get("order", 99))
