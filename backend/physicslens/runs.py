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
from .pipeline import AnalysisError, RunOptions, analyze
from .settings import Settings

log = logging.getLogger("physicslens.runs")


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
        self._runs: dict[str, RunState] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers)

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

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
            return self._runs.get(run_id)

    def _work(self, state: RunState, video: Path, options: RunOptions) -> None:
        state.status = "running"
        try:
            state.result = analyze(self.client, video, video.parent, state.run_id, options, progress=state.progress.append)
            state.status = "done"
            (video.parent / "result.json").write_text(json.dumps(state.result, default=str))
        except (AnalysisError, LLMError) as e:
            state.status, state.error = "error", str(e)
        except CostGuardError as e:
            # Judges and public visitors never see budget wording.
            state.status = "error"
            state.error = str(e) if options.audience == "dev" else BUSY_MESSAGE
            log.error("Run %s stopped by costguard: %s", state.run_id, e)
        except Exception:
            log.exception("Run %s crashed", state.run_id)
            state.status, state.error = "error", "Something went wrong while analyzing this clip. Please try another recording."

    def public_view(self, state: RunState) -> dict[str, Any]:
        return {k: v for k, v in asdict(state).items() if k != "started"}
