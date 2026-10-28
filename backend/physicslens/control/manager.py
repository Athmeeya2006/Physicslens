"""Controller tasks in the background, like analyses: one worker, state in memory, results under the run's dir."""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agent.schemas import ControlTask
from ..costguard import CostGuardClient, CostGuardError
from ..costguard.guard import BUSY_MESSAGE
from .loop import LoopConfig, solve, task_key
from .system import ControlUnsupported, check_supported, robustness_params

log = logging.getLogger("physicslens.control")


@dataclass
class ControlState:
    task_id: str
    status: str = "queued"  # queued | running | done | error
    progress: list[str] = field(default_factory=list)
    error: str | None = None
    result: dict[str, Any] | None = None

    def view(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "status": self.status, "progress": self.progress, "error": self.error,
                "result": self.result}


class ControlManager:
    def __init__(self, client: CostGuardClient, root: Path, workers: int = 1):
        self.client = client
        self.root = root  # var/runs: never the tracked gallery directory
        self.cfg = client.limits.get("control") or {}
        self._tasks: dict[tuple[str, str], ControlState] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers)

    @property
    def reps(self) -> int:
        return int(self.cfg.get("bootstrap_samples", 50))

    @property
    def seed(self) -> int:
        return int(self.cfg.get("sample_seed", 7))

    def task_dir(self, run_id: str, task_id: str) -> Path:
        return self.root / run_id / "control" / task_id

    def start(self, run_id: str, result: dict[str, Any], task: ControlTask, audience: str) -> ControlState:
        check_supported(result, task)  # unsupported tasks fail fast, before anything is queued
        task_id = task_key(task, result["fit_state"], self.reps, self.seed)
        with self._lock:
            state = self._tasks.get((run_id, task_id))
            if state is not None and state.status != "error":
                return state
            saved = self._saved(run_id, task_id)
            if saved is not None:
                # Same task, same model, same prompt version: the answer is already known and costs nothing.
                state = ControlState(task_id, "done", ["Done"], result=saved)
                self._tasks[(run_id, task_id)] = state
                return state
            state = ControlState(task_id)
            self._tasks[(run_id, task_id)] = state
        self._pool.submit(self._work, state, run_id, result, task, audience)
        return state

    def get(self, run_id: str, task_id: str) -> ControlState | None:
        with self._lock:
            state = self._tasks.get((run_id, task_id))
        if state is None and task_id.isalnum():
            saved = self._saved(run_id, task_id)
            if saved is not None:
                state = ControlState(task_id, "done", ["Done"], result=saved)
        return state

    def _saved(self, run_id: str, task_id: str) -> dict[str, Any] | None:
        f = self.task_dir(run_id, task_id) / "metrics.json"
        return json.loads(f.read_text()) if f.is_file() else None

    def samples(self, run_id: str, result: dict[str, Any]) -> list[dict[str, float]]:
        """Nominal plus bootstrap parameter sets, computed once per run and stored with their seed."""
        f = self.root / run_id / "control" / f"samples-seed{self.seed}-n{self.reps}.json"
        if f.is_file():
            return json.loads(f.read_text())["params"]
        params = robustness_params(result, self.reps, self.seed)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"seed": self.seed, "bootstrap_samples": self.reps, "params": params}))
        return params

    def _work(self, state: ControlState, run_id: str, result: dict[str, Any], task: ControlTask, audience: str) -> None:
        state.status = "running"
        try:
            state.progress.append(f"Drawing {self.reps} variants of the identified model from its uncertainty")
            params = self.samples(run_id, result)
            calibrated = (result.get("calibration") or {}).get("source", "none") != "none"
            state.result = solve(self.client, result, task, params, self.task_dir(run_id, state.task_id),
                                 ledger_run_id=f"control-{run_id}-{state.task_id}", audience=audience,
                                 calibrated=calibrated, progress=state.progress.append,
                                 cfg=LoopConfig.from_client(self.client))
            state.status = "done"
        except ControlUnsupported as e:
            state.status, state.error = "error", str(e)
        except CostGuardError as e:
            # Judges never see budget wording; the reason is in the server log.
            state.status, state.error = "error", str(e) if audience == "dev" else BUSY_MESSAGE
            log.error("Control task %s stopped by costguard: %s", state.task_id, e)
        except Exception:
            log.exception("Control task %s crashed", state.task_id)
            state.status, state.error = "error", "Something went wrong while testing controllers for this system."
