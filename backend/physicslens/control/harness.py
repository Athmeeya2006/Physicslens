"""Runs a controller in a separate, resource-limited process and returns its metrics.

Generated code is never imported into the server. The child is `python -I` (no project or user paths, no
PYTHON* variables) with an environment holding only thread-count settings, in an empty temporary directory, with
RLIMIT_CPU, RLIMIT_AS, RLIMIT_FSIZE and RLIMIT_NPROC set by the child before the code is compiled, and a
wall-clock timeout here that kills its whole process group.
"""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("physicslens.control")

WORKER = Path(__file__).with_name("worker.py")
# One BLAS thread: no thread pools to account for, and RLIMIT_NPROC can then forbid new ones.
CHILD_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "LANG": "C.UTF-8"}
STDERR_TAIL = 400


@dataclass(frozen=True)
class SandboxLimits:
    cpu_seconds: int = 25
    wall_seconds: float = 30.0
    memory_mb: int = 1536

    @classmethod
    def from_config(cls, control_cfg: dict[str, Any]) -> SandboxLimits:
        return cls(int(control_cfg.get("sandbox_cpu_seconds", 25)), float(control_cfg.get("sandbox_wall_seconds", 30)),
                   int(control_cfg.get("sandbox_memory_mb", 1536)))


def _stopped(n: int, reason: str) -> list[dict[str, Any]]:
    return [{"success": False, "settle_s": None, "overshoot": None, "final_error": None, "peak_u": None,
             "rms_u": None, "saturation": None, "nan": False, "error": reason} for _ in range(n)]


def _signal_reason(code: int) -> str:
    if code == -signal.SIGXCPU or code == -signal.SIGKILL:
        return "stopped: the CPU time limit was reached (an endless loop or far too much work per step)"
    if code < 0:
        return f"stopped by signal {signal.Signals(-code).name}"
    return f"the evaluation process exited with code {code}"


def run_controller(code: str, job: dict[str, Any], limits: SandboxLimits) -> list[dict[str, Any]]:
    """Per-sample metrics, in the order of job["samples"]. Never raises for anything the controller does."""
    n = len(job["samples"])
    payload = json.dumps({**job, "code": code})
    with tempfile.TemporaryDirectory(prefix="physicslens-ctl-") as cwd:
        try:
            proc = subprocess.Popen(
                [sys.executable, "-I", str(WORKER), str(limits.cpu_seconds), str(limits.memory_mb)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=dict(CHILD_ENV),
                start_new_session=True, close_fds=True,
            )
        except OSError as e:
            log.error("Could not start the controller sandbox: %s", e)
            return _stopped(n, "the evaluation could not start")
        try:
            out, err = proc.communicate(payload.encode(), timeout=limits.wall_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                log.info("controller sandbox ended just as its time limit was reached")
            proc.communicate()
            return _stopped(n, f"stopped: the {limits.wall_seconds:.0f} s time limit for the whole evaluation was reached")
    if proc.returncode != 0:
        tail = err.decode(errors="replace").strip().splitlines()[-1:] if err else []
        if tail and "MemoryError" in tail[0]:
            return _stopped(n, "stopped: the memory limit was reached")
        log.info("controller sandbox exited %s: %s", proc.returncode, " ".join(tail)[:STDERR_TAIL])
        return _stopped(n, _signal_reason(proc.returncode))
    try:
        samples = json.loads(out.decode().strip().splitlines()[-1])["samples"]
    except (ValueError, IndexError, KeyError) as e:
        log.warning("controller sandbox returned no readable metrics: %s", e)
        return _stopped(n, "the evaluation returned no readable result")
    if len(samples) != n:
        return _stopped(n, "the evaluation returned the wrong number of results")
    return samples


def summarize(samples: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Accepted only when at least `threshold` of the set succeeds AND the nominal model (sample 0) does."""
    ok = sum(bool(s.get("success")) for s in samples)
    n = len(samples)

    def mean(key: str) -> float | None:
        vals = [s[key] for s in samples if isinstance(s.get(key), int | float) and math.isfinite(s[key])]
        return float(sum(vals) / len(vals)) if vals else None

    settled = [s["settle_s"] for s in samples if s.get("success")]
    return {
        "successes": ok, "samples": n, "success_rate": ok / n if n else 0.0,
        "nominal_success": bool(samples and samples[0].get("success")),
        "passed": bool(n) and ok / n >= threshold and bool(samples[0].get("success")),
        "mean_settle_s": float(sum(settled) / len(settled)) if settled else None,
        "mean_final_error": mean("final_error"), "mean_peak_u": mean("peak_u"), "mean_rms_u": mean("rms_u"),
        "mean_saturation": mean("saturation"), "errors": sum(1 for s in samples if s.get("error")),
    }
