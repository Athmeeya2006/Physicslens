"""Runs a controller in a separate, resource-limited process and returns its metrics.

Generated code is never imported into the server. The child is `python -I` (no project or user paths, no
PYTHON* variables) with an environment holding only thread-count settings, in an empty temporary directory, with
RLIMIT_CPU, RLIMIT_AS, RLIMIT_FSIZE and RLIMIT_NPROC set by the child before the code is compiled, and a
wall-clock timeout here that kills its whole process group.

When the machine allows it, the operating system adds a wall: bubblewrap runs the child in new namespaces (no
network, its own process tree) with only the system libraries and the Python install visible, read-only, and the
empty work directory writable; the project, its .env and the home directory do not exist in there. Failing that,
`unshare -rn` removes the network only. Which one is in use is probed once and reported by isolation().
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
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("physicslens.control")

WORKER = Path(__file__).with_name("worker.py")
# One BLAS thread: no thread pools to account for, and RLIMIT_NPROC can then forbid new ones.
CHILD_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "LANG": "C.UTF-8"}
STDERR_TAIL = 400
SANDBOX_WORKER = "/sandbox/worker.py"
SANDBOX_CWD = "/work"

_isolation: str | None = None
_isolation_lock = threading.Lock()


def _bwrap_prefix(cwd: str) -> list[str]:
    args = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL"]
    for d in ("/usr", "/lib", "/lib64", "/lib32", "/bin", "/sbin"):
        if os.path.islink(d):
            args += ["--symlink", os.readlink(d), d]
        elif os.path.isdir(d):
            args += ["--ro-bind", d, d]
    seen = {"/usr"}
    for d in (sys.base_prefix, sys.prefix, os.path.dirname(os.path.realpath(sys.executable))):
        if d and d not in seen and not any(d.startswith(s + "/") for s in seen):
            args += ["--ro-bind", d, d]
            seen.add(d)
    if os.path.exists("/etc/ld.so.cache"):
        args += ["--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"]
    return args + ["--ro-bind", str(WORKER), SANDBOX_WORKER, "--bind", cwd, SANDBOX_CWD, "--chdir", SANDBOX_CWD,
                   "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--"]


def _command(cwd: str, limits: SandboxLimits, mode: str) -> list[str]:
    tail = [str(limits.cpu_seconds), str(limits.memory_mb)]
    if mode == "bubblewrap":
        return [*_bwrap_prefix(cwd), sys.executable, "-I", SANDBOX_WORKER, *tail]
    prefix = ["unshare", "-rn", "--"] if mode == "unshare" else []
    return [*prefix, sys.executable, "-I", str(WORKER), *tail]


def _works(mode: str) -> bool:
    with tempfile.TemporaryDirectory(prefix="physicslens-probe-") as cwd:
        cmd = _command(cwd, SandboxLimits(), mode)
        # The worker reads a job from stdin; the probe only checks that Python, numpy and MuJoCo start inside.
        cmd = [*cmd[:cmd.index("-I") + 1], "-c", "import numpy, mujoco"]
        try:
            done = subprocess.run(cmd, check=False, cwd=cwd, env=dict(CHILD_ENV), capture_output=True, timeout=20)
            return done.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False


def isolation() -> str:
    """The OS-level isolation the sandbox uses here: "bubblewrap", "unshare" (network only) or "none"."""
    with _isolation_lock:
        return _isolation if _isolation is not None else _probe()


def _probe() -> str:
    global _isolation
    if _isolation is None:
        forced = os.environ.get("PHYSICSLENS_SANDBOX_ISOLATION")  # tests and operators can pin a mode
        candidates = [forced] if forced else ["bubblewrap", "unshare"]
        _isolation = next((m for m in candidates if m in ("bubblewrap", "unshare") and _works(m)), "none")
        if _isolation == "none":
            log.warning("Controller sandbox: no OS isolation available here (bubblewrap and unprivileged namespaces "
                        "are missing or blocked); generated code runs with resource limits and the static check only.")
        else:
            log.info("Controller sandbox isolation: %s", _isolation)
    return _isolation


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
    if code > 128 and (code - 128) in {sig.value for sig in signal.Signals}:
        code = 128 - code  # bubblewrap reports a child killed by a signal as 128 + signal, like a shell
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
                _command(cwd, limits, isolation()),
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
