"""Child process that runs one untrusted controller on a set of MuJoCo models and prints the metrics.

Started by harness.py as `python -I worker.py CPU_SECONDS MEMORY_MB` with an almost empty environment, in an
empty temporary directory. It must stay self-contained (standard library, numpy and mujoco only): isolated mode
puts neither this directory nor the project on sys.path, which is the point. Never import it from the server.

Input (stdin, JSON): code, params, dt, horizon_s, time_limit_s, u_max, tolerance, error, samples[{xml, q0, qdot0}].
Output: one JSON line on the original stdout, which the controller code cannot reach.
"""

import json
import math
import os
import resource
import sys


def _limit(cpu_s: int, memory_mb: int) -> None:
    # Applied before anything else is loaded, so the generated code never runs without them.
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
    resource.setrlimit(resource.RLIMIT_AS, (memory_mb << 20, memory_mb << 20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


_limit(int(sys.argv[1]), int(sys.argv[2]))

import io

import mujoco
import numpy as np

# No new processes or threads once numpy and MuJoCo are loaded.
resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))

ERROR_TEXT_MAX = 300
SAFE_BUILTINS = {name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name) for name in (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "len", "list", "map", "max", "min",
    "pow", "range", "reversed", "round", "sorted", "sum", "tuple", "zip", "isinstance", "divmod",
    "ValueError", "ZeroDivisionError", "ArithmeticError",
)}
_real_import = __import__


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0 or name not in ("math", "numpy"):
        raise ImportError(f"import of {name!r} is not allowed")
    return _real_import(name, globals, locals, fromlist, level)


SAFE_BUILTINS["__import__"] = _guarded_import


def load_controller(compiled):
    namespace = {"__builtins__": dict(SAFE_BUILTINS), "__name__": "controller"}
    exec(compiled, namespace)  # noqa: S102  (the reason this process exists; it runs only inside the sandbox)
    return namespace["control"]


def short(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:ERROR_TEXT_MAX]


def reference(spec: dict, t: float) -> float:
    if spec.get("sine"):
        return spec["offset"] + spec["amplitude"] * math.sin(2 * math.pi * spec["frequency_hz"] * t)
    return spec["offset"] + spec["target"]


def tracking_error(spec: dict, t: float, q: float, qdot: float) -> float:
    if spec["kind"] == "velocity":
        return qdot
    e = q - reference(spec, t)
    if spec["kind"] == "angle":
        e = (e + math.pi) % (2 * math.pi) - math.pi
    return e


def run_sample(compiled, job: dict, sample: dict) -> dict:
    out = {"success": False, "settle_s": None, "overshoot": None, "final_error": None, "peak_u": 0.0, "rms_u": 0.0,
           "saturation": 0.0, "nan": False, "error": None}
    try:
        control = load_controller(compiled)
    except Exception as e:  # noqa: BLE001  (anything the controller raises is reported, the sample fails)
        out["error"] = "while loading: " + short(e)
        return out
    model = mujoco.MjModel.from_xml_string(sample["xml"])
    model.opt.timestep = job["dt"]
    data = mujoco.MjData(model)
    data.qpos[0], data.qvel[0] = sample["q0"], sample["qdot0"]
    mujoco.mj_forward(model, data)
    params = dict(job["params"])
    u_max, dt, spec = job["u_max"], job["dt"], job["error"]
    n = round(job["horizon_s"] / dt)
    err = np.empty(n)
    u = np.empty(n)
    saturated = 0
    for i in range(n):
        t = i * dt
        q, qdot = float(data.qpos[0]), float(data.qvel[0])
        if not (math.isfinite(q) and math.isfinite(qdot)):
            out["nan"], out["error"] = True, f"the simulation became unstable at t = {t:.3f} s"
            return out
        err[i] = tracking_error(spec, t, q, qdot)
        try:
            raw = control(t, [q], [qdot], params)
            cmd = float(raw[0]) if isinstance(raw, list | tuple | np.ndarray) else float(raw)
        except Exception as e:  # noqa: BLE001  (MemoryError included: the address-space limit surfaces here)
            out["error"] = f"at t = {t:.3f} s: " + short(e)
            return out
        if not math.isfinite(cmd):
            out["nan"], out["error"] = True, f"control returned a non-finite command at t = {t:.3f} s"
            return out
        if abs(cmd) > u_max:
            saturated += 1
            cmd = math.copysign(u_max, cmd)
        u[i] = cmd
        data.ctrl[0] = cmd
        mujoco.mj_step(model, data)

    t = np.arange(n) * dt
    inside = np.abs(err) <= job["tolerance"]
    if inside[-1]:
        outside = np.nonzero(~inside)[0]
        settle = float(t[outside[-1] + 1]) if outside.size else 0.0
        out["settle_s"] = settle
        out["success"] = settle <= job["time_limit_s"] + 1e-9
    if spec["kind"] != "velocity" and not spec.get("sine") and abs(err[0]) > 0:
        # How far it went past the target, against the direction it started from.
        out["overshoot"] = float(max(0.0, np.max(-math.copysign(1.0, err[0]) * err)))
    hold = t >= job["horizon_s"] - job["hold_s"]
    out["final_error"] = float(np.mean(np.abs(err[hold])))
    out["peak_u"] = float(np.max(np.abs(u)))
    out["rms_u"] = float(np.sqrt(np.mean(u * u)))
    out["saturation"] = saturated / n
    return out


def main() -> None:
    result_fd = os.dup(1)
    # Anything the controller manages to write to stdout lands here, never in the result channel.
    sys.stdout = io.StringIO()
    job = json.loads(sys.stdin.read())
    try:
        compiled = compile(job["code"], "controller.py", "exec")
    except SyntaxError as e:
        results = [{"success": False, "error": short(e)} for _ in job["samples"]]
    else:
        results = [run_sample(compiled, job, s) for s in job["samples"]]
    payload = memoryview((json.dumps({"samples": results}) + "\n").encode())
    while payload:
        payload = payload[os.write(result_fd, payload):]


if __name__ == "__main__":
    main()
