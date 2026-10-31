"""Known input signals for a run: measured, never fitted, recorded with where they came from.

A robot's logged motor command, a second body's measured path, or a simulated perturber's position can enter a law
the agent writes (physics/custom.py). They arrive in clip time (seconds from the start of the uploaded video) and are
shifted here to the track's time base (seconds from the first tracked frame), which is what the fit uses.
"""

from __future__ import annotations

import csv
import io
import math
import re
from typing import Any

import numpy as np

from .physics.custom import MAX_INPUT_CHARS, InputSignals

MAX_INPUT_ROWS = 100_000
MAX_INPUT_COLUMNS = 6
MAX_INPUT_BYTES = 4 * 1024 * 1024
_HEADER = re.compile(rf"^\s*([A-Za-z][A-Za-z0-9]{{0,{MAX_INPUT_CHARS - 1}}})\s*(?:\[([^\]]{{0,24}})\])?\s*$")


class InputError(ValueError):
    pass


def parse_inputs_csv(data: bytes, provenance: str = "uploaded CSV") -> dict[str, Any]:
    """A CSV with a header 't,u[N],x1[m]': time in seconds from the start of the clip, then one column per signal
    with an optional unit in brackets. Returns the raw inputs for RunOptions.inputs (clip time)."""
    if len(data) > MAX_INPUT_BYTES:
        raise InputError(f"The inputs file is larger than {MAX_INPUT_BYTES // (1024 * 1024)} MB.")
    try:
        rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
    except UnicodeDecodeError as e:
        raise InputError("The inputs file is not UTF-8 text.") from e
    rows = [r for r in rows if r and any(c.strip() for c in r)]
    if len(rows) < 5:
        raise InputError("The inputs file needs a header and at least 4 rows.")
    header, body = rows[0], rows[1:]
    if len(body) > MAX_INPUT_ROWS:
        raise InputError(f"The inputs file has more than {MAX_INPUT_ROWS} rows.")
    if not header or header[0].strip().lower() not in ("t", "t[s]", "time", "time_s"):
        raise InputError("The first column must be the time in seconds from the start of the clip, headed 't'.")
    if not 2 <= len(header) <= MAX_INPUT_COLUMNS + 1:
        raise InputError(f"Give between 1 and {MAX_INPUT_COLUMNS} signals after the time column.")
    names, units = [], {}
    for h in header[1:]:
        m = _HEADER.match(h)
        if not m:
            raise InputError(f"Column {h!r}: a name of letters and digits (at most {MAX_INPUT_CHARS}), "
                             "optionally followed by a unit in brackets, e.g. x1[m].")
        if m.group(1) in names:
            raise InputError(f"Column {m.group(1)!r} appears twice.")
        names.append(m.group(1))
        units[m.group(1)] = (m.group(2) or "").strip()
    try:
        table = np.array([[float(c) for c in r] for r in body], dtype=float)
    except ValueError as e:
        raise InputError(f"Every value must be a number: {e}") from e
    if table.shape[1] != len(header):
        raise InputError("Every row needs one value per column.")
    if not np.all(np.isfinite(table)):
        raise InputError("The inputs file contains NaN or infinite values.")
    if np.any(np.diff(table[:, 0]) <= 0):
        raise InputError("Times must increase from row to row.")
    return {"t": table[:, 0].tolist(), "values": {n: table[:, i + 1].tolist() for i, n in enumerate(names)},
            "units": units, "provenance": dict.fromkeys(names, provenance)}


def track_signals(raw: dict[str, Any] | None, first_frame: int, fps: float) -> InputSignals | None:
    """Clip-time inputs on the track's time base (seconds from the first tracked frame)."""
    if not raw or not raw.get("values"):
        return None
    shift = first_frame / fps if fps > 0 and math.isfinite(fps) else 0.0
    t = tuple(float(v) - shift for v in raw["t"])
    return InputSignals(t, {k: tuple(float(x) for x in v) for k, v in raw["values"].items()},
                        dict(raw.get("units") or {}), dict(raw.get("provenance") or {}))


def coverage_note(signals: InputSignals | None, t_track: np.ndarray) -> str | None:
    """Laws using the inputs can only be simulated where the inputs were measured: say so when that is short."""
    if signals is None or len(t_track) == 0:
        return None
    lo, hi = signals.t[0], signals.t[-1]
    if lo <= float(t_track[0]) + 1e-9 and hi >= float(t_track[-1]) - 1e-9:
        return None
    return (f"The known inputs cover {lo:.2f} s to {hi:.2f} s of the track, which runs {t_track[0]:.2f} s to "
            f"{t_track[-1]:.2f} s: laws that use them cannot be tested on this clip.")


def describe(signals: InputSignals | None) -> list[dict[str, Any]]:
    """For the result, the trail and the agent: each input's name, unit, provenance and span (never its values)."""
    if signals is None:
        return []
    return [{"name": n, "unit": signals.units.get(n, ""), "provenance": signals.provenance.get(n, ""),
             "from_s": signals.t[0], "to_s": signals.t[-1], "samples": len(signals.t)} for n in sorted(signals.values)]


def mover_signals(frames: np.ndarray, centers_px: np.ndarray, first_frame: int, fps: float, px_per_m: float,
                  origin_px: np.ndarray, name: str) -> InputSignals:
    """A second tracked object's path as known inputs x2, y2: meters from the measured object's origin, y up, on the
    measured object's time base, so a law can use (x - x2) directly."""
    c = np.asarray(centers_px, dtype=float)
    o = np.asarray(origin_px, dtype=float)
    x, y = (c[:, 0] - o[0]) / px_per_m, -(c[:, 1] - o[1]) / px_per_m
    t = (np.asarray(frames, dtype=float) - first_frame) / fps
    where = f"second tracked object ({name})"
    return InputSignals(tuple(t), {"x2": tuple(x), "y2": tuple(y)}, {"x2": "m", "y2": "m"}, {"x2": where, "y2": where})


def merge_signals(a: InputSignals | None, b: InputSignals | None, t: np.ndarray) -> InputSignals | None:
    """Two sources on one time base (the track's own samples, inside both spans); a name in both keeps the first."""
    if a is None or b is None:
        return a or b
    lo, hi = max(a.t[0], b.t[0], float(t[0])), min(a.t[-1], b.t[-1], float(t[-1]))
    grid = np.asarray(t, dtype=float)
    grid = grid[(grid >= lo) & (grid <= hi)]
    if len(grid) < 4:
        return a  # no common span: the uploaded inputs stand, and the coverage note tells the user
    values, units, prov = {}, {}, {}
    for sig in (b, a):  # a last, so its names win
        for n, v in sig.values.items():
            values[n] = tuple(np.interp(grid, sig.t, v))
            units[n], prov[n] = sig.units.get(n, ""), sig.provenance.get(n, "")
    return InputSignals(tuple(grid), values, units, prov)
