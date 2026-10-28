"""Combine one quantity measured in several recordings: a weighted mean, and a test that they actually agree.

Pure math. Recordings that disagree beyond their stated uncertainties are reported as disagreeing (something
changed, or one measurement is off), never averaged into a falsely precise number.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import chi2

CONSISTENT_P = 0.01  # below this the recordings disagree beyond their error bars


def combine(results: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """results: (recording label, analysis result). Uses accepted results only, parameter by parameter."""
    by_name: dict[str, dict[str, Any]] = {}
    for label, r in results:
        if r.get("status") != "accepted":
            continue
        for p in r.get("parameters", []):
            if p.get("derived") or not p.get("std") or p["std"] <= 0:
                continue  # given or assumed values are not measurements to average
            entry = by_name.setdefault(p["name"], {"name": p["name"], "latex": p["latex"], "unit": p["unit"],
                                                   "uncalibrated": bool(p.get("uncalibrated")), "values": []})
            if entry["unit"] != p["unit"]:
                continue
            entry["values"].append({"recording": label, "value": float(p["value"]), "std": float(p["std"])})
    out = []
    for entry in by_name.values():
        v = np.array([x["value"] for x in entry["values"]])
        s = np.array([x["std"] for x in entry["values"]])
        w = 1.0 / s**2
        mean = float(np.sum(w * v) / np.sum(w))
        se = float(np.sqrt(1.0 / np.sum(w)))
        n = len(v)
        if n > 1:
            stat = float(np.sum(((v - mean) / s) ** 2))
            p_value = float(chi2.sf(stat, n - 1))
            consistent = p_value >= CONSISTENT_P
        else:
            p_value, consistent = None, True
        out.append({**entry, "value": mean, "std": se, "n": n, "p_value": p_value, "consistent": consistent})
    return out
