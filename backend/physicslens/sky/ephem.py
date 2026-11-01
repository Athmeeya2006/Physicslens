"""Cached JPL states and masses (data/sky/, written by scripts/sky_fetch.py): the sky engine never needs a network."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .bodies import BY_NAME
from .nbody import AU_KM, Body, gm_au3_d2

DATA = Path(__file__).resolve().parents[3] / "data" / "sky"
J2000 = 2451545.0


class MissingSkyData(FileNotFoundError):
    pass


def _rows(name: str) -> list[list[str]]:
    path = DATA / name
    if not path.is_file():
        raise MissingSkyData(f"{path} is missing: run `python scripts/sky_fetch.py` once (free JPL Horizons API).")
    return [line.strip().split(",") for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


@lru_cache(maxsize=1)
def masses_km3_s2() -> dict[str, float]:
    path = DATA / "gm.json"
    if not path.is_file():
        raise MissingSkyData(f"{path} is missing: run `python scripts/sky_fetch.py` once (free JPL Horizons API).")
    return {k: float(v["gm_km3_s2"]) for k, v in json.loads(path.read_text())["bodies"].items()}


@lru_cache(maxsize=1)
def heliocentric_states() -> dict[str, np.ndarray]:
    """Every catalogue body's heliocentric state at J2000: [x, y, z, vx, vy, vz] in AU and AU/day."""
    return {r[0]: np.array([float(v) for v in r[2:8]]) for r in _rows("states_j2000.csv")}


def sun_barycentric() -> np.ndarray:
    return np.array([float(v) for v in _rows("sun_ssb_j2000.csv")[0][1:7]])


def jovian_states() -> dict[str, np.ndarray]:
    """The Galilean moons relative to Jupiter's centre at J2000."""
    return {r[0]: np.array([float(v) for v in r[2:8]]) for r in _rows("jovian_j2000.csv")}


def validation_positions() -> dict[str, np.ndarray]:
    """Yearly heliocentric JPL positions 2000 to 2010: name -> rows [jd, x, y, z]."""
    out: dict[str, list[list[float]]] = {}
    for r in _rows("validation.csv"):
        out.setdefault(r[0], []).append([float(v) for v in r[1:5]])
    return {k: np.array(v) for k, v in out.items()}


def sun_gm() -> float:
    return gm_au3_d2(masses_km3_s2()["Sun"])


def catalogue_system(names: list[str] | None = None, mass_scale: dict[str, float] | None = None) -> list[Body]:
    """Bodies for nbody.simulate at J2000, heliocentric: every catalogue body by default (the Sun is the centre,
    not a body). Massless test particles (Apophis, Halley) get gm 0."""
    states, gms = heliocentric_states(), masses_km3_s2()
    scale = mass_scale or {}
    out = []
    for name in names or [n for n in states]:
        cat = BY_NAME[name]
        gm = gm_au3_d2(gms[name]) * scale.get(name, 1.0) if cat.massive else 0.0
        out.append(Body(name, gm, states[name].copy(), massive=cat.massive and gm > 0))
    return out


__all__ = ["AU_KM", "J2000", "MissingSkyData", "catalogue_system", "heliocentric_states", "jovian_states",
           "masses_km3_s2", "sun_barycentric", "sun_gm", "validation_positions"]
