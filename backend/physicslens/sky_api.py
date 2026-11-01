"""The Solar System Lab's API: scenarios, simulations and free direct-track discovery (no model calls), plus rendering
a scenario for the paid pixel analysis behind the same access checks as an upload.

Every endpoint here but /render is pure CPU work on cached JPL data. They are still bounded: per-address rate limits,
a cap on concurrent simulations, a time limit per run, limits on bodies, years and output points, and a cache keyed by
the canonical request, since presets are asked for again and again.
"""

from __future__ import annotations

import json
import math
import threading
from collections import OrderedDict
from dataclasses import replace
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from .costguard.limiter import RequestWindows
from .sky import scenarios as S
from .sky.bodies import BY_NAME, GALILEAN_COLOURS
from .sky.discover import discover
from .sky.elements import to_elements
from .sky.ephem import masses_km3_s2
from .sky.nbody import SkyTooSlow

router = APIRouter(prefix="/api/sky")

LIMITS = {"max_bodies": 30, "max_particles": 200, "max_years": 300.0, "max_points": 2000, "per_ip_per_minute": 20,
          "global_per_minute": 120, "max_concurrent": 2, "max_seconds": 90.0, "cache_entries": 64}


def _windows() -> RequestWindows:
    return RequestWindows({"sky": {k: LIMITS[k] for k in ("per_ip_per_minute", "global_per_minute")}})


WINDOWS = _windows()


def configure(limits: dict[str, Any] | None) -> None:
    """From config/limits.yaml (sky:)."""
    global WINDOWS
    LIMITS.update({k: v for k, v in (limits or {}).items() if k in LIMITS})
    WINDOWS = _windows()


class Lax(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ExtraBody(Lax):
    name: str = Field(default="hypothetical body", max_length=40)
    gm_suns: float = Field(gt=0, le=0.1, description="mass as a fraction of the Sun's")
    a_au: float = Field(gt=0.05, le=1000)
    e: float = Field(default=0.0, ge=0, lt=0.99)
    i_deg: float = Field(default=0.0, ge=0, le=180)
    M_deg: float = Field(default=0.0, ge=-360, le=360)


class Star(Lax):
    gm_suns: float = Field(gt=0, le=2)
    closest_au: float = Field(gt=0, le=1e5)
    speed_km_s: float = Field(gt=0, le=1000)
    closest: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class SimulateRequest(Lax):
    scenario: str = Field(max_length=60)
    mass_scale: dict[str, Annotated[float, Field(ge=0, le=1000)]] = Field(default_factory=dict)
    disabled: list[str] = Field(default_factory=list, max_length=30)
    hidden: list[str] | None = Field(default=None, max_length=30)
    extra_body: ExtraBody | None = None
    star: Star | None = None
    gr: bool | None = None
    c_scale: float | None = Field(default=None, gt=0, le=1e6)
    years: float | None = Field(default=None, gt=0, le=300)


class DiscoverRequest(SimulateRequest):
    noise_px: float = Field(default=2.0, ge=0, le=20)


class _Cache:
    def __init__(self) -> None:
        self.items: OrderedDict[str, Any] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self.lock:
            if key in self.items:
                self.items.move_to_end(key)
                return self.items[key]
        return None

    def put(self, key: str, value: Any) -> None:
        with self.lock:
            self.items[key] = value
            while len(self.items) > int(LIMITS["cache_entries"]):
                self.items.popitem(last=False)


CACHE = _Cache()
_running = threading.BoundedSemaphore(int(LIMITS["max_concurrent"]))


def _rate_limit(request: Request) -> None:
    from .public import client_address
    addr = client_address(request.client.host if request.client else None, request.headers.get("x-forwarded-for"),
                          request.headers.get("cf-connecting-ip"))
    wait = WINDOWS.check("sky", addr)
    if wait is not None:
        raise HTTPException(status_code=429, detail="Too many simulations in a minute; please wait a moment.",
                            headers={"Retry-After": str(math.ceil(wait))})


def scenario_for(req: SimulateRequest) -> S.Scenario:
    try:
        sc = S.load(req.scenario)
    except S.ScenarioError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    known = set(sc.bodies) | {sc.target}
    for name in (*req.mass_scale, *req.disabled, *(req.hidden or [])):
        if name not in known:
            raise HTTPException(status_code=400, detail=f"{name!r} is not in this scenario ({', '.join(sorted(known))}).")
    if sc.target in req.disabled:
        raise HTTPException(status_code=400, detail="The target cannot be removed.")
    sc = replace(sc, bodies=[b for b in sc.bodies if b not in req.disabled],
                 mass_scale={**sc.mass_scale, **req.mass_scale}, hidden=req.hidden if req.hidden is not None else sc.hidden)
    if req.gr is not None:
        sc = replace(sc, gr=req.gr)
    if req.c_scale is not None:
        sc = replace(sc, c_scale=req.c_scale, gr=True if req.gr is None else req.gr)
    if req.years is not None:
        sc = replace(sc, clip_days=req.years * 365.25, frame_step_days=max(sc.frame_step_days, req.years * 365.25 / 5000))
    if req.extra_body is not None:
        sc = replace(sc, extra_body={**req.extra_body.model_dump(), "node_deg": 0.0, "peri_deg": 0.0})
    if req.star is not None:
        sc = replace(sc, star=req.star.model_dump())
    edited = req.model_dump(exclude={"scenario", "noise_px"}, exclude_defaults=True)
    if edited:
        # Anything changed from the file is a what-if: marked as such, and compared with the scenario as written.
        sc = replace(sc, fictional=sc.fictional or bool(edited), baseline=req.scenario if edited else sc.baseline)
    if sc.c_scale != 1.0:
        sc = replace(sc, fictional=True)
    if len(sc.bodies) > int(LIMITS["max_bodies"]):
        raise HTTPException(status_code=400, detail=f"At most {LIMITS['max_bodies']} bodies.")
    if sc.clip_days > float(LIMITS["max_years"]) * 365.25:
        raise HTTPException(status_code=400, detail=f"At most {LIMITS['max_years']:.0f} years.")
    return sc


def _guarded(key: str, work):
    hit = CACHE.get(key)
    if hit is not None:
        return {**hit, "cached": True}
    if not _running.acquire(timeout=5):
        raise HTTPException(status_code=503, detail="The lab is busy with other simulations; please retry in a moment.",
                            headers={"Retry-After": "10"})
    try:
        out = work()
    except SkyTooSlow as e:
        raise HTTPException(status_code=503, detail=f"That simulation is too long for the lab: {e}.") from e
    except S.ScenarioError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        _running.release()
    CACHE.put(key, out)
    return {**out, "cached": False}


def _decimate(n: int, most: int) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, min(n, most)).astype(int))


def _simulation(sc: S.Scenario) -> dict[str, Any]:
    sr = S.run(sc)
    rows = np.isin(sr.run.jd, sr.frames_jd)
    keep = _decimate(int(rows.sum()), int(LIMITS["max_points"]))
    t = (sr.frames_jd - sr.frames_jd[0])[keep]
    bodies = []
    for i, name in enumerate(sr.run.names):
        # x, y in the target's orbital plane and z along its normal: the lab draws all three.
        xyz = (sr.run.pos[rows, i] @ sr.basis.T)[keep]
        cat = BY_NAME.get(name)
        colour = cat.colour if cat else GALILEAN_COLOURS.get(name, "#cccccc")
        bodies.append({"name": name, "colour": colour, "hidden": name in sc.hidden,
                       "target": name == sc.target, "radius_km": cat.radius_km if cat else 0.0,
                       "points": np.round(np.column_stack([t, xyz]), 7).tolist()})
    pos, vel = sr.run.of(sc.target)
    pick = np.flatnonzero(rows)[_decimate(int(rows.sum()), 400)]
    elements = []
    for k in pick:
        try:
            el = to_elements(np.concatenate([pos[k], vel[k]]), sr.central_gm)
        except ValueError:
            continue  # an unbound moment (a passing star's kick) has no ellipse
        elements.append({"t": float(sr.run.jd[k] - sr.frames_jd[0]), "a": el.a, "e": el.e,
                         "peri_longitude_deg": math.degrees(el.longitude_of_perihelion), "period_days": el.period})
    drift = None
    if len(elements) >= 2:
        span_centuries = (elements[-1]["t"] - elements[0]["t"]) / 36525
        if span_centuries > 0:
            unwrapped = np.unwrap(np.radians([e["peri_longitude_deg"] for e in elements]))
            drift = float(np.degrees(unwrapped[-1] - unwrapped[0]) * 3600 / span_centuries)
    return {"scenario": sc.summary(), "units": {"length": "AU", "time": "day"}, "bodies": bodies, "target": sc.target,
            "elements": elements, "precession_arcsec_per_century": drift,
            "energy_drift_per_century": sr.run.energy_drift, "warnings": sr.run.warnings,
            "detectability": None if sr.detectability is None else {
                "max_px": sr.detectability.max_px, "visible": sr.detectability.visible,
                "message": sr.detectability.message},
            "source": "simulation, no pixels", "seconds": sr.run.seconds}


def _key(kind: str, req: BaseModel) -> str:
    return kind + ":" + json.dumps(req.model_dump(), sort_keys=True)


@router.get("/scenarios")
def scenarios() -> list[dict[str, Any]]:
    return [S.load(n).summary() for n in S.names()]


@router.get("/bodies")
def bodies() -> list[dict[str, Any]]:
    gms = masses_km3_s2()
    return [{"name": b.name, "colour": b.colour, "massive": b.massive, "gm_km3_s2": gms.get(b.name),
             "gm_source": "JPL Horizons physical data (data/sky/gm.json)" if b.name in gms else None,
             "radius_km": b.radius_km} for b in BY_NAME.values()]


@router.post("/simulate")
def simulate_endpoint(request: Request, req: Annotated[SimulateRequest, Body()]) -> dict[str, Any]:
    _rate_limit(request)
    sc = scenario_for(req)
    return _guarded(_key("simulate", req), lambda: _simulation(sc))


@router.post("/discover")
def discover_endpoint(request: Request, req: Annotated[DiscoverRequest, Body()]) -> dict[str, Any]:
    _rate_limit(request)
    sc = scenario_for(req)
    return _guarded(_key("discover", req), lambda: discovery(sc, req.noise_px))


def discovery(sc: S.Scenario, noise_px: float) -> dict[str, Any]:
    """The law from a scenario's simulated track, with its detectability: what /discover returns and the static site
    bundles for each scenario as written."""
    sr = S.run(sc)
    out = discover(sr, noise_px=noise_px, bootstrap_reps=16)
    if sr.detectability is not None:
        out["detectability"] = {"max_px": sr.detectability.max_px, "visible": sr.detectability.visible,
                                "message": sr.detectability.message}
    return json.loads(json.dumps(out, default=float))  # plain numbers for the JSON response and the cache


def static_bundle(name: str) -> dict[str, Any]:
    """Everything the lab page needs to show a scenario without a server, and to re-run changes in the browser: the
    scenario's own simulation (as /simulate returns it) plus each body's 3D state and GM at the start, the central GM,
    and the plane the target is drawn in."""
    sc = S.load(name)
    central, bodies, options, start = S.build(sc)
    sim = _simulation(sc)
    sr_basis = S.run(sc).basis
    return {**sim, "initial": {"central_gm": central, "start_jd": start, "clip_days": sc.clip_days,
                               "frame_step_days": sc.frame_step_days, "basis": sr_basis.tolist(),
                               "gr": options.gr, "c_scale": options.c_scale,
                               "bodies": [{"name": b.name, "gm": b.gm, "massive": b.massive, "state": b.state.tolist()}
                                          for b in bodies]}}
