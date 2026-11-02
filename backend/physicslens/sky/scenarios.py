"""Scenarios (data/sky/scenarios/*.yaml): a system, a clip, and what to compare it with, run by the sky engine.

A real scenario starts where the body really was: JPL's J2000 states, carried to the scenario's start with every
catalogue body (or the scenario's seed_bodies, for the slow outer planets), and only then switched to the scenario's own
physics (Sun only, two giants, a ten-times Jupiter...). So "the Sun alone" means the real Apophis of January 2022 under
the Sun alone, not a different asteroid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .bodies import JUPITER_J2, JUPITER_J2_RADIUS_KM
from .elements import Elements, to_state
from .ephem import DATA, J2000, catalogue_system, jovian_states, masses_km3_s2, sun_gm
from .nbody import (
    AU_KM,
    Body,
    J2Field,
    Options,
    Trajectories,
    gm_au3_d2,
    km_per_s_to_au_per_day,
    passing_star,
    simulate,
)
from .render import TRACKING_NOISE_PX, Detectability, detectability, plane_basis

SCENARIOS = DATA / "scenarios"
JD_UNIX0 = 2440587.5
EVERY_MASSIVE = ["Mercury", "Venus", "Earth", "Moon", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto", "Ceres"]
# Jupiter's spin axis (IAU WGCCRE, Archinal et al. 2018: RA 268.056595 deg, Dec 64.495303 deg, J2000 equatorial),
# rotated into the ecliptic frame the states use (obliquity 23.4392911 deg).
_RA, _DEC, _EPS = math.radians(268.056595), math.radians(64.495303), math.radians(23.4392911)
_EQ = (math.cos(_DEC) * math.cos(_RA), math.cos(_DEC) * math.sin(_RA), math.sin(_DEC))
JUPITER_POLE = (_EQ[0], _EQ[1] * math.cos(_EPS) + _EQ[2] * math.sin(_EPS), -_EQ[1] * math.sin(_EPS) + _EQ[2] * math.cos(_EPS))
MAX_BODIES = 30
MAX_PARTICLES = 200
MAX_YEARS = 300.0


class ScenarioError(ValueError):
    pass


def jd_of(date: str) -> float:
    fmt = "%Y-%m-%d %H:%M" if " " in date else "%Y-%m-%d"
    return datetime.strptime(date, fmt).replace(tzinfo=UTC).timestamp() / 86400 + JD_UNIX0


def date_of(jd: float) -> str:
    return datetime.fromtimestamp((jd - JD_UNIX0) * 86400, UTC).strftime("%Y-%m-%d %H:%M")


@dataclass
class Scenario:
    name: str
    title: str
    description: str
    fictional: bool
    start: str
    clip_days: float
    frame_step_days: float
    target: str
    bodies: list[str]
    predict_until: str | None = None
    hidden: list[str] = field(default_factory=list)
    mass_scale: dict[str, float] = field(default_factory=dict)
    gr: bool = False
    c_scale: float = 1.0
    j2: bool = False
    system: str = "solar"  # or "jovian": around Jupiter's centre
    seed_bodies: list[str] | None = None
    extra_particle: dict[str, Any] | None = None
    extra_body: dict[str, Any] | None = None  # a hypothetical massive body: gm_suns plus elements as extra_particle
    star: dict[str, Any] | None = None
    # A push to a body's starting velocity, in m/s: [along its motion, outward from the centre, out of its orbit's plane].
    kicks: dict[str, list[float]] = field(default_factory=dict)
    baseline: str | None = None
    compare: str = "none"

    def summary(self) -> dict[str, Any]:
        return {"name": self.name, "title": self.title, "description": self.description, "fictional": self.fictional,
                "target": self.target, "start": self.start, "clip_days": self.clip_days, "bodies": self.bodies,
                "hidden": self.hidden, "mass_scale": self.mass_scale, "gr": self.gr, "c_scale": self.c_scale,
                "baseline": self.baseline, "compare": self.compare}


def names() -> list[str]:
    return sorted(p.stem for p in SCENARIOS.glob("*.yaml"))


def load(name: str) -> Scenario:
    path = SCENARIOS / f"{name}.yaml"
    if not path.is_file() or path.parent != SCENARIOS:
        raise ScenarioError(f"No scenario named {name!r}; known: {', '.join(names())}")
    raw = yaml.safe_load(path.read_text())
    return Scenario(**{k: v for k, v in raw.items() if k in Scenario.__dataclass_fields__})


def validate(sc: Scenario) -> None:
    if sc.clip_days <= 0 or sc.frame_step_days <= 0:
        raise ScenarioError("The clip and the frame step must be positive.")
    span = max(sc.clip_days, (jd_of(sc.predict_until) - jd_of(sc.start)) if sc.predict_until else 0.0)
    if span > MAX_YEARS * 365.25:
        raise ScenarioError(f"At most {MAX_YEARS:.0f} years.")
    if len(sc.bodies) > MAX_BODIES:
        raise ScenarioError(f"At most {MAX_BODIES} bodies.")
    for k, v in sc.mass_scale.items():
        if not (math.isfinite(v) and 0 <= v <= 1000):
            raise ScenarioError(f"mass_scale for {k} must be between 0 and 1000.")
    if not (math.isfinite(sc.c_scale) and 0 < sc.c_scale <= 1e6):
        raise ScenarioError("c_scale must be between 0 and 1,000,000.")
    for k, v in sc.kicks.items():
        if len(v) != 3 or not all(math.isfinite(x) and abs(x) <= MAX_KICK_M_S for x in v):
            raise ScenarioError(f"The push on {k} must be three speeds of at most {MAX_KICK_M_S:,.0f} m/s.")
    if sc.clip_days / sc.frame_step_days > 20_000:
        raise ScenarioError("Too many frames (at most 20,000).")


@lru_cache(maxsize=16)
def _seeded(start_jd: float, seed: tuple[str, ...], particles: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Heliocentric states at start_jd: JPL's J2000 states carried there with the seed bodies' real masses."""
    if abs(start_jd - J2000) < 1e-9:
        return {b.name: b.state for b in catalogue_system([*seed, *particles])}
    run = simulate(sun_gm(), catalogue_system([*seed, *particles]), J2000, np.array([start_jd]))
    return {n: np.concatenate([run.pos[0, i], run.vel[0, i]]) for i, n in enumerate(run.names)}


def _particle_from_elements(spec: dict[str, Any], gm: float) -> np.ndarray:
    el = Elements(float(spec["a_au"]), float(spec.get("e", 0.0)), math.radians(float(spec.get("i_deg", 0.0))),
                  math.radians(float(spec.get("node_deg", 0.0))), math.radians(float(spec.get("peri_deg", 0.0))),
                  math.radians(float(spec.get("M_deg", 0.0))), gm)
    return to_state(el)


MAX_KICK_M_S = 20_000.0
M_S_TO_AU_DAY = 86400.0 / 149_597_870_700.0


def kicked(state: np.ndarray, dv_m_s: list[float]) -> np.ndarray:
    """The state with its velocity pushed by dv (m/s) along the motion, outward (radial, made perpendicular to the
    motion) and along the orbit's normal: the three directions a thruster or a close encounter is described in."""
    r, v = state[:3], state[3:6]
    along = v / np.linalg.norm(v)
    normal = np.cross(r, v)
    normal /= np.linalg.norm(normal)
    out = np.cross(along, normal)  # in the plane, perpendicular to the motion, pointing away from the centre
    if np.dot(out, r) < 0:
        out = -out
    dv = (dv_m_s[0] * along + dv_m_s[1] * out + dv_m_s[2] * normal) * M_S_TO_AU_DAY
    return np.concatenate([r, v + dv, state[6:]])


def _kick_all(bodies: list[Body], kicks: dict[str, list[float]]) -> list[Body]:
    for b in bodies:
        if b.name in kicks and any(kicks[b.name]):
            b.state = kicked(np.asarray(b.state, float), kicks[b.name])
    return bodies


@dataclass
class ScenarioRun:
    scenario: Scenario
    run: Trajectories
    frames_jd: np.ndarray  # the clip's frame times
    basis: np.ndarray  # the target's orbital plane over the clip
    central_gm: float
    units: tuple[str, str]  # length, time
    comparison: dict[str, Any] = field(default_factory=dict)
    detectability: Detectability | None = None


def build(sc: Scenario) -> tuple[float, list[Body], Options, float]:
    """(central GM, bodies, options, start jd) for a scenario, every body at its real (or stated) start."""
    validate(sc)
    start = jd_of(sc.start)
    if sc.system == "jovian":
        gms = masses_km3_s2()
        central = gm_au3_d2(gms["Jupiter (planet)"])
        states = jovian_states()
        bodies = [Body(n, gm_au3_d2(gms[n]) * sc.mass_scale.get(n, 1.0), states[n].copy()) for n in sc.bodies]
        j2 = J2Field(central, JUPITER_J2, JUPITER_J2_RADIUS_KM / AU_KM, JUPITER_POLE) if sc.j2 else None
        return central, _kick_all(bodies, sc.kicks), Options(j2=j2), start
    seed = tuple(sc.seed_bodies or EVERY_MASSIVE)
    particles = tuple(n for n in (sc.target,) if n in ("Apophis", "Halley"))
    states = _seeded(start, seed, particles)
    want = [*sc.bodies, *(p for p in particles if p not in sc.bodies)]
    bodies = []
    for b in catalogue_system(want, mass_scale=sc.mass_scale):
        bodies.append(Body(b.name, b.gm, states[b.name].copy(), massive=b.massive))
    if sc.extra_particle:
        bodies.append(Body(sc.extra_particle["name"], 0.0, _particle_from_elements(sc.extra_particle, sun_gm()),
                           massive=False))
    if sc.extra_body:
        bodies.append(Body(sc.extra_body.get("name", "hypothetical body"), sun_gm() * float(sc.extra_body["gm_suns"]),
                           _particle_from_elements(sc.extra_body, sun_gm())))
    prescribed = ()
    if sc.star:
        prescribed = (passing_star(sun_gm() * float(sc.star["gm_suns"]), float(sc.star["closest_au"]),
                                   km_per_s_to_au_per_day(float(sc.star["speed_km_s"])), jd_of(sc.star["closest"])),)
    return sun_gm(), _kick_all(bodies, sc.kicks), Options(gr=sc.gr, c_scale=sc.c_scale, prescribed=prescribed), start


def run(sc: Scenario, extra_jd: np.ndarray | None = None) -> ScenarioRun:
    central, bodies, options, start = build(sc)
    frames = start + np.arange(0.0, sc.clip_days, sc.frame_step_days)
    jd = frames if extra_jd is None else np.union1d(frames, extra_jd)
    traj = simulate(central, bodies, start, jd, options)
    pos, vel = traj.of(sc.target)
    in_clip = np.isin(jd, frames)
    basis = plane_basis(pos[in_clip], vel[in_clip])
    out = ScenarioRun(sc, traj, frames, basis, central, ("AU", "day"))
    if sc.baseline:
        base = run(load(sc.baseline))
        px_per_au = auto_scale(pos[in_clip] @ basis[:2].T)
        mine = (pos[in_clip] @ basis[:2].T) * px_per_au
        theirs = (base.run.of(sc.target)[0][np.isin(base.run.jd, base.frames_jd)] @ basis[:2].T) * px_per_au
        n = min(len(mine), len(theirs))
        out.detectability = detectability(theirs[:n], mine[:n], TRACKING_NOISE_PX)
    return out


def auto_scale(xy: np.ndarray, size: int = 1080) -> float:
    """Pixels per unit that fit the whole track and the origin into the frame with a margin."""
    pts = np.vstack([xy, [[0.0, 0.0]]])
    return 0.8 * size / float(np.ptp(pts, axis=0).max())


def track(sr: ScenarioRun, noise_px: float = TRACKING_NOISE_PX, px_per_unit: float | None = None,
          seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """What a tracker would see of the target: (t in days from the clip start, xy in the orbital plane, in AU) with
    Gaussian noise of noise_px pixels at the video's scale, on the frames the video would have (render_scenario keeps
    at most MAX_FRAMES, evenly strided: more frames of the same noise would make the measured accelerations far
    noisier than any clip's). The source is a simulation, never pixels."""
    frames = video_frames(sr.frames_jd)
    pos = sr.run.of(sr.scenario.target)[0][np.isin(sr.run.jd, frames)]
    xy = pos @ sr.basis[:2].T
    scale = px_per_unit or auto_scale(xy)
    if noise_px > 0:
        xy = xy + np.random.default_rng(seed).normal(0, noise_px / scale, xy.shape)
    return frames - frames[0], xy, scale


def video_frames(frames_jd: np.ndarray) -> np.ndarray:
    """The frame times a rendered clip keeps: every frame up to MAX_FRAMES, else an even stride."""
    stride = max(1, math.ceil(len(frames_jd) / MAX_FRAMES))
    return frames_jd[::stride][:MAX_FRAMES]


FRAME_SIZE, FRAME_FPS, MAX_FRAMES = 1080, 30, 590  # the analysis takes clips up to 20 s at 30 fps
DRAW_COLOURS = {"Sun": (60, 200, 255), "Earth": (230, 150, 60), "Mars": (60, 96, 210), "Jupiter": (102, 160, 217),
                "Saturn": (138, 207, 227), "Uranus": (227, 214, 143), "Neptune": (224, 111, 74),
                "Mercury": (168, 168, 168), "Venus": (122, 194, 230), "Moon": (207, 207, 207)}
FRAME_NOTE = ("Rendered top-down video of an orbital plane. The coordinate origin (0, 0) is the fixed yellow body at the "
              "centre of the frame. Units: wherever the numbers say m, read AU; wherever they say s, read day.")


def render_scenario(sr: ScenarioRun, out: Path) -> dict[str, Any]:
    """The video a scenario's target would make (the one the pixel analysis watches), and its geometry."""
    from .render import Drawn, Frame, project, render
    sc = sr.scenario
    keep = video_frames(sr.frames_jd)
    rows = np.isin(sr.run.jd, keep)
    scale = auto_scale(sr.run.of(sc.target)[0][rows] @ sr.basis[:2].T, FRAME_SIZE)
    origin = (FRAME_SIZE / 2, FRAME_SIZE / 2)
    drawn = [Drawn("Sun" if sc.system == "solar" else "Jupiter", np.array(origin), 18, DRAW_COLOURS["Sun"])]
    for i, body in enumerate(sr.run.names):
        if body != sc.target:
            drawn.append(Drawn(body, project(sr.run.pos[rows, i], sr.basis, scale, origin), 9,
                               DRAW_COLOURS.get(body, (180, 180, 180)), hidden=body in sc.hidden))
    drawn.append(Drawn(sc.target, project(sr.run.pos[rows, sr.run.names.index(sc.target)], sr.basis, scale, origin),
                       13, (235, 235, 235)))
    step = float(np.median(np.diff(keep))) if len(keep) > 1 else sc.frame_step_days
    label = "FICTIONAL scenario, simulated by PhysicsLens" if sc.fictional else "Simulated by PhysicsLens from NASA/JPL states"
    frame = Frame(FRAME_SIZE, FRAME_FPS, [date_of(j)[:10] for j in keep],
                  [f"{label}. 1 frame = {step:g} days.",
                   f"white: {sc.target}" + ("   hidden: " + ", ".join(sc.hidden) if sc.hidden else "")])
    meta = {"scenario": sc.name, "fictional": sc.fictional, "start_jd": float(keep[0]), "step_days": step,
            "px_per_au": scale, "sun_px": list(origin), "basis": sr.basis.tolist(), "target": sc.target}
    render(drawn, frame, out, meta)
    return {**meta, "frames": len(keep), "fps": FRAME_FPS}
