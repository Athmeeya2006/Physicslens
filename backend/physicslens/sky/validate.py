"""The sky engine against NASA/JPL, from the cached data alone (tests/test_sky.py runs every one of these).

Each function returns its numbers rather than asserting a hoped-for value: the tests state the limits, and the README
quotes what these return.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from .elements import angle_diff, to_elements
from .ephem import J2000, catalogue_system, heliocentric_states, sun_barycentric, sun_gm, validation_positions
from .nbody import AU_KM, Body, Options, simulate

ROOT = Path(__file__).resolve().parents[3]
MASSIVE = ["Mercury", "Venus", "Earth", "Moon", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto", "Ceres"]
ARCSEC = math.pi / 180 / 3600


def planets_ten_years() -> dict[str, float]:
    """Every massive catalogue body from JPL's J2000 state, 10 years on: the largest heliocentric position error
    (AU) of Earth, Jupiter and Saturn at JPL's yearly positions. What remains: the other asteroids, relativity (off
    here), and the Earth-Moon tides JPL models."""
    truth = validation_positions()
    jd = truth["Earth"][:, 0]
    run = simulate(sun_gm(), catalogue_system(MASSIVE), J2000, jd)
    return {name: float(np.max(np.linalg.norm(run.of(name)[0] - rows[:, 1:4], axis=1))) for name, rows in truth.items()}


def frames_agree(years: float = 10.0) -> float:
    """Heliocentric and barycentric runs of the same system: the largest difference (AU) in positions relative to
    the Sun over the span. They are the same physics written two ways, so this measures the integrator alone."""
    jd = J2000 + np.linspace(0, 365.25 * years, 41)
    names = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    sun_bary = sun_barycentric()
    # The same system around its barycentre: bodies shifted by the Sun's barycentric state, plus the Sun itself.
    bodies = [Body("Sun", sun_gm(), sun_bary.copy())]
    gm_total = sun_gm() + sum(b.gm for b in catalogue_system(names))
    for b in catalogue_system(names):
        bodies.append(Body(b.name, b.gm, b.state + sun_bary))
    # Make the barycentre exactly stationary in this frame (JPL's includes bodies we leave out).
    p = sum(b.gm * b.state[3:] for b in bodies) / gm_total
    x = sum(b.gm * b.state[:3] for b in bodies) / gm_total
    for b in bodies:
        b.state = b.state - np.concatenate([x, p])
    helio = simulate(sun_gm(), catalogue_system(names), J2000, jd)
    bary = simulate(0.0, bodies, J2000, jd, Options(frame="barycentric"))
    sun_path = bary.pos[:, 0]
    rel = bary.pos[:, 1:] - sun_path[:, None, :]
    return float(np.max(np.abs(rel - helio.pos)))


def mercury_relativity(years: float = 100.0, c_scale: float = 1.0) -> float:
    """Mercury's perihelion advance from the Sun's general relativity alone (Sun and Mercury, nothing else), in
    arcseconds per century: the run with the 1PN term minus the run without it."""
    jd = J2000 + np.array([0.0, 365.25 * years])
    longitudes = []
    for gr in (False, True):
        run = simulate(sun_gm(), catalogue_system(["Mercury"], mass_scale={"Mercury": 0.0}), J2000, jd,
                       Options(gr=gr, c_scale=c_scale))
        pos, vel = run.of("Mercury")
        el0 = to_elements(np.concatenate([pos[0], vel[0]]), sun_gm())
        el1 = to_elements(np.concatenate([pos[-1], vel[-1]]), sun_gm())
        longitudes.append(angle_diff(el1.longitude_of_perihelion, el0.longitude_of_perihelion))
    return (longitudes[1] - longitudes[0]) / ARCSEC / (years / 100.0)


def apophis_2029() -> dict[str, float]:
    """Apophis from JPL's J2000 state through its 2029 Earth flyby with every massive body: closest approach time
    (hours from JPL's) and distance (km), against JPL's close-approach record."""
    import json
    cad = json.loads((ROOT / "data" / "orbit" / "cad_2029.json").read_text())
    jd_cad, d_cad = float(cad["data"][0][2]), float(cad["data"][0][4]) * AU_KM
    jd = np.concatenate([[J2000], np.arange(jd_cad - 2, jd_cad + 2, 1 / 1440)])
    run = simulate(sun_gm(), catalogue_system([*MASSIVE, "Apophis"]), J2000, jd)
    d = np.linalg.norm(run.of("Apophis")[0][1:] - run.of("Earth")[0][1:], axis=1) * AU_KM
    k = int(np.argmin(d))
    return {"hours": float((jd[1:][k] - jd_cad) * 24), "km": float(d[k]), "jpl_km": d_cad}


def halley_2061(planets: bool = True) -> dict[str, float]:
    """Halley from JPL's J2000 state to its next perihelion: days from JPL's 2061 date. Sun-only and all-planets
    runs differ by about the planets' share of the README's 2063-against-2061 gap; the numbers are returned, not
    assumed."""
    later = np.loadtxt(ROOT / "data" / "halley" / "halley_2061.csv", delimiter=",", skiprows=2)
    jpl_peri = float(later[np.argmin(np.linalg.norm(later[:, 1:4], axis=1)), 0])
    jd = np.concatenate([[J2000], np.arange(jpl_peri - 900, jpl_peri + 900, 0.05)])
    names = [*MASSIVE, "Halley"] if planets else ["Halley"]
    run = simulate(sun_gm(), catalogue_system(names), J2000, jd)
    r = np.linalg.norm(run.of("Halley")[0][1:], axis=1)
    k = int(np.argmin(r))
    return {"days": float(jd[1:][k] - jpl_peri), "q_au": float(r[k])}


def performance(years: float = 100.0, particles: int = 10) -> dict[str, float]:
    """8 planets and test particles for a century, daily output: the wall time (seconds)."""
    planets = catalogue_system(["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"])
    rng = np.random.default_rng(0)
    base = heliocentric_states()["Ceres"]
    tests = [Body(f"particle {i}", 0.0, base * (1 + rng.normal(0, 0.01, 6)), massive=False) for i in range(particles)]
    started = time.monotonic()
    simulate(sun_gm(), planets + tests, J2000, J2000 + np.arange(1, 365.25 * years, 1.0))
    return {"seconds": time.monotonic() - started, "bodies": len(planets), "particles": particles, "years": years}
