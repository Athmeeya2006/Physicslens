"""The N-body integrator: Newtonian gravity in AU and days, validated against NASA/JPL (sky/validate.py).

Massive bodies pull on each other (one vectorised pairwise sum, no Python loop over pairs). Test particles (asteroids,
comets, spacecraft) feel every massive body and pull on none, as a separate block, so adding many of them costs
O(massive x particles). Two frames give the same relative motion (tests/test_sky.py):

  heliocentric: the Sun at the origin. For a body at r and other massive bodies k at r_k,
      a = -G(M_sun + m) r/|r|^3 + sum_k G m_k [ (r_k - r)/|r_k - r|^3 - r_k/|r_k|^3 ]
  The second term inside the sum is the indirect term: the bodies pull the Sun too, and the frame moves with it.
  barycentric: plain pairwise Newton with the Sun as one of the bodies.

Optional terms, each off by default:
  - general relativity from the Sun, first post-Newtonian order (Schwarzschild, harmonic gauge):
      a_GR = GM/(c^2 r^3) [ (4 GM/r - v^2) r + 4 (r . v) v ]
    with c_scale multiplying 1/c^2: 1 is real physics, anything else is FICTIONAL (the lab's exaggerated relativity);
  - the central body's oblateness J2 (for a moon system around Jupiter);
  - prescribed perturbers: bodies moving on a given path, pulling but not pulled (a passing star, or a planet from a
    tabulated ephemeris);
  - per-body mass scaling (set when the system is built).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

AU_KM = 149_597_870.7
C_AU_PER_DAY = 173.144632674  # the speed of light, 299792.458 km/s, in AU/day
ENERGY_DRIFT_WARN_PER_CENTURY = 1e-9


def gm_au3_d2(gm_km3_s2: float) -> float:
    return gm_km3_s2 * 86400**2 / AU_KM**3


@dataclass
class Body:
    name: str
    gm: float  # AU^3/day^2 (0 for a test particle)
    state: np.ndarray  # [x, y, z, vx, vy, vz] in the run's frame at the start epoch, AU and AU/day
    massive: bool = True


@dataclass(frozen=True)
class Prescribed:
    """A body on a given path: it pulls, nothing pulls it (a passing star, a tabulated planet)."""

    name: str
    gm: float
    where: Callable[[float], np.ndarray]  # heliocentric position (AU) at a Julian date


@dataclass(frozen=True)
class J2Field:
    gm: float  # the central body's GM
    j2: float
    radius: float  # reference radius, same length unit as the states
    pole: tuple[float, float, float]  # unit vector of the spin axis in the states' frame


@dataclass(frozen=True)
class Options:
    frame: str = "heliocentric"  # or "barycentric" (then the central body must be one of the bodies, named central)
    central: str = "Sun"
    gr: bool = False
    c_scale: float = 1.0  # multiplies 1/c^2 in the GR term: anything but 1 is fictional physics
    j2: J2Field | None = None
    prescribed: tuple[Prescribed, ...] = ()
    rtol: float = 1e-12
    atol: float = 1e-14
    max_seconds: float = 120.0
    # "scipy" (DOP853, always available) or "rebound" (IAS15, only if `import rebound` works; never required). REBOUND
    # covers plain Newtonian runs in the heliocentric frame; relativity, J2 and prescribed bodies stay with scipy.
    backend: str = "scipy"
    # An extra acceleration on every body, (r, v) -> a, all (N, 3): a law's own extra term (the Apophis fit's small
    # velocity term), so a prediction runs the law exactly as it was found.
    extra: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None


@dataclass
class Trajectories:
    names: list[str]
    jd: np.ndarray  # (T,)
    pos: np.ndarray  # (T, N, 3)
    vel: np.ndarray  # (T, N, 3)
    energy_drift: float | None  # relative, per century, from the first to the last sample (massive bodies, Newtonian)
    angular_momentum_drift: float | None
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def of(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        i = self.names.index(name)
        return self.pos[:, i], self.vel[:, i]


class SkyTooSlow(RuntimeError):
    pass


def _pairwise(r: np.ndarray, gm: np.ndarray) -> np.ndarray:
    """Acceleration of each of N bodies from all the others: (N, 3)."""
    d = r[None, :, :] - r[:, None, :]  # d[i, j] = r_j - r_i
    dist3 = np.sum(d * d, axis=2) ** 1.5
    np.fill_diagonal(dist3, np.inf)
    return np.einsum("ijk,j->ik", d / dist3[:, :, None], gm)


def _from_sources(r: np.ndarray, src: np.ndarray, gm: np.ndarray) -> np.ndarray:
    """Acceleration of M points at r (M, 3) from K sources at src (K, 3)."""
    d = src[None, :, :] - r[:, None, :]
    dist3 = np.sum(d * d, axis=2) ** 1.5
    return np.einsum("ijk,j->ik", d / dist3[:, :, None], gm)


def gr_acceleration(r: np.ndarray, v: np.ndarray, gm: float, c_scale: float = 1.0) -> np.ndarray:
    """1PN Schwarzschild correction from a central mass at the origin, for points r, v (M, 3)."""
    rn = np.linalg.norm(r, axis=1)[:, None]
    v2 = np.sum(v * v, axis=1)[:, None]
    rv = np.sum(r * v, axis=1)[:, None]
    k = gm * c_scale / (C_AU_PER_DAY**2 * rn**3)
    return k * ((4 * gm / rn - v2) * r + 4 * rv * v)


def j2_acceleration(r: np.ndarray, f: J2Field) -> np.ndarray:
    """Oblateness of a central body at the origin: a = -3/2 J2 GM R^2 / r^5 [ (1 - 5 z^2/r^2) r + 2 z k ]."""
    k = np.asarray(f.pole, dtype=float)
    rn2 = np.sum(r * r, axis=1)[:, None]
    z = (r @ k)[:, None]
    return -1.5 * f.j2 * f.gm * f.radius**2 / rn2**2.5 * ((1 - 5 * z * z / rn2) * r + 2 * z * k)


def simulate(central_gm: float, bodies: list[Body], jd0: float, jd_eval: np.ndarray,
             options: Options | None = None) -> Trajectories:
    """Integrate from jd0 and sample at jd_eval (increasing, after jd0, or before it to run backward)."""
    o = options or Options()
    jd_eval = np.asarray(jd_eval, dtype=float)
    if o.backend == "rebound":
        return _simulate_rebound(central_gm, bodies, jd0, jd_eval, o)
    if o.backend != "scipy":
        raise ValueError(f"Unknown backend {o.backend!r}: 'scipy' or 'rebound'.")
    names = [b.name for b in bodies]
    massive = np.array([b.massive and b.gm > 0 for b in bodies])
    gm = np.array([b.gm if m else 0.0 for b, m in zip(bodies, massive, strict=True)])
    n = len(bodies)
    y0 = np.concatenate([np.array([b.state[:3] for b in bodies]).ravel(), np.array([b.state[3:] for b in bodies]).ravel()])
    helio = o.frame == "heliocentric"
    if not helio:
        if o.central not in names:
            raise ValueError("A barycentric run needs its central body among the bodies.")
        if o.gr or o.j2 is not None or o.prescribed:
            raise ValueError("Relativity, J2 and prescribed perturbers are implemented for the heliocentric frame.")
    mi = np.flatnonzero(massive)
    rest = np.flatnonzero(~massive)
    gm_m = gm[mi]
    # Each massive body's pull on itself is left out by zeroing its own entry in the (bodies x massive) weights.
    self_mask = np.ones((n, len(mi)))
    self_mask[mi, np.arange(len(mi))] = 0.0
    deadline = time.monotonic() + o.max_seconds
    calls = 0

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        # The hot loop (hundreds of thousands of calls per century): einsum and in-place arithmetic, no small-array
        # wrappers, because at this size numpy's per-call overhead is most of the cost.
        nonlocal calls
        calls += 1
        if calls % 500 == 0 and time.monotonic() > deadline:
            raise SkyTooSlow(f"the integration took longer than {o.max_seconds:.0f} s")
        r = y[: 3 * n].reshape(n, 3)
        v = y[3 * n:].reshape(n, 3)
        if helio:
            a = r * (-central_gm * np.einsum("ij,ij->i", r, r) ** -1.5)[:, None]
            if len(mi):
                rm = r[mi]
                d = rm[None, :, :] - r[:, None, :]  # d[i, k] = r_k - r_i
                with np.errstate(divide="ignore"):
                    w = np.einsum("ijk,ijk->ij", d, d) ** -1.5
                w[self_mask == 0] = 0.0
                a += np.einsum("ijk,ij->ik", d, w * gm_m)
                a -= (gm_m * np.einsum("ij,ij->i", rm, rm) ** -1.5) @ rm  # the indirect term, every body included
            if o.prescribed:
                jd = jd0 + t
                for p in o.prescribed:
                    rp = np.asarray(p.where(jd), dtype=float)
                    d1 = rp[None, :] - r
                    a += p.gm * (d1 * (np.einsum("ij,ij->i", d1, d1) ** -1.5)[:, None] - rp / np.linalg.norm(rp) ** 3)
            if o.gr:
                a += gr_acceleration(r, v, central_gm, o.c_scale)
            if o.j2 is not None:
                a += j2_acceleration(r, o.j2)
            if o.extra is not None:
                a += o.extra(r, v)
        else:
            a = np.zeros((n, 3))
            if len(mi):
                a[mi] = _pairwise(r[mi], gm_m)
                if len(rest):
                    a[rest] = _from_sources(r[rest], r[mi], gm_m)
        return np.concatenate([y[3 * n:], a.ravel()])

    t_eval = jd_eval - jd0
    span = (0.0, float(t_eval[-1])) if len(t_eval) else (0.0, 0.0)
    started = time.monotonic()
    sol = solve_ivp(rhs, span, y0, method="DOP853", t_eval=t_eval, rtol=o.rtol, atol=o.atol)
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    y = sol.y.T
    pos = y[:, : 3 * n].reshape(len(t_eval), n, 3)
    vel = y[:, 3 * n:].reshape(len(t_eval), n, 3)
    out = Trajectories(names, jd_eval, pos, vel, None, None, seconds=time.monotonic() - started)
    if len(t_eval) >= 2 and len(mi):
        e0, e1 = (energy(pos[k], vel[k], gm, central_gm if helio else 0.0) for k in (0, -1))
        l0, l1 = (angular_momentum(pos[k], vel[k], gm, central_gm if helio else 0.0) for k in (0, -1))
        centuries = max(abs(t_eval[-1]) / 36525.0, 1e-9)
        out.energy_drift = abs((e1 - e0) / e0) / centuries
        out.angular_momentum_drift = float(np.linalg.norm(l1 - l0) / np.linalg.norm(l0)) / centuries
        if out.energy_drift > ENERGY_DRIFT_WARN_PER_CENTURY and not (o.gr or o.prescribed or o.j2):
            out.warnings.append(f"Total energy drifted by {out.energy_drift:.1e} per century (the warning level is "
                                f"{ENERGY_DRIFT_WARN_PER_CENTURY:.0e}): tighten rtol/atol for this run.")
    return out


def rebound_available() -> bool:
    try:
        import rebound  # noqa: F401
    except ImportError:
        return False
    return True


def _simulate_rebound(central_gm: float, bodies: list[Body], jd0: float, jd_eval: np.ndarray, o: Options) -> Trajectories:
    """The same system with REBOUND's IAS15 (an adaptive 15th-order integrator), in G = 1 units: masses are GMs."""
    if not rebound_available():
        raise ValueError("The REBOUND backend needs `pip install rebound`; the default scipy backend needs nothing.")
    if o.frame != "heliocentric" or o.gr or o.j2 is not None or o.prescribed:
        raise ValueError("The REBOUND backend covers plain Newtonian heliocentric runs; use the scipy backend.")
    import rebound

    sim = rebound.Simulation()
    sim.G = 1.0
    sim.integrator = "ias15"
    sim.add(m=central_gm)
    # REBOUND wants every massive particle before the test particles; results go back in the caller's order.
    order = sorted(range(len(bodies)), key=lambda i: not (bodies[i].massive and bodies[i].gm > 0))
    for i in order:
        b = bodies[i]
        sim.add(m=b.gm if b.massive else 0.0, x=b.state[0], y=b.state[1], z=b.state[2],
                vx=b.state[3], vy=b.state[4], vz=b.state[5])
    sim.N_active = 1 + sum(1 for b in bodies if b.massive and b.gm > 0)
    n = len(bodies)
    pos, vel = np.zeros((len(jd_eval), n, 3)), np.zeros((len(jd_eval), n, 3))
    started = time.monotonic()
    for k, jd in enumerate(jd_eval):
        sim.integrate(jd - jd0, exact_finish_time=1)
        sun = sim.particles[0]
        for slot, i in enumerate(order):
            p = sim.particles[slot + 1]
            pos[k, i] = (p.x - sun.x, p.y - sun.y, p.z - sun.z)
            vel[k, i] = (p.vx - sun.vx, p.vy - sun.vy, p.vz - sun.vz)
    return Trajectories([b.name for b in bodies], jd_eval, pos, vel, None, None,
                        ["Integrated with REBOUND IAS15."], time.monotonic() - started)


def _barycentre(pos: np.ndarray, vel: np.ndarray, gm: np.ndarray, central_gm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions and velocities relative to the barycentre, with the central body (at the origin in a heliocentric
    run) included as the last row; and the GM of each row."""
    if central_gm > 0:
        pos = np.vstack([pos, np.zeros(3)])
        vel = np.vstack([vel, np.zeros(3)])
        gm = np.append(gm, central_gm)
    total = gm.sum()
    return pos - gm @ pos / total, vel - gm @ vel / total, gm


def energy(pos: np.ndarray, vel: np.ndarray, gm: np.ndarray, central_gm: float = 0.0) -> float:
    """Newtonian energy of the massive bodies divided by G (units of GM AU^2/day^2), barycentric."""
    r, v, g = _barycentre(pos, vel, gm, central_gm)
    keep = g > 0
    r, v, g = r[keep], v[keep], g[keep]
    kinetic = 0.5 * float(np.sum(g * np.sum(v * v, axis=1)))
    d = np.sqrt(np.sum((r[None] - r[:, None]) ** 2, axis=2))
    iu = np.triu_indices(len(g), 1)
    return kinetic - float(np.sum(g[iu[0]] * g[iu[1]] / d[iu]))


def angular_momentum(pos: np.ndarray, vel: np.ndarray, gm: np.ndarray, central_gm: float = 0.0) -> np.ndarray:
    r, v, g = _barycentre(pos, vel, gm, central_gm)
    return np.sum(g[:, None] * np.cross(r, v), axis=0)


def passing_star(gm: float, closest_au: float, speed_au_per_day: float, jd_closest: float,
                 direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
                 offset: tuple[float, float, float] = (0.0, 1.0, 0.0)) -> Prescribed:
    """A star on a straight line past the Sun: closest at jd_closest, at closest_au along `offset`, moving along
    `direction` (made perpendicular to it). Fictional unless its numbers come from a catalogue."""
    if speed_au_per_day <= 0 or closest_au <= 0:
        raise ValueError("A passing star needs a positive speed and closest distance.")
    b = np.asarray(offset, dtype=float)
    b = b / np.linalg.norm(b)
    u = np.asarray(direction, dtype=float)
    u = u - (u @ b) * b
    if np.linalg.norm(u) == 0:
        raise ValueError("The star's direction must not be along its offset.")
    u = u / np.linalg.norm(u)

    def where(jd: float) -> np.ndarray:
        return closest_au * b + speed_au_per_day * (jd - jd_closest) * u

    return Prescribed("passing star", gm, where)


def km_per_s_to_au_per_day(v: float) -> float:
    return v * 86400 / AU_KM


__all__ = ["AU_KM", "C_AU_PER_DAY", "Body", "J2Field", "Options", "Prescribed", "SkyTooSlow", "Trajectories",
           "angular_momentum", "energy", "gm_au3_d2", "gr_acceleration", "j2_acceleration", "km_per_s_to_au_per_day",
           "passing_star", "simulate"]
