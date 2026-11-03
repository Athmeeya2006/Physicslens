"""A small body moving under the Sun and the planets, heliocentric, in AU and days.

Used to check a law found from video against JPL: the law gives the Sun's GM, and Newton's pull of the planets (their
positions and masses from JPL Horizons) is added to see which part of a miss the planets explain. Pure math, no
network: callers pass the planets' tabulated positions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

AU_KM = 149_597_870.7


def gm_au3_d2(gm_km3_s2: float) -> float:
    return gm_km3_s2 * 86400**2 / AU_KM**3


@dataclass(frozen=True)
class Body:
    name: str
    gm: float  # AU^3/day^2
    where: Callable[[float], np.ndarray]  # heliocentric position (AU) at a Julian date

    @staticmethod
    def from_table(name: str, gm_km3_s2: float, table: np.ndarray) -> Body:
        """table rows: jd, x, y, z (AU, heliocentric). A cubic spline between rows, so daily rows serve a planet."""
        return Body(name, gm_au3_d2(gm_km3_s2), CubicSpline(table[:, 0], table[:, 1:4]))


def acceleration(r: np.ndarray, jd: float, gm_sun: float, bodies: list[Body]) -> np.ndarray:
    a = -gm_sun * r / np.linalg.norm(r) ** 3
    for b in bodies:
        rp = b.where(jd)
        d = rp - r
        # Heliocentric frame: the body pulls the small body, and pulls the Sun as well (the second, indirect term).
        a = a + b.gm * (d / np.linalg.norm(d) ** 3 - rp / np.linalg.norm(rp) ** 3)
    return a


def propagate(state0: np.ndarray, jd0: float, jd1: float, gm_sun: float, bodies: list[Body],
              rtol: float = 1e-12, atol: float = 1e-15, extra=None):
    """Integrate [x, y, z, vx, vy, vz] from jd0 to jd1; returns a function of Julian date giving the state there.

    extra(pos, vel, jd), when given, adds an acceleration that depends on the velocity too (relativity, a thermal push)."""
    def rhs(t: float, s: np.ndarray) -> np.ndarray:
        a = acceleration(s[:3], jd0 + t, gm_sun, bodies)
        if extra is not None:
            a = a + extra(s[:3], s[3:], jd0 + t)
        return np.concatenate([s[3:], a])

    sol = solve_ivp(rhs, (0.0, jd1 - jd0), np.asarray(state0, float), method="DOP853", rtol=rtol, atol=atol,
                    dense_output=True)
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return lambda jd: sol.sol(np.asarray(jd) - jd0)
