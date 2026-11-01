"""Osculating Keplerian elements from state vectors and back, for any central GM (bound orbits, e < 1).

Angles in radians; a in the states' length unit; period and time of perihelion in their time unit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Elements:
    a: float  # semi-major axis
    e: float  # eccentricity
    i: float  # inclination
    node: float  # longitude of the ascending node, Omega
    peri: float  # argument of perihelion, omega
    M: float  # mean anomaly at the epoch
    gm: float

    @property
    def period(self) -> float:
        return 2 * math.pi * math.sqrt(self.a**3 / self.gm)

    @property
    def q(self) -> float:
        """Perihelion distance."""
        return self.a * (1 - self.e)

    def time_to_perihelion(self) -> float:
        """Time from the epoch until the next perihelion passage."""
        return ((2 * math.pi - self.M) % (2 * math.pi)) / (2 * math.pi) * self.period

    @property
    def longitude_of_perihelion(self) -> float:
        return (self.node + self.peri) % (2 * math.pi)


def to_elements(state: np.ndarray, gm: float) -> Elements:
    r, v = np.asarray(state[:3], float), np.asarray(state[3:6], float)
    rn, v2 = float(np.linalg.norm(r)), float(v @ v)
    h = np.cross(r, v)
    hn = float(np.linalg.norm(h))
    n = np.cross([0.0, 0.0, 1.0], h)
    nn = float(np.linalg.norm(n))
    ev = ((v2 - gm / rn) * r - float(r @ v) * v) / gm
    e = float(np.linalg.norm(ev))
    energy = v2 / 2 - gm / rn
    if energy >= 0:
        raise ValueError("Not a bound orbit (e >= 1): no period or semi-major axis.")
    a = -gm / (2 * energy)
    i = math.acos(max(-1.0, min(1.0, h[2] / hn)))
    node = math.atan2(n[1], n[0]) % (2 * math.pi) if nn > 1e-15 else 0.0
    if e > 1e-12:
        if nn > 1e-15:
            peri = math.acos(max(-1.0, min(1.0, float(n @ ev) / (nn * e))))
            if ev[2] < 0:
                peri = 2 * math.pi - peri
        else:  # in the reference plane: measure from the x axis, in the sense of the motion
            peri = math.atan2(ev[1], ev[0]) % (2 * math.pi)
            if h[2] < 0:
                peri = (2 * math.pi - peri) % (2 * math.pi)
        nu = math.acos(max(-1.0, min(1.0, float(ev @ r) / (e * rn))))
        if float(r @ v) < 0:
            nu = 2 * math.pi - nu
    else:  # circular: no perihelion; the angle is counted from the node (or the x axis)
        peri = 0.0
        ref = n / nn if nn > 1e-15 else np.array([1.0, 0.0, 0.0])
        nu = math.acos(max(-1.0, min(1.0, float(ref @ r) / rn)))
        if float(np.cross(ref, r) @ h) < 0:
            nu = 2 * math.pi - nu
    E = 2 * math.atan2(math.sqrt(1 - e) * math.sin(nu / 2), math.sqrt(1 + e) * math.cos(nu / 2))
    M = (E - e * math.sin(E)) % (2 * math.pi)
    return Elements(a, e, i, node, peri, M, gm)


def kepler_E(M: float, e: float) -> float:
    E = M if e < 0.8 else math.pi
    for _ in range(100):
        dE = (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
        E -= dE
        if abs(dE) < 1e-15:
            break
    return E


def to_state(el: Elements) -> np.ndarray:
    E = kepler_E(el.M % (2 * math.pi), el.e)
    nu = 2 * math.atan2(math.sqrt(1 + el.e) * math.sin(E / 2), math.sqrt(1 - el.e) * math.cos(E / 2))
    p = el.a * (1 - el.e**2)
    r = p / (1 + el.e * math.cos(nu))
    pos_pf = np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
    vel_pf = math.sqrt(el.gm / p) * np.array([-math.sin(nu), el.e + math.cos(nu), 0.0])
    cO, sO, ci, si, cw, sw = (math.cos(el.node), math.sin(el.node), math.cos(el.i), math.sin(el.i),
                              math.cos(el.peri), math.sin(el.peri))
    rot = np.array([[cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
                    [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
                    [sw * si, cw * si, ci]])
    return np.concatenate([rot @ pos_pf, rot @ vel_pf])


def angle_diff(a: float, b: float) -> float:
    """a - b wrapped to (-pi, pi]."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi
