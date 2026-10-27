"""Turn a 2D track into the generalized coordinate each model family is written in."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..sigproc.trajectory import Trajectory

FAMILIES = ("pendulum", "oscillator", "projectile")


@dataclass(frozen=True)
class Coordinates:
    family: str
    t: np.ndarray
    q: np.ndarray  # (N, d)
    names: tuple[str, ...]
    # Geometry needed to map q back to 2D (for the ghost overlay) and model constants such as length.
    ctx: dict[str, float] = field(default_factory=dict)

    def to_xy(self, q: np.ndarray) -> np.ndarray:
        q = q[:, None] if q.ndim == 1 else q
        if self.family == "pendulum":
            th = q[:, 0]
            L = self.ctx["length_m"]
            return np.column_stack([self.ctx["pivot_x"] + L * np.sin(th), self.ctx["pivot_y"] - L * np.cos(th)])
        if self.family == "oscillator":
            s = q[:, 0]
            return np.column_stack([self.ctx["origin_x"] + s * self.ctx["axis_x"], self.ctx["origin_y"] + s * self.ctx["axis_y"]])
        return q


# Above this relative spread the swing's arc is too flat to measure the length (and so g) from geometry.
MAX_LENGTH_REL_STD = 0.03


def length_is_reliable(c: Coordinates) -> bool:
    return c.family == "pendulum" and c.ctx["length_std"] <= MAX_LENGTH_REL_STD * c.ctx["length_m"]


def fit_circle(xy: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) least-squares circle: (cx, cy, r)."""
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x**2 + y**2
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(cx), float(cy), float(np.sqrt(c + cx**2 + cy**2))


def _radius_std(xy: np.ndarray, reps: int = 200) -> float:
    """Bootstrap spread of the fitted length; it feeds g's uncertainty because g scales with L."""
    rng = np.random.default_rng(0)
    radii = [fit_circle(xy[rng.integers(0, len(xy), len(xy))])[2] for _ in range(reps)]
    return float(np.std(radii))


def to_coordinates(traj: Trajectory, family: str) -> Coordinates:
    xy = traj.xy
    if family == "pendulum":
        cx, cy, r = fit_circle(xy)
        if cy < xy[:, 1].mean():
            raise ValueError("Path curves the wrong way for a hanging pendulum (pivot below the bob).")
        theta = np.arctan2(xy[:, 0] - cx, -(xy[:, 1] - cy))
        return Coordinates(family, traj.t, theta[:, None], ("theta",),
                           {"pivot_x": cx, "pivot_y": cy, "length_m": r, "length_std": _radius_std(xy)})
    if family == "oscillator":
        mean = xy.mean(axis=0)
        _, _, vt = np.linalg.svd(xy - mean, full_matrices=False)
        axis = vt[0] if vt[0][1] >= 0 else -vt[0]  # point the axis up (or right) for readable signs
        s = (xy - mean) @ axis
        return Coordinates(family, traj.t, s[:, None], ("s",),
                           {"origin_x": float(mean[0]), "origin_y": float(mean[1]), "axis_x": float(axis[0]), "axis_y": float(axis[1])})
    if family == "projectile":
        return Coordinates(family, traj.t, xy.copy(), ("x", "y"))
    raise ValueError(f"Unknown family {family!r}; expected one of {FAMILIES}.")
