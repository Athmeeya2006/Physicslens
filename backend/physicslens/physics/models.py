"""Candidate ODE models. The language model chooses among these by id; it never writes the math that gets accepted."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Rhs = Callable[[np.ndarray, dict[str, float], dict[str, float]], np.ndarray]


@dataclass(frozen=True)
class Param:
    name: str
    latex: str
    unit: str
    lower: float
    upper: float


@dataclass(frozen=True)
class ModelSpec:
    id: str
    family: str
    title: str
    latex: str  # uses {param} placeholders, filled with fitted values for display
    params: tuple[Param, ...]
    rhs: Rhs  # state = [q..., qdot...]

    @property
    def param_names(self) -> list[str]:
        return [p.name for p in self.params]


G = Param("g", "g", "m/s^2", 0.5, 40.0)


def _pendulum(damping: str | None, small_angle: bool = False) -> Rhs:
    def rhs(s: np.ndarray, p: dict[str, float], ctx: dict[str, float]) -> np.ndarray:
        th, w = s
        restoring = th if small_angle else np.sin(th)
        a = -(p["g"] / ctx["length_m"]) * restoring
        if damping == "linear":
            a -= p["gamma"] * w
        elif damping == "quadratic":
            a -= p["beta"] * w * abs(w)
        return np.array([w, a])
    return rhs


def _oscillator(damped: bool) -> Rhs:
    def rhs(s: np.ndarray, p: dict[str, float], ctx: dict[str, float]) -> np.ndarray:
        x, v = s
        a = -p["omega"] ** 2 * (x - p["s0"])
        if damped:
            a -= 2 * p["zeta"] * p["omega"] * v
        return np.array([v, a])
    return rhs


def _projectile(drag: str | None) -> Rhs:
    def rhs(s: np.ndarray, p: dict[str, float], ctx: dict[str, float]) -> np.ndarray:
        _, _, vx, vy = s
        ax, ay = 0.0, -p["g"]
        if drag == "linear":
            ax, ay = ax - p["k"] * vx, ay - p["k"] * vy
        elif drag == "quadratic":
            speed = np.hypot(vx, vy)
            ax, ay = ax - p["c"] * speed * vx, ay - p["c"] * speed * vy
        return np.array([vx, vy, ax, ay])
    return rhs


MODELS: dict[str, ModelSpec] = {
    m.id: m
    for m in [
        ModelSpec("pendulum", "pendulum", "Simple pendulum",
                  r"\ddot\theta = -\frac{g}{L}\sin\theta", (G,), _pendulum(None)),
        ModelSpec("pendulum_small_angle", "pendulum", "Small-angle pendulum",
                  r"\ddot\theta = -\frac{g}{L}\theta", (G,), _pendulum(None, small_angle=True)),
        ModelSpec("pendulum_linear_damping", "pendulum", "Pendulum with linear damping",
                  r"\ddot\theta = -\frac{g}{L}\sin\theta - \gamma\dot\theta",
                  (G, Param("gamma", r"\gamma", "1/s", 0.0, 5.0)), _pendulum("linear")),
        ModelSpec("pendulum_quadratic_damping", "pendulum", "Pendulum with air drag",
                  r"\ddot\theta = -\frac{g}{L}\sin\theta - \beta\dot\theta|\dot\theta|",
                  (G, Param("beta", r"\beta", "1/rad", 0.0, 5.0)), _pendulum("quadratic")),
        ModelSpec("harmonic", "oscillator", "Harmonic oscillator",
                  r"\ddot s = -\omega^2 (s - s_0)",
                  (Param("omega", r"\omega", "rad/s", 0.1, 200.0), Param("s0", "s_0", "m", -10.0, 10.0)), _oscillator(False)),
        ModelSpec("damped_harmonic", "oscillator", "Damped harmonic oscillator",
                  r"\ddot s = -\omega^2 (s - s_0) - 2\zeta\omega\dot s",
                  (Param("omega", r"\omega", "rad/s", 0.1, 200.0), Param("s0", "s_0", "m", -10.0, 10.0),
                   Param("zeta", r"\zeta", "", 0.0, 2.0)), _oscillator(True)),
        ModelSpec("free_fall", "projectile", "Projectile, no drag",
                  r"\ddot x = 0,\quad \ddot y = -g", (G,), _projectile(None)),
        ModelSpec("linear_drag", "projectile", "Projectile with linear drag",
                  r"\dot{\mathbf v} = -g\hat y - k\mathbf v", (G, Param("k", "k", "1/s", 0.0, 50.0)), _projectile("linear")),
        ModelSpec("quadratic_drag", "projectile", "Projectile with quadratic drag",
                  r"\dot{\mathbf v} = -g\hat y - c|\mathbf v|\mathbf v", (G, Param("c", "c", "1/m", 0.0, 50.0)), _projectile("quadratic")),
    ]
}


def models_for(family: str) -> list[ModelSpec]:
    return [m for m in MODELS.values() if m.family == family]
