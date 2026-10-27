"""Candidate ODE models. The language model chooses among these by id, or writes a free-form law in the closed
math language of expr.py (see custom.py). Either way only fitting and held-out verification accept a model."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

_LENGTH_UNIT = re.compile(r"(^|[^a-z])m($|[^a-z])")


def needs_scale(unit: str) -> bool:
    """True for anything measured in meters (m, m/s^2, 1/m); angles, times and rates are scale-free."""
    return bool(_LENGTH_UNIT.search(unit or ""))


Rhs = Callable[[np.ndarray, dict[str, float], dict[str, float]], np.ndarray]


@dataclass(frozen=True)
class Param:
    name: str
    latex: str
    unit: str
    lower: float
    upper: float
    guess: float | None = None  # starting point for the fit; set for free-form parameters, which have no rule of thumb


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
                  r"\dot{\mathbf{v}} = -g\hat{y} - k\mathbf{v}", (G, Param("k", "k", "1/s", 0.0, 50.0)), _projectile("linear")),
        ModelSpec("quadratic_drag", "projectile", "Projectile with quadratic drag",
                  r"\dot{\mathbf{v}} = -g\hat{y} - c|\mathbf{v}|\mathbf{v}", (G, Param("c", "c", "1/m", 0.0, 50.0)), _projectile("quadratic")),
    ]
}


SPIN_EPS = 0.05  # rad/s: smooths sign(w) so Coulomb friction stops the spin without solver chatter


def _spin(coulomb: bool, viscous: bool, air: bool) -> Rhs:
    def rhs(s: np.ndarray, p: dict[str, float], ctx: dict[str, float]) -> np.ndarray:
        _, w = s
        a = 0.0
        if coulomb:
            a -= p["c"] * np.tanh(w / SPIN_EPS)
        if viscous:
            a -= p["k"] * w
        if air:
            a -= p["q"] * w * abs(w)
        return np.array([w, a])
    return rhs


C_DRY = Param("c", "c", "rad/s^2", 0.0, 200.0)
K_VISC = Param("k", "k", "1/s", 0.0, 50.0)
Q_AIR = Param("q", "q", "1/rad", 0.0, 50.0)
for _m in [
    ModelSpec("spin_free", "rotation", "Free spin (no friction)", r"\ddot\varphi = 0", (), _spin(False, False, False)),
    ModelSpec("spin_dry", "rotation", "Spin-down, dry friction", r"\ddot\varphi = -c\,\mathrm{sgn}(\dot\varphi)", (C_DRY,),
              _spin(True, False, False)),
    ModelSpec("spin_viscous", "rotation", "Spin-down, viscous friction", r"\ddot\varphi = -k\dot\varphi", (K_VISC,),
              _spin(False, True, False)),
    ModelSpec("spin_air", "rotation", "Spin-down, air drag", r"\ddot\varphi = -q\dot\varphi|\dot\varphi|", (Q_AIR,),
              _spin(False, False, True)),
    ModelSpec("spin_dry_viscous", "rotation", "Spin-down, dry and viscous friction",
              r"\ddot\varphi = -c\,\mathrm{sgn}(\dot\varphi) - k\dot\varphi", (C_DRY, K_VISC), _spin(True, True, False)),
]:
    MODELS[_m.id] = _m


def _line(accel: bool) -> Rhs:
    def rhs(s: np.ndarray, p: dict[str, float], ctx: dict[str, float]) -> np.ndarray:
        return np.array([s[1], p["a0"] if accel else 0.0])
    return rhs


def _plane(accel: bool) -> Rhs:
    def rhs(s: np.ndarray, p: dict[str, float], ctx: dict[str, float]) -> np.ndarray:
        ax, ay = (p["ax0"], p["ay0"]) if accel else (0.0, 0.0)
        return np.array([s[2], s[3], ax, ay])
    return rhs


# Controls for motion that fits no named family ("line": along one axis, "plane": anywhere in the image plane).
# Any free-form law the agent writes is tested against these, so it has to beat "nothing acts on it".
for _m in [
    ModelSpec("line_constant_velocity", "line", "Constant velocity (no net force)", r"\ddot s = 0", (), _line(False)),
    ModelSpec("line_constant_accel", "line", "Constant acceleration", r"\ddot s = a_0",
              (Param("a0", "a_0", "m/s^2", -100.0, 100.0, 0.0),), _line(True)),
    ModelSpec("plane_constant_velocity", "plane", "Constant velocity (no net force)", r"\ddot{\mathbf{r}} = 0", (), _plane(False)),
    ModelSpec("plane_constant_accel", "plane", "Constant acceleration",
              r"\ddot x = a_x,\quad \ddot y = a_y",
              (Param("ax0", "a_x", "m/s^2", -100.0, 100.0, 0.0), Param("ay0", "a_y", "m/s^2", -100.0, 100.0, 0.0)), _plane(True)),
]:
    MODELS[_m.id] = _m


def _double_rhs(s: np.ndarray, p: dict[str, float], ctx: dict[str, float]) -> np.ndarray:
    from .double import (
        rhs,  # local import: double.py imports coords, which must not import models at load time
    )
    return rhs(s, p, ctx["L1"], ctx["L2"])


# Fitted by its own short-horizon method (physics/double.py), not the whole-clip fit used for the others.
MODELS["double_pendulum"] = ModelSpec(
    "double_pendulum", "double_pendulum", "Double pendulum (point masses)",
    r"\ddot\theta_{1,2} = f(\theta_1, \theta_2, \dot\theta_1, \dot\theta_2;\ g, \mu, \gamma)",
    (G, Param("mu", r"\mu", "", 0.05, 20.0), Param("gamma", r"\gamma", "1/s", 0.0, 5.0)), _double_rhs)


def models_for(family: str) -> list[ModelSpec]:
    return [m for m in MODELS.values() if m.family == family]
