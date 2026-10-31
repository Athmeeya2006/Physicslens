"""Dimensional analysis for the laws the agent writes: length L, time T, mass M, with rational exponents.

An acceleration law must come out as L T^-2. Sums need equal units, function arguments (sin, exp, log, ...) must be
dimensionless, and a power needs a constant exponent unless its base is dimensionless. A law that breaks this is
rejected before any fitting, with a message that says which term is wrong and what it would need. Unit text the
parser does not know is a warning, never a rejection: the agent must not be blocked by notation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction

from .expr import ExprError, Node


@dataclass(frozen=True)
class Dim:
    L: Fraction = Fraction(0)
    T: Fraction = Fraction(0)
    M: Fraction = Fraction(0)

    def __mul__(self, o: Dim) -> Dim:
        return Dim(self.L + o.L, self.T + o.T, self.M + o.M)

    def __truediv__(self, o: Dim) -> Dim:
        return Dim(self.L - o.L, self.T - o.T, self.M - o.M)

    def __pow__(self, k: Fraction) -> Dim:
        return Dim(self.L * k, self.T * k, self.M * k)

    @property
    def dimensionless(self) -> bool:
        return self == NONE

    def __str__(self) -> str:
        parts = []
        for sym, e in (("M", self.M), ("L", self.L), ("T", self.T)):
            if e:
                parts.append(sym if e == 1 else f"{sym}^{e}")
        return " ".join(parts) or "dimensionless"


NONE = Dim()
LENGTH, TIME, MASS = Dim(L=Fraction(1)), Dim(T=Fraction(1)), Dim(M=Fraction(1))
VELOCITY = LENGTH / TIME
ACCELERATION = LENGTH / TIME / TIME

# Unit words. Orbit mode writes AU and days, the Jovian system planet radii (R_J) and hours: the same dimensions.
_UNITS: dict[str, Dim] = {
    **dict.fromkeys(("m", "cm", "mm", "km", "AU", "au", "R_J", "RJ", "px"), LENGTH),
    **dict.fromkeys(("s", "ms", "min", "h", "hr", "hour", "hours", "day", "days", "d", "yr", "year"), TIME),
    **dict.fromkeys(("kg", "g"), MASS),
    "N": MASS * ACCELERATION, "Hz": NONE / TIME, "J": MASS * LENGTH * LENGTH / TIME / TIME,
    **dict.fromkeys(("rad", "deg", "1", "", "-", "rev", "turn", "cycle", "cycles"), NONE),
}
_TOKEN = re.compile(r"\s*(R_J|[A-Za-z]+|\d+(?:\.\d+)?|\*\*|[\^*/()-])")


def parse_unit(text: str | None) -> Dim | None:
    """'m/s^2' -> L T^-2; '1/(m^2 s^2)' -> L^-2 T^-2 (a space multiplies). None when the text is not understood."""
    text = (text or "").strip()
    if text in ("", "1", "-", "dimensionless", "none"):
        return NONE
    try:
        tokens = _tokens(text)
        dim, pos = _product(tokens, 0)
    except (ValueError, IndexError, ZeroDivisionError):
        return None
    return dim if pos == len(tokens) else None


def _tokens(text: str) -> list[str]:
    out, pos = [], 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise ValueError(text)
        out.append(m.group(1))
        pos = m.end()
    return out


def _product(tok: list[str], i: int) -> tuple[Dim, int]:
    dim, i = _factor(tok, i)
    while i < len(tok) and tok[i] != ")":
        if tok[i] == "/":
            rhs, i = _factor(tok, i + 1)
            dim = dim / rhs
        else:
            if tok[i] == "*":
                i += 1
            rhs, i = _factor(tok, i)
            dim = dim * rhs
    return dim, i


def _factor(tok: list[str], i: int) -> tuple[Dim, int]:
    if tok[i] == "(":
        dim, i = _product(tok, i + 1)
        if tok[i] != ")":
            raise ValueError("unbalanced")
        i += 1
    elif re.fullmatch(r"\d+(?:\.\d+)?", tok[i]):
        if float(tok[i]) != 1:
            raise ValueError("a number in a unit")
        dim, i = NONE, i + 1
    elif tok[i] in _UNITS:
        dim, i = _UNITS[tok[i]], i + 1
    else:
        raise ValueError(f"unknown unit {tok[i]}")
    if i < len(tok) and tok[i] in ("^", "**"):
        sign = 1
        i += 1
        if tok[i] == "-":
            sign, i = -1, i + 1
        dim, i = dim ** (sign * Fraction(tok[i])), i + 1
    return dim, i


@dataclass(frozen=True)
class UnitReport:
    ok: bool
    message: str = ""
    unknown: tuple[str, ...] = ()  # names whose unit text was not understood: their terms were not checked


def _const(node: Node) -> Fraction | None:
    """The value of a constant exponent (a number, possibly negated or a ratio of numbers), else None."""
    if node.kind == "num":
        return Fraction(float(node.value)).limit_denominator(1000)  # type: ignore[arg-type]
    if node.kind == "neg":
        v = _const(node.args[0])
        return None if v is None else -v
    if node.kind in ("div", "mul"):
        a, b = _const(node.args[0]), _const(node.args[1])
        if a is None or b is None or (node.kind == "div" and b == 0):
            return None
        return a / b if node.kind == "div" else a * b
    return None


class _Mismatch(ExprError):
    pass


def dim_of(node: Node, dims: dict[str, Dim | None]) -> Dim | None:
    """The dimension of an expression, or None when it depends on a name of unknown unit (not checked)."""
    k = node.kind
    if k == "num":
        # Small whole numbers, halves, thirds, quarters and pi are pure numbers (2*zeta*omega). Any other literal may
        # stand for a constant with units (9.81 for g, which the agent is told to write when only mu*g is fitted):
        # its unit is unknown, so its term is not checked rather than wrongly rejected.
        v = abs(float(node.value))  # type: ignore[arg-type]
        pure = v == round(v) and v <= 10 or any(abs(v * d - round(v * d)) < 1e-12 for d in (2, 3, 4)) and v <= 10
        return NONE if pure or abs(v - math.pi) < 1e-12 else None
    if k == "name":
        return dims.get(str(node.value))
    if k == "neg":
        return dim_of(node.args[0], dims)
    if k in ("add", "sub"):
        a, b = dim_of(node.args[0], dims), dim_of(node.args[1], dims)
        if a is not None and b is not None and a != b:
            raise _Mismatch(f"adds {a} to {b}")
        return a if a is not None else b
    if k in ("mul", "div"):
        a, b = dim_of(node.args[0], dims), dim_of(node.args[1], dims)
        if a is None or b is None:
            return None
        return a * b if k == "mul" else a / b
    if k == "pow":
        base = dim_of(node.args[0], dims)
        exponent = _const(node.args[1])
        e_dim = dim_of(node.args[1], dims)
        if e_dim is not None and not e_dim.dimensionless:
            raise _Mismatch(f"raises to a power with units ({e_dim})")
        if base is None:
            return None
        if base.dimensionless:
            return NONE
        if exponent is None:
            raise _Mismatch(f"raises a quantity with units ({base}) to a power that is not a constant number")
        return base ** exponent
    name = str(node.value)
    args = [dim_of(a, dims) for a in node.args]
    if name in ("sin", "cos", "tan", "tanh", "atan", "exp", "log"):
        if args[0] is not None and not args[0].dimensionless:
            raise _Mismatch(f"takes {name}() of a quantity with units ({args[0]}); its argument must be a pure number")
        return NONE
    if name == "sign":
        return NONE  # sign(v) is +1, 0 or -1 whatever v's unit
    if name == "abs":
        return args[0]
    if name == "sqrt":
        return None if args[0] is None else args[0] ** Fraction(1, 2)
    if name == "hypot":
        a, b = args
        if a is not None and b is not None and a != b:
            raise _Mismatch(f"takes hypot() of {a} and {b}")
        return a if a is not None else b
    return None


def _terms(node: Node) -> list[Node]:
    if node.kind in ("add", "sub"):
        return _terms(node.args[0]) + _terms(node.args[1])
    return [node]


def _hint(got: Dim, want: Dim) -> str:
    ratio = want / got
    if ratio == NONE / TIME:
        return ": did you mean it multiplied by a rate (a parameter in 1/s), as in c*v?"
    if ratio == TIME:
        return ": did you mean it divided by a rate, or multiplied by a time?"
    return f": it needs a factor in {ratio}"


def check_acceleration(trees: list[Node], dims: dict[str, Dim | None]) -> UnitReport:
    """Each tree must be an acceleration (L T^-2). Unknown names make their terms unchecked, never rejected."""
    unknown = tuple(sorted(n for n, d in dims.items() if d is None))
    for axis, tree in enumerate(trees, 1):
        for i, term in enumerate(_terms(tree), 1):
            try:
                got = dim_of(term, dims)
            except _Mismatch as e:
                return UnitReport(False, f"acceleration {axis}, term {i} {e}", unknown)
            if got is not None and got != ACCELERATION:
                return UnitReport(False, f"acceleration {axis}, term {i} has units {got}, expected {ACCELERATION}"
                                         f"{_hint(got, ACCELERATION)}", unknown)
    return UnitReport(True, "", unknown)
