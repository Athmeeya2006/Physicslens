"""A small, closed math language for equations the agent writes itself.

The agent may propose an acceleration law as text ("-mu*g*sign(v) - k*v"). It is parsed here by a
recursive-descent parser over a fixed grammar: numbers, declared names, + - * / ^, parentheses and a
whitelist of functions. Nothing is ever passed to eval, exec or sympy's parser, so a proposal can only
ever be arithmetic. The same tree is compiled for the ODE solver, printed as LaTeX for the UI and as Python for
the exported script, so what is shown is exactly what was fitted.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

MAX_CHARS = 200
MAX_NODES = 60
MAX_DEPTH = 12

FUNCTIONS: dict[str, tuple[int, str]] = {
    # name: (arity, LaTeX command)
    "sin": (1, r"\sin"),
    "cos": (1, r"\cos"),
    "tan": (1, r"\tan"),
    "tanh": (1, r"\tanh"),
    "atan": (1, r"\arctan"),
    "exp": (1, r"\exp"),
    "log": (1, r"\ln"),
    "sqrt": (1, r"\sqrt"),
    "abs": (1, ""),
    "sign": (1, r"\operatorname{sgn}"),
    "hypot": (2, ""),
}
CONSTANTS = {"pi": math.pi}

_TOKEN = re.compile(r"\s*(?:(\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|(\*\*|[-+*/^(),]))")
NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,11}$")


class ExprError(ValueError):
    pass


@dataclass(frozen=True)
class Node:
    kind: str  # num | name | neg | add | sub | mul | div | pow | call
    value: float | str | None = None
    args: tuple[Node, ...] = ()


def tokenize(text: str) -> list[str]:
    if len(text) > MAX_CHARS:
        raise ExprError(f"expression longer than {MAX_CHARS} characters")
    tokens, pos = [], 0
    text = text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise ExprError(f"unexpected character {text[pos:pos + 1]!r} at position {pos}")
        tokens.append(m.group(m.lastindex or 0))
        pos = m.end()
    return tokens


class _Parser:
    def __init__(self, tokens: list[str], names: set[str]):
        self.tokens, self.i, self.names, self.nodes = tokens, 0, names, 0

    def peek(self) -> str | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self, expected: str | None = None) -> str:
        tok = self.peek()
        if tok is None or (expected is not None and tok != expected):
            raise ExprError(f"expected {expected or 'more input'}, got {tok or 'end of expression'}")
        self.i += 1
        return tok

    def node(self, kind: str, value: float | str | None = None, args: tuple[Node, ...] = (), depth: int = 0) -> Node:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise ExprError(f"expression has more than {MAX_NODES} terms")
        if depth > MAX_DEPTH:
            raise ExprError(f"expression nested deeper than {MAX_DEPTH}")
        return Node(kind, value, args)

    def expr(self, d: int) -> Node:
        left = self.term(d)
        while self.peek() in ("+", "-"):
            op = self.take()
            left = self.node("add" if op == "+" else "sub", args=(left, self.term(d)), depth=d)
        return left

    def term(self, d: int) -> Node:
        left = self.unary(d)
        while self.peek() in ("*", "/"):
            op = self.take()
            left = self.node("mul" if op == "*" else "div", args=(left, self.unary(d)), depth=d)
        return left

    def unary(self, d: int) -> Node:
        if self.peek() == "-":
            self.take()
            return self.node("neg", args=(self.unary(d + 1),), depth=d)
        if self.peek() == "+":
            self.take()
            return self.unary(d + 1)
        return self.power(d)

    def power(self, d: int) -> Node:
        base = self.atom(d)
        if self.peek() in ("^", "**"):
            self.take()
            return self.node("pow", args=(base, self.unary(d + 1)), depth=d)  # right-associative
        return base

    def atom(self, d: int) -> Node:
        tok = self.take()
        if tok == "(":
            inner = self.expr(d + 1)
            self.take(")")
            return inner
        if tok[0].isdigit() or tok[0] == ".":
            value = float(tok)
            if not math.isfinite(value) or abs(value) > 1e6:
                raise ExprError(f"number {tok} is out of range")
            return self.node("num", value, depth=d)
        if tok in FUNCTIONS:
            arity = FUNCTIONS[tok][0]
            self.take("(")
            args = [self.expr(d + 1)]
            while self.peek() == ",":
                self.take()
                args.append(self.expr(d + 1))
            self.take(")")
            if len(args) != arity:
                raise ExprError(f"{tok} takes {arity} argument(s), got {len(args)}")
            return self.node("call", tok, tuple(args), depth=d)
        if tok in CONSTANTS:
            return self.node("num", CONSTANTS[tok], depth=d)
        if NAME.match(tok):
            if tok not in self.names:
                raise ExprError(f"unknown name {tok!r}; declare it as a parameter or use one of {sorted(self.names)}")
            return self.node("name", tok, depth=d)
        raise ExprError(f"unexpected {tok!r}")


def parse(text: str, names: set[str]) -> Node:
    p = _Parser(tokenize(text), names)
    if not p.tokens:
        raise ExprError("empty expression")
    tree = p.expr(0)
    if p.peek() is not None:
        raise ExprError(f"unexpected {p.peek()!r} after the end of the expression")
    return tree


def names_used(node: Node) -> set[str]:
    out = {str(node.value)} if node.kind == "name" else set()
    for a in node.args:
        out |= names_used(a)
    return out


def _sign(a: float) -> float:
    return float((a > 0) - (a < 0))


def _pow(a: float, b: float) -> float:
    r = a**b
    return math.nan if isinstance(r, complex) else r


_SCALAR: dict[str, Callable[..., float]] = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "tanh": math.tanh, "atan": math.atan, "exp": math.exp,
    "log": math.log, "sqrt": math.sqrt, "abs": abs, "sign": _sign, "hypot": math.hypot,
}
_BINARY: dict[str, Callable[[float, float], float]] = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b, "div": lambda a, b: a / b, "pow": _pow,
}


def compile_expr(node: Node) -> Callable[[dict[str, float]], float]:
    """Nested closures over math functions: built once, then called thousands of times inside the ODE solver."""
    k = node.kind
    if k == "num":
        c = float(node.value)  # type: ignore[arg-type]
        return lambda env: c
    if k == "name":
        name = str(node.value)
        return lambda env: env[name]
    args = [compile_expr(a) for a in node.args]
    if k == "neg":
        (f,) = args
        return lambda env: -f(env)
    if k in _BINARY:
        op, (f, g) = _BINARY[k], args
        return lambda env: op(f(env), g(env))
    fn = _SCALAR[str(node.value)]
    if len(args) == 1:
        (f,) = args
        return lambda env: fn(f(env))
    f, g = args
    return lambda env: fn(f(env), g(env))


def safe_eval(fn: Callable[[dict[str, float]], float], env: dict[str, float]) -> float:
    """Domain errors (log of a negative, overflow, division by zero) mean the law broke down: NaN, not a crash."""
    try:
        return float(fn(env))
    except (ValueError, OverflowError, ZeroDivisionError):
        return math.nan


_PREC = {"add": 1, "sub": 1, "mul": 2, "div": 2, "neg": 3, "pow": 4, "num": 5, "name": 5, "call": 5}


def to_latex(node: Node, symbols: dict[str, str]) -> str:
    def wrap(child: Node, min_prec: int) -> str:
        s = to_latex(child, symbols)
        return f"\\left({s}\\right)" if _PREC[child.kind] < min_prec else s

    k = node.kind
    if k == "num":
        v = float(node.value)  # type: ignore[arg-type]
        if v == math.pi:
            return r"\pi"
        return f"{v:g}"
    if k == "name":
        return symbols.get(str(node.value), str(node.value))
    if k == "neg":
        return "-" + wrap(node.args[0], 3)
    if k == "add":
        return f"{to_latex(node.args[0], symbols)} + {wrap(node.args[1], 1)}"
    if k == "sub":
        return f"{to_latex(node.args[0], symbols)} - {wrap(node.args[1], 2)}"
    if k == "mul":
        # hypot(x,y)*hypot(x,y)*hypot(x,y) reads as a cube, not three roots in a row that overflow a phone screen.
        factors: list[Node] = []
        stack = [node]
        while stack:
            n = stack.pop()
            if n.kind == "mul":
                stack.extend(reversed(n.args))
            else:
                factors.append(n)
        groups: list[tuple[Node, int]] = []
        for f in factors:
            if groups and groups[-1][0] == f:
                groups[-1] = (f, groups[-1][1] + 1)
            else:
                groups.append((f, 1))
        if len(groups) < len(factors):
            return "\\,".join(wrap(f, 3 if i else 2) if c == 1 else f"{{{wrap(f, 5)}}}^{{{c}}}"
                               for i, (f, c) in enumerate(groups))
        return f"{wrap(node.args[0], 2)}\\,{wrap(node.args[1], 3)}"
    if k == "div":
        return f"\\frac{{{to_latex(node.args[0], symbols)}}}{{{to_latex(node.args[1], symbols)}}}"
    if k == "pow":
        return f"{{{wrap(node.args[0], 5)}}}^{{{to_latex(node.args[1], symbols)}}}"
    name = str(node.value)
    args = [to_latex(a, symbols) for a in node.args]
    if name == "abs":
        return f"\\left|{args[0]}\\right|"
    if name == "sqrt":
        return f"\\sqrt{{{args[0]}}}"
    if name == "hypot":
        return f"\\sqrt{{{wrap(node.args[0], 5)}^2 + {wrap(node.args[1], 5)}^2}}"
    return f"{FUNCTIONS[name][1]}\\left({', '.join(args)}\\right)"


_PY = {"sin": "np.sin", "cos": "np.cos", "tan": "np.tan", "tanh": "np.tanh", "atan": "np.arctan", "exp": "np.exp",
       "log": "np.log", "sqrt": "np.sqrt", "abs": "np.abs", "sign": "np.sign", "hypot": "np.hypot"}


def to_python(node: Node) -> str:
    """Python source for the exported script, generated from the parsed tree (never from the raw text)."""
    k = node.kind
    if k == "num":
        return repr(float(node.value))  # type: ignore[arg-type]
    if k == "name":
        return str(node.value)
    a = [to_python(x) for x in node.args]
    if k == "neg":
        return f"(-{a[0]})"
    if k in ("add", "sub", "mul", "div", "pow"):
        op = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**"}[k]
        return f"({a[0]} {op} {a[1]})"
    return f"{_PY[str(node.value)]}({', '.join(a)})"
