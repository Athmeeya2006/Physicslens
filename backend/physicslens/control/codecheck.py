"""Static checks on generated controller code, before it is ever executed.

The code is untrusted. These checks are the first of two walls (the second is the sandboxed subprocess in
harness.py): they reject anything outside a small, explicit subset of Python, and the reason goes back to the
model as a failure report.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

MAX_LINES = 300
MAX_CHARS = 30_000
ALLOWED_MODULES = ("math", "numpy")
CONTRACT = "def control(t: float, q: list[float], qdot: list[float], params: dict[str, float]) -> list[float]"
PARAMS = ("t", "q", "qdot", "params")

BANNED_NAMES = frozenset({
    "eval", "exec", "compile", "__import__", "open", "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "input", "breakpoint",
    # Not in the brief's list, but each one is a way to reach objects or the interpreter a controller never needs.
    "memoryview", "help", "exit", "quit", "dir", "type", "object", "super", "classmethod", "staticmethod",
    "property", "issubclass", "hasattr",
})
# numpy functions that touch files, raw memory or process-wide state; a controller has no use for any of them.
BANNED_ATTRIBUTES = frozenset({
    "load", "save", "savez", "savez_compressed", "loadtxt", "savetxt", "genfromtxt", "fromfile", "tofile", "fromregex",
    "memmap", "ctypes", "ctypeslib", "lib", "testing", "distutils", "f2py", "seterr", "seterrcall", "setbufsize",
    "set_printoptions", "DataSource", "dump", "dumps", "frombuffer", "show_config", "os", "sys",
    # str.format can walk attributes ("{0.__class__}") without any attribute appearing in the syntax tree.
    "format", "format_map",
})
BANNED_NODES: dict[type, str] = {
    ast.ClassDef: "class definitions", ast.AsyncFunctionDef: "async functions", ast.Await: "await",
    ast.Yield: "generators", ast.YieldFrom: "generators", ast.Try: "try blocks", ast.With: "with blocks",
    ast.AsyncWith: "with blocks", ast.AsyncFor: "async loops", ast.Global: "global statements",
    ast.Nonlocal: "nonlocal statements", ast.Raise: "raise", ast.Delete: "del",
}
if hasattr(ast, "TryStar"):
    BANNED_NODES[ast.TryStar] = "try blocks"


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    reason: str | None = None


def _fail(reason: str, node: ast.AST | None = None) -> CheckResult:
    line = getattr(node, "lineno", None)
    return CheckResult(False, f"line {line}: {reason}" if line else reason)


def _is_simple_constant(node: ast.AST) -> bool:
    """Literals, arithmetic on them, earlier constants and module constants such as math.pi."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp):
        return _is_simple_constant(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_simple_constant(node.left) and _is_simple_constant(node.right)
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name)
    if isinstance(node, ast.List | ast.Tuple):
        return all(_is_simple_constant(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is not None and _is_simple_constant(k) for k in node.keys) and all(
            _is_simple_constant(v) for v in node.values)
    return False


def _check_import(node: ast.Import | ast.ImportFrom) -> CheckResult | None:
    if isinstance(node, ast.ImportFrom):
        if node.level or node.module not in ALLOWED_MODULES:
            return _fail(f"import from {node.module!r} is not allowed; only {', '.join(ALLOWED_MODULES)}", node)
        for alias in node.names:
            if alias.name == "*":
                return _fail("star imports are not allowed", node)
            if alias.name in BANNED_ATTRIBUTES or alias.name in BANNED_NAMES or alias.name.startswith("_"):
                return _fail(f"importing {alias.name!r} is not allowed", node)
        return None
    for alias in node.names:
        if alias.name not in ALLOWED_MODULES:
            return _fail(f"import {alias.name!r} is not allowed; only {', '.join(ALLOWED_MODULES)}", node)
    return None


def check_controller(code: str) -> CheckResult:
    if len(code) > MAX_CHARS:
        return _fail(f"the code is longer than {MAX_CHARS} characters")
    lines = code.count("\n") + 1
    if lines > MAX_LINES:
        return _fail(f"the code has {lines} lines; the limit is {MAX_LINES}")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return CheckResult(False, f"line {e.lineno}: syntax error: {e.msg}")

    functions = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Import | ast.ImportFrom):
            if (bad := _check_import(stmt)) is not None:
                return bad
        elif isinstance(stmt, ast.FunctionDef):
            functions.append(stmt)
        elif isinstance(stmt, ast.Assign | ast.AnnAssign):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            if not all(isinstance(t, ast.Name) for t in targets) or stmt.value is None or not _is_simple_constant(stmt.value):
                return _fail("module level allows only simple constants (NAME = literal)", stmt)
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue  # a docstring
        else:
            return _fail(f"{type(stmt).__name__} statements are not allowed at module level", stmt)

    if [f.name for f in functions] != ["control"]:
        return _fail(f"define exactly one top-level function named control ({CONTRACT}); found "
                     f"{[f.name for f in functions] or 'none'}")
    fn = functions[0]
    a = fn.args
    names = [p.arg for p in a.args]
    if names != list(PARAMS) or a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs or a.defaults:
        return _fail(f"control must take exactly (t, q, qdot, params): {CONTRACT}", fn)
    if fn.decorator_list:
        return _fail("decorators are not allowed", fn)

    for node in ast.walk(tree):
        for cls, what in BANNED_NODES.items():
            if isinstance(node, cls):
                return _fail(f"{what} are not allowed", node)
        if isinstance(node, ast.Import | ast.ImportFrom) and node not in tree.body and (bad := _check_import(node)):
            return bad
        if isinstance(node, ast.Name):
            if node.id.startswith("_"):
                return _fail(f"names starting with an underscore are not allowed ({node.id})", node)
            if node.id in BANNED_NAMES:
                return _fail(f"{node.id} is not allowed", node)
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                return _fail(f"attributes starting with an underscore are not allowed (.{node.attr})", node)
            if node.attr in BANNED_ATTRIBUTES:
                return _fail(f".{node.attr} is not allowed", node)
            if isinstance(node.ctx, ast.Store | ast.Del):
                # Assigning attributes could rewrite numpy itself, which the harness also uses to score the run.
                return _fail("assigning to attributes is not allowed; keep state in a module-level dict", node)
        elif isinstance(node, ast.arg) and node.arg.startswith("_"):
            return _fail(f"names starting with an underscore are not allowed ({node.arg})", node)
        elif isinstance(node, ast.FunctionDef | ast.Lambda) and node is not fn:
            if isinstance(node, ast.FunctionDef) and (node.name.startswith("_") or node.name in BANNED_NAMES):
                return _fail(f"function name {node.name!r} is not allowed", node)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            return _fail("strings containing a double underscore are not allowed", node)
    return CheckResult(True)
