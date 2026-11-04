"""The flying-machine cases: the rocket integrator reproduces the rocket equation, the standard atmosphere its table,
the throttle bucket lowers thrust only inside its window, and the drone's law selection keeps the simpler law unless
an extra term earns its place."""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load("rocket_demo")
D = _load("drone_demo")


def test_standard_atmosphere_matches_its_table():
    rho, p = R.atmosphere(np.array([0.0, 11000.0, 20000.0]))
    assert abs(rho[0] - 1.225) < 0.001 and abs(p[0] - 101325) < 1
    assert abs(p[1] - 22632) < 5 and abs(p[2] - 5474.9) < 5


def flight(n: int = 1200) -> dict:
    t = 10.0 + np.arange(n) * R.DT
    return {"t": t, "v": np.full(n, 30.0), "h": np.full(n, 90000.0), "sin": np.zeros(n)}  # in vacuum, flying level


def test_ideal_rocket_follows_the_rocket_equation():
    f = flight()
    p = {"c": 3000.0, "tau": 200.0}
    v = R.simulate("ideal", p, f)
    exact = f["v"][0] + p["c"] * np.log((p["tau"] - f["t"][0]) / (p["tau"] - f["t"]))
    assert np.max(np.abs(v - exact)) < 0.05


def test_throttle_bucket_lowers_thrust_only_inside_its_window():
    f = flight()
    full = R.simulate("ideal", {"c": 3000.0, "tau": 200.0}, f)
    b = R.simulate("bucket", {"c": 3000.0, "tau": 200.0, "kb": 0.7, "t1": 40.0, "t2": 60.0}, f)
    a = np.gradient(b, R.DT) / np.gradient(full, R.DT)
    t = f["t"]
    assert abs(np.median(a[(t > 20) & (t < 35)]) - 1.0) < 0.02  # before: full throttle
    assert abs(np.median(a[(t > 45) & (t < 55)]) - 0.7) < 0.05  # inside: 70%


def test_drone_selection_keeps_the_simpler_law_unless_a_term_earns_it():
    assert D.choose(["a", "b", "c"], {"a": 1.0, "b": 0.99, "c": 0.985}) == "a"  # under 2% better: not kept
    assert D.choose(["a", "b", "c"], {"a": 1.0, "b": 0.80, "c": 0.79}) == "b"
    assert D.choose(["a", "b", "c"], {"a": 1.0, "b": 0.80, "c": 0.70}) == "c"
