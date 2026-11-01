"""Check the README's numbers against the files they come from.

Each claim is a pattern in README.md that captures one number as written, and a function that recomputes it from
committed data (gallery results, data/, benchmark/results/) or from the test suite. A number matches when the
recomputed value, rounded to the precision the README writes, is the written number. Anything that does not match,
or a pattern the README no longer contains, fails the check.

  python scripts/readme_numbers.py              # every claim, including the test count (collects the tests)
  python scripts/readme_numbers.py --no-tests   # skip collecting tests (a few seconds faster)

Local and free: no network, no model calls.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
GM_SUN = 2.9591220828559115e-4  # AU^3/day^2, JPL DE440 (as in data/orbit and data/sky)
# How a number is written: digits with optional thousands commas, decimals and an exponent; never a full stop after it.
NUMBER = r"(\d[\d,]*(?:\.\d+)?(?:e-?\d+)?)"
CAPTURES = (r"([\d.]+)", r"([\d.e-]+)", r"([\d,]+)", r"(\d+)")


def strict(pattern: str) -> str:
    for loose in CAPTURES:
        pattern = pattern.replace(loose, NUMBER)
    return pattern


@cache
def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def gallery(name: str) -> dict:
    return load(f"gallery/gallery-{name}/result.json")


def param(name: str, p: str, field: str = "value") -> float:
    return next(x[field] for x in gallery(name)["parameters"] if x["name"] == p)


def ranked(name: str, model: str, field: str = "heldout_nrmse") -> float:
    return next(x[field] for x in gallery(name)["ranking"] if x["model_id"] == model)


def ranked_param(name: str, p: str) -> float:
    return next(x["params"][p] for x in gallery(name)["ranking"] if p in (x.get("params") or {}))


def sky(key: str) -> dict:
    return load("data/sky/validation_results.json")[key]


def jupiter() -> dict:
    return load("data/jupiter/report.json")


def audit(set_name: str) -> float:
    return load("data/audit_study/analysis.json")["sets"][set_name]["verified_law_among_tracked"]["rate"] * 100


def const(module: str, name: str) -> float:
    """A rule's constant from the code the README describes (imported, so a changed rule fails the check)."""
    sys.path.insert(0, str(ROOT / "backend"))
    import importlib
    return float(getattr(importlib.import_module(f"physicslens.physics.{module}"), name))


@cache
def collected(marker: str | None = None) -> int:
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests"]
    if marker:
        cmd += ["-m", marker]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False,
                         env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "backend"), "MOCK_MODELS": "1",
                              "APP_ENV": "dev"}).stdout
    m = re.search(r"(\d+)(?:/\d+)? tests collected", out)
    if not m:
        raise RuntimeError(f"could not count tests: {out[-300:]}")
    return int(m.group(1))


@dataclass(frozen=True)
class Claim:
    what: str
    pattern: str  # one capture group: the number as the README writes it
    compute: Callable[[], float]
    tests: bool = False  # needs the test suite collected


CLAIMS = [
    # In 30 seconds and the judging guide
    Claim("Apophis: the Sun's GM, % from JPL", r"the Sun's GM within ([\d.]+)% and Jupiter's",
          lambda: abs(param("apophis", "mu") / GM_SUN - 1) * 100),
    Claim("Jupiter: four moons combined, % from JPL", r"and Jupiter's within ([\d.]+)% of NASA/JPL",
          lambda: abs(jupiter()["combined"]["gm_error_percent"])),
    Claim("Kepler's third law exponent", r"Kepler's third law \(exponent ([\d.]+) ±",
          lambda: jupiter()["kepler_third_law"]["exponent"]),
    Claim("Kepler's third law exponent, 1 sd", r"Kepler's third law \(exponent [\d.]+ ± ([\d.]+)\)",
          lambda: jupiter()["kepler_third_law"]["exponent_std"]),
    Claim("Halley: free exponent", r"on Halley's comet a force law with the exponent left free fitted ([\d.]+)",
          lambda: ranked_param("halley", "n")),
    Claim("Pendulum g", r"a 0.5 m pendulum gives g = ([\d.]+) ±", lambda: param("pendulum", "g")),
    Claim("Pendulum g, 1 sd", r"a 0.5 m pendulum gives g = [\d.]+ ± ([\d.]+) m/s²", lambda: param("pendulum", "g", "std")),
    Claim("Pendulum g, % from 9.81", r"m/s² \(([\d.]+)% from 9.81\)", lambda: abs(param("pendulum", "g") / 9.81 - 1) * 100),
    Claim("Io: Jupiter's GM, % from JPL", r"Jupiter's GM from Io's orbit, within ([\d.]+)% of JPL",
          lambda: abs(jupiter()["moons"][0]["gm_error_percent"])),
    Claim("Audit study: clips", r"\| (\d+) clips: real motion", lambda: load("data/audit_study/summary.json")["clips"]),
    Claim("Audit study: real clips with a law, %", r"real motion has a forecasting law (\d+)% of the time",
          lambda: audit("real")),
    Claim("Audit study: generated clips with a law, %", r"AI-generated (\d+)%; no real clip", lambda: audit("generated")),
    # The engine against JPL
    Claim("Engine: Earth, Newton, 10 years (AU)", r"largest error ([\d.e-]+) AU \(Earth\)",
          lambda: sky("planets_ten_years_au")["Earth"]),
    Claim("Engine: Jupiter, Newton (AU)", r"AU \(Earth\), ([\d.e-]+) AU \(Jupiter\)",
          lambda: sky("planets_ten_years_au")["Jupiter"]),
    Claim("Engine: Saturn, Newton (AU)", r"AU \(Jupiter\), ([\d.e-]+) AU \(Saturn\) \|",
          lambda: sky("planets_ten_years_au")["Saturn"]),
    Claim("Engine: Earth with relativity (AU)", r"\| ([\d.e-]+) AU \(Earth, about 2 km\)",
          lambda: sky("planets_ten_years_with_relativity_au")["Earth"]),
    Claim("Engine: Jupiter with relativity (AU)", r"about 2 km\), ([\d.e-]+) AU \(Jupiter\)",
          lambda: sky("planets_ten_years_with_relativity_au")["Jupiter"]),
    Claim("Engine: Saturn with relativity (AU)", r"about 2 km\), [\d.e-]+ AU \(Jupiter\), ([\d.e-]+) AU \(Saturn\)",
          lambda: sky("planets_ten_years_with_relativity_au")["Saturn"]),
    Claim("Engine: frames agree (AU)", r"agree to ([\d.e-]+) AU", lambda: sky("frames_agree_au")),
    Claim("Engine: Mercury's relativistic advance", r"\| ([\d.]+) arcseconds per century \(Einstein",
          lambda: sky("mercury_relativity_arcsec_per_century")),
    Claim("Engine: Apophis 2029 distance (km)", r"at ([\d,]+) km \(JPL: [\d,]+ km\)",
          lambda: sky("apophis_2029_from_j2000")["km"]),
    Claim("Engine: Halley 2061, Sun alone (days late)", r"; (\d+) days late with the Sun alone",
          lambda: sky("halley_2061_from_j2000_sun_only")["days"]),
    Claim("Direct track: GM with no noise", r"recovers the Sun's GM to ([\d.e-]+) with no noise",
          lambda: abs(sky("direct_track_apophis_sun_only")["no_noise_gm_rel_error"])),
    Claim("Direct track: GM with 2 px noise, %", r"and to ([\d.]+)% \(plus or minus",
          lambda: abs(sky("direct_track_apophis_sun_only")["two_px_gm_rel_error"]) * 100),
    Claim("Direct track: its spread, %", r"plus or minus ([\d.]+)%\) with tracker noise",
          lambda: sky("direct_track_apophis_sun_only")["two_px_gm_rel_std"] * 100),
    # Validation on real footage
    Claim("Validation: pendulum g", r"length given\): g = ([\d.]+) ± [\d.]+ m/s²", lambda: param("pendulum", "g")),
    Claim("Validation: pendulum g, 1 sd", r"length given\): g = [\d.]+ ± ([\d.]+) m/s²", lambda: param("pendulum", "g", "std")),
    Claim("Validation: pendulum damping beats the exact law (BIC)", r"fits the whole swing (\d+) BIC better",
          lambda: ranked("pendulum", "pendulum", "bic") - ranked("pendulum", "pendulum_linear_damping", "bic")),
    Claim("Validation: pendulum held-out forecast, %", r"forecasts the held-out quarter \(within ([\d.]+)%\)",
          lambda: ranked("pendulum", "pendulum_linear_damping") * 100),
    Claim("Validation: paper bag, free fall misses by %", r"free fall misses the held-out last quarter by (\d+)%",
          lambda: ranked("paperbag", "free_fall") * 100),
    Claim("Validation: paper bag, linear drag within %", r"linear air drag forecasts it within ([\d.]+)%",
          lambda: ranked("paperbag", "linear_drag") * 100),
    Claim("Validation: paper bag, time constant (s)", r"velocity time constant of ([\d.]+) s", lambda: 1 / param("paperbag", "k")),
    Claim("Validation: paper bag, quadratic drag forecast %", r"quadratic drag forecasts within ([\d.]+)%",
          lambda: ranked("paperbag", "quadratic_drag") * 100),
    Claim("Validation: paper bag, quadratic drag BIC", r"but scores (\d+) BIC worse",
          lambda: ranked("paperbag", "quadratic_drag", "bic") - ranked("paperbag", "linear_drag", "bic")),
    Claim("Validation: spring omega", r"ω = ([\d.]+) ± [\d.]+ rad/s", lambda: param("spring", "omega")),
    Claim("Validation: spring omega, 1 sd", r"ω = [\d.]+ ± ([\d.]+) rad/s", lambda: param("spring", "omega", "std")),
    Claim("Validation: spring, dry friction forecast %", r"dry friction forecasts the held-out motion within ([\d.]+)%",
          lambda: ranked("spring", "oscillator_dry") * 100),
    Claim("Validation: spring, viscous misses by %", r"viscous damping misses by ([\d.]+)%",
          lambda: ranked("spring", "damped_harmonic") * 100),
    Claim("Validation: spring, undamped misses by %", r"the undamped model by (\d+)%", lambda: ranked("spring", "harmonic") * 100),
    # The acceptance rules, as the code sets them
    Claim("Rule: training share, %", r"Each model is fitted on the first (\d+)% of the motion",
          lambda: const("fit", "TRAIN_FRACTION") * 100),
    Claim("Rule: held-out limit, %", r"held-out part must be at most (\d+)%", lambda: const("verify", "MAX_HELDOUT_NRMSE") * 100),
    Claim("Rule: hard held-out limit, %", r"or at most (\d+)% when the error is also within",
          lambda: const("verify", "HARD_MAX_HELDOUT_NRMSE") * 100),
    Claim("Rule: noise multiple", r"also within (\d+) times the measured tracking noise",
          lambda: const("verify", "MAX_HELDOUT_NOISE_RATIO")),
    Claim("Rule: decisive BIC", r"within (\d+) BIC, the agent is asked", lambda: const("verify", "DECISIVE_DELTA_BIC")),
    Claim("Rule: noise prescreen, %", r"explains more than (\d+)% of the measured acceleration",
          lambda: const("prescreen", "MAX_ACCEL_R2") * 100),
    Claim("Rule: bootstrap floor", r"\(40 targeted, at least (\d+) under the time limit", lambda: const("uncertainty", "MIN_BOOTSTRAP_REPS")),
    Claim("Rule: identifiability correlation", r"move together \(correlation above ([\d.]+)\)",
          lambda: const("uncertainty", "IDENTIFIABILITY_R")),
    Claim("Rule: identifiability correlation (pipeline)", r"move together \(correlation beyond ([\d.]+)\)",
          lambda: const("uncertainty", "IDENTIFIABILITY_R")),
    Claim("Rule: bootstrap time limit (s)", r"never fewer than \d+ within (\d+) s", lambda: const("uncertainty", "MAX_BOOTSTRAP_SECONDS")),
    # Testing
    Claim("Tests collected", r"The suite \((\d+) tests", lambda: collected(), tests=True),
    Claim("Tests marked slow", r"; (\d+) marked slow", lambda: collected("slow"), tests=True),
]


def written_value(text: str) -> tuple[float, int]:
    """The number as written and its significant digits ("4.1e-6" -> 2, "38,817" -> 5, "0.023" -> 2)."""
    clean = text.replace(",", "")
    mantissa = re.split(r"[eE]", clean)[0].lstrip("-")
    digits = mantissa.replace(".", "").lstrip("0") or "0"
    return float(clean), len(digits)


def matches(text: str, value: float) -> bool:
    written, sig = written_value(text)
    if "." not in text.split("e")[0] and "e" not in text and abs(written) >= 1:
        return round(value) == written  # an integer as written: the value rounded to the unit
    return float(f"{value:.{sig}g}") == written


def check_baselines(readme: str) -> list[str]:
    """The baseline table in the README is the generated one, row for row."""
    generated = (ROOT / "benchmark" / "results" / "BASELINES.md").read_text()
    rows = [line for line in generated.splitlines() if line.startswith("| ") and not line.startswith("| Clip")]
    return [f"baseline row missing or changed: {row[:60]}..." for row in rows if row not in readme]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-tests", action="store_true", help="skip the claims that need the test suite collected")
    a = ap.parse_args()
    readme = README.read_text()
    failures = 0
    for c in CLAIMS:
        if c.tests and a.no_tests:
            continue
        m = re.search(strict(c.pattern), readme)
        if not m:
            print(f"MISSING  {c.what}: pattern not found in README.md")
            failures += 1
            continue
        try:
            value = float(c.compute())
        except (KeyError, StopIteration, FileNotFoundError, RuntimeError) as e:
            print(f"ERROR    {c.what}: {type(e).__name__} {e}")
            failures += 1
            continue
        ok = matches(m.group(1), value)
        failures += not ok
        print(f"{'ok      ' if ok else 'MISMATCH'} {c.what}: README {m.group(1)}, source {value:.6g}")
    for problem in check_baselines(readme):
        print(f"MISMATCH {problem}")
        failures += 1
    print(f"\n{len(CLAIMS)} claims and the baseline table checked: {failures} problem(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
