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
    """A fitted parameter of the accepted law, or (when laws tie) the value they all agree on."""
    r = gallery(name)
    return next(x[field] for x in r.get("parameters") or r.get("agreed_parameters") or [] if x["name"] == p)


def ranked(name: str, model: str, field: str = "heldout_nrmse") -> float:
    return next(x[field] for x in gallery(name)["ranking"] if x["model_id"] == model)


def ranked_param(name: str, p: str) -> float:
    return next(x["params"][p] for x in gallery(name)["ranking"] if p in (x.get("params") or {}))


def timing(case: str) -> dict:
    return load("data/astro/timing/results.json")[case]


def timing_law(case: str, law: str, field: str) -> float:
    return next(x[field] for x in timing(case)["laws"] if x["law"] == law)


def fireball() -> dict:
    return load("data/astro/fireball/results.json")


def sun() -> dict:
    return load("data/astro/sun/results.json")


def sun_law(law: str) -> dict:
    return next(x for x in sun()["laws"] if x["law"] == law)


def occ() -> dict:
    return load("data/astro/occultation/results.json")


def ast() -> dict:
    return load("data/astro/asteroids/results.json")


def sat() -> dict:
    return load("data/astro/satellites/results.json")


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
def centroid_rows() -> dict[float, dict[str, float]]:
    sys.path.insert(0, str(ROOT / "backend"))
    from physicslens.astro.centroid import accuracy_vs_snr
    return {r["snr"]: r for r in accuracy_vs_snr()}


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
    Claim("Validation: pendulum damping's BIC lead", r"fits the whole swing (\d+) BIC better",
          lambda: ranked("pendulum", "pendulum", "bic") - ranked("pendulum", "pendulum_linear_damping", "bic")),
    Claim("Validation: pendulum damping's held-out forecast, %", r"forecasts the held-out quarter worse \(within ([\d.]+)% against",
          lambda: ranked("pendulum", "pendulum_linear_damping") * 100),
    Claim("Validation: simple pendulum's held-out forecast, %", r"against ([\d.]+)% for the simple pendulum\)",
          lambda: ranked("pendulum", "pendulum") * 100),
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
    # Real sky data: PSF against mask centroids on the same synthetic stars (deterministic, seed 0)
    Claim("Centroid: PSF fit at SNR 20 (px)", r"the error is ([\d.]+) px at a\s+signal-to-noise ratio of 20",
          lambda: centroid_rows()[20.0]["psf_rms_px"]),
    Claim("Centroid: PSF fit at SNR 100 (px)", r"ratio of 20 and ([\d.]+) px at 100", lambda: centroid_rows()[100.0]["psf_rms_px"]),
    Claim("Centroid: mask at SNR 20 (px)", r"tracker gives\) is off by ([\d.]+) px and", lambda: centroid_rows()[20.0]["mask_rms_px"]),
    Claim("Centroid: mask at SNR 100 (px)", r"is off by [\d.]+ px and ([\d.]+) px on the same frames",
          lambda: centroid_rows()[100.0]["mask_rms_px"]),
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
    # Real-world cases on public data
    Claim("WASP-12b: period change (ms/yr)", r"The orbit shrinks by ([\d.]+) ± [\d.]+ ms per year",
          lambda: -timing("wasp12b")["period_change_ms_per_year"][0]),
    Claim("WASP-12b: its 1 sd", r"The orbit shrinks by [\d.]+ ± ([\d.]+) ms per year",
          lambda: timing("wasp12b")["period_change_ms_per_year"][1]),
    Claim("WASP-12b: decay held-out chi-square", r"\(held-out chi-square ([\d.]+) per event; a constant period",
          lambda: timing_law("wasp12b", "decay", "heldout_chi2")),
    Claim("WASP-12b: linear held-out chi-square", r"per event; a constant period ([\d.]+)\)",
          lambda: timing_law("wasp12b", "linear", "heldout_chi2")),
    Claim("Kepler-88: transits measured", r"\(KOI-142\): (\d+) transit times", lambda: timing("kepler88b")["transits_used"]),
    Claim("Kepler-88: TTV amplitude (h)", r"arrive up to ([\d.]+) hours", lambda: timing("kepler88b")["measured_ttv_amplitude_h"]),
    Claim("Kepler-88: TTV period (d)", r"repeating every (\d+) days", lambda: timing("kepler88b")["measured_ttv_period_days"]),
    Claim("Kepler-88: sine's miss (min)", r"a sine misses the later transits by (\d+)",
          lambda: timing_law("kepler88b", "sinusoid", "heldout_rms_min")),
    Claim("Kepler-88: N-body forecast (min)", r"forecasts them to (\d+) min and puts", lambda: timing("kepler88b")["companion"]["heldout_rms_min"]),
    Claim("Kepler-88 c: period (d)", r"unseen planet at\s+([\d.]+) ± [\d.]+ days", lambda: timing("kepler88b")["companion"]["Pc"]),
    Claim("Kepler-88 c: period 1 sd", r"unseen planet at\s+[\d.]+ ± ([\d.]+) days", lambda: timing("kepler88b")["companion"]["Pc_sigma"]),
    Claim("Kepler-88 c: mass (Earths)", r"days and (\d+) ± \d+ Earth masses", lambda: timing("kepler88b")["companion"]["mc_earth"]),
    Claim("Kepler-88 c: mass 1 sd", r"days and \d+ ± (\d+) Earth masses", lambda: timing("kepler88b")["companion"]["mc_earth_sigma"]),
    Claim("Ephemeris rot: planets with uncertainties", r"of ([\d,]+) transiting planets", lambda: timing("rot")["with_uncertainties"]),
    Claim("Ephemeris rot: over 10 min", r"worse than 10 minutes for ([\d,]+)", lambda: timing("rot")["over_10_min"]),
    Claim("Ephemeris rot: over 1 h", r"worse than an hour for\s+([\d,]+)", lambda: timing("rot")["over_1_h"]),
    Claim("Ephemeris rot: bright, over 1 h", r"\((\d+) of those around stars brighter", lambda: timing("rot")["bright_over_1_h"]),
    Claim("Winchcombe: worst orbit pull (sigma)", r"matches every published element within\s+([\d.]+) sigma",
          lambda: max(abs(v) for v in fireball()["orbit"]["pulls"].values())),
    Claim("Winchcombe: perihelion (AU)", r"\(perihelion ([\d.]+) AU against", lambda: fireball()["orbit"]["ours"]["q_au"]),
    Claim("Winchcombe: main mass impact (m/s)", r"main mass hits at (\d+) m/s",
          lambda: fireball()["dark_flight_vs_paper"][0]["ours_impact_ms"]),
    Claim("Winchcombe: main mass minutes", r"main mass hits at \d+ m/s after ([\d.]+) minutes",
          lambda: fireball()["dark_flight_vs_paper"][0]["ours_minutes"]),
    Claim("Winchcombe: farthest find across the track (m)", r"runs within (\d+) m of all ten", lambda: fireball()["summary"]["max_across_m"]),
    Claim("Winchcombe: finds matched, release at the end", r"Along it, (\d+) match stones released", lambda: fireball()["summary"]["along_at_end"]),
    Claim("Winchcombe: more matched, release after the flare", r"luminous end and (\d+) more match",
          lambda: fireball()["summary"]["along_either"] - fireball()["summary"]["along_at_end"]),
    Claim("Sun: differential beats rigid (BIC)", r"wins over a rigid Sun by (\d+) BIC", lambda: sun_law("rigid")["bic"] - sun_law("sin2")["bic"]),
    Claim("Sun: sunspots", r"From (\d+) sunspots, ω", lambda: len(sun()["rates"])),
    Claim("Sun: A", r"ω = ([\d.]+) ± [\d.]+ \+", lambda: sun_law("sin2")["coef"]["A"]),
    Claim("Sun: A sd", r"ω = [\d.]+ ± ([\d.]+) \+", lambda: sun_law("sin2")["sigma"]["A"]),
    Claim("Sun: B", r"\+ \(−([\d.]+) ± [\d.]+\) sin² φ", lambda: -sun_law("sin2")["coef"]["B"]),
    Claim("Sun: B sd", r"\+ \(−[\d.]+ ± ([\d.]+)\) sin² φ", lambda: sun_law("sin2")["sigma"]["B"]),
    Claim("Sun: worst gap to the sunspot profile", r"within ([\d.]+)°/day of the published sunspot profile",
          lambda: max(abs(v) for v in sun()["difference_from_published"]["allen"].values())),
    Claim("Sun: Earth's rate, lowest", r"Carrington longitude for each date: ([\d.]+) to", lambda: min(sun()["earth_rate_deg_day"]["from_sunpy_L0"].values())),
    Claim("Sun: Earth's rate, highest", r"for each date: [\d.]+ to ([\d.]+)°/day", lambda: max(sun()["earth_rate_deg_day"]["from_sunpy_L0"].values())),
    Claim("Sun: regions forecast", r"of (\d+) regions still on the disc", lambda: sun()["returns_summary"]["forecast"]),
    Claim("Sun: regions back", r"end of a period, (\d+) came back", lambda: sun()["returns_summary"]["came_back"]),
    Claim("Sun: median miss (days)", r"east limb a median ([\d.]+) days from the forecast \(at most", lambda: sun()["returns_summary"]["median_error_days"]),
    Claim("Sun: worst miss (days)", r"\(at most ([\d.]+)\)", lambda: sun()["returns_summary"]["max_error_days"]),
    Claim("Sun: NOAA's compared", r"lists, for\s+(\d+) of them", lambda: sun()["returns_summary"]["noaa_compared"]),
    Claim("Sun: NOAA's median miss", r"of them, were ([\d.]+) days off", lambda: sun()["returns_summary"]["noaa_median_error_days"]),
    Claim("Umbriel: radius", r"Radius ([\d.]+) ± [\d.]+ km against the published", lambda: occ()["fits"]["circle"]["params"]["radius"]),
    Claim("Umbriel: radius sd", r"Radius [\d.]+ ± ([\d.]+) km against the published", lambda: occ()["fits"]["circle"]["sigma"]["radius"]),
    Claim("Umbriel: limb roughness", r"comes out at ([\d.]+) km, as in the paper", lambda: occ()["fits"]["circle"]["limb_sigma_km"]),
    Claim("Umbriel: chords", r"fitted to the (\d+) chords", lambda: sum(c["positive"] for c in occ()["chords"])),
    Claim("Umbriel: misses", r"and the (\d+) observers who saw nothing", lambda: sum(not c["positive"] for c in occ()["chords"])),
    Claim("Occultation edges: lowest coverage %", r"error holds (\d+) to \d+% of the time", lambda: min(t["within_1_sigma"] for t in occ()["edge_trials"]) * 100),
    Claim("Occultation edges: highest coverage %", r"error holds \d+ to (\d+)% of the time", lambda: max(t["within_1_sigma"] for t in occ()["edge_trials"]) * 100),
    Claim("YR4: first positions", r"Minor Planet Center observations \((\d+) positions\)", lambda: ast()["n_first"]),
    Claim("YR4: your positions", r"next night \((\d+) positions from ATLAS Chile\)", lambda: ast()["n_yours"]),
    Claim("YR4: before (arcsec)", r"shrinks from ([\d.]+) to\s+[\d.]+ arcsec", lambda: ast()["before"]["uncertainty_ahead"]["major_arcsec"]),
    Claim("YR4: after (arcsec)", r"shrinks from [\d.]+ to\s+([\d.]+) arcsec", lambda: ast()["after"]["uncertainty_ahead"]["major_arcsec"]),
    Claim("YR4: later miss (arcsec)", r"fell ([\d.]+) arcsec from the forecast", lambda: ast()["after"]["forecast_check"]["median_miss_arcsec"]),
    Claim("ISS: one pass, catalogue behind (s)", r"measures the catalogue orbit ([\d.]+) ± [\d.]+ s behind", lambda: sat()["single_pass"]["checks"][0]["time_offset_s"]),
    Claim("ISS: its sd", r"measures the catalogue orbit [\d.]+ ± ([\d.]+) s behind", lambda: sat()["single_pass"]["checks"][0]["time_offset_sigma_s"]),
    Claim("ISS: NASA says (s)", r"where NASA's trajectory says ([\d.]+) s", lambda: sat()["single_pass"]["truth_catalogue_behind_s"]),
    Claim("ISS: km per clock second", r"moves the object about ([\d.]+) km along", lambda: sat()["single_pass"]["checks"][0]["km_per_clock_second"]),
    Claim("ISS: 0.5 m/s found in-track", r"along-track parts of ([\d.]+) and", lambda: sat()["manoeuvres"][1]["impulse"]["dv_ric_m_s"][1]),
    Claim("ISS: 2 m/s found in-track", r"along-track parts of [\d.]+ and ([\d.]+) m/s", lambda: sat()["manoeuvres"][2]["impulse"]["dv_ric_m_s"][1]),
    Claim("Tumble: period", r"\(synthetic: ([\d.]+) ± [\d.]+ s", lambda: sat()["tumble_synthetic"]["period_s"]),
    Claim("Tumble: sd", r"\(synthetic: [\d.]+ ± ([\d.]+) s", lambda: sat()["tumble_synthetic"]["sigma_s"]),
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
    """The baseline tables in the README are the generated ones, row for row."""
    out = []
    for rel in ("benchmark/results/BASELINES.md", "benchmark/llm_alone/LLM_ALONE.md"):
        generated = (ROOT / rel).read_text()
        rows = [line for line in generated.splitlines() if line.startswith("| ") and not line.startswith("| Clip")]
        out += [f"{rel} row missing or changed: {row[:60]}..." for row in rows if row not in readme]
    return out


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
    print(f"\n{len(CLAIMS)} claims and the baseline tables checked: {failures} problem(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
