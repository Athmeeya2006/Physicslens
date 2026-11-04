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


def load_index_rows() -> list:
    sys.path.insert(0, str(ROOT / "backend"))
    from physicslens import prereg
    return prereg.index_rows()


def pendulum_check() -> dict:
    return load("data/pendulum/check.json")


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


def orbit_more(name: str) -> dict:
    return json.loads((ROOT / "data" / "orbit_more" / "results.json").read_text())[name]


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


def spring_longer_gap() -> float:
    """How far viscous damping falls behind the winner by BIC on the spring's longer recording."""
    r = load("gallery/gallery-spring-longer/result.json")
    return next(f["bic"] for f in r["ranking"] if f["model_id"] == "damped_harmonic") - min(f["bic"] for f in r["ranking"])


def glimmer_gap() -> float:
    """How far viscous damping falls behind the winner by BIC in the Glimmer re-analysis of the spring."""
    r = load("data/glimmer_check/spring_longer.json")
    return next(f["bic"] for f in r["ranking"] if f["model_id"] == "damped_harmonic") - min(f["bic"] for f in r["ranking"])


def pendulum_longer_gap() -> float:
    """How far the simple pendulum falls behind by BIC on the pendulum's longer recording."""
    r = load("gallery/gallery-pendulum-longer/result.json")
    return next(f["bic"] for f in r["ranking"] if f["model_id"] == "pendulum") - min(f["bic"] for f in r["ranking"])


CLAIMS = [
    # In 30 seconds and the judging guide
    Claim("Apophis: the Sun's GM, % from JPL", r"the Sun's GM within ([\d.]+)% and Jupiter's",
          lambda: abs(param("apophis", "mu") / GM_SUN - 1) * 100),
    Claim("Jupiter: four moons combined, % from JPL", r"and Jupiter's within ([\d.]+)% of NASA/JPL",
          lambda: abs(jupiter()["combined"]["gm_error_percent"])),
    Claim("Kepler's third law exponent", r"Kepler's third law \(exponent ([\d.]+), 95%",
          lambda: jupiter()["kepler_third_law"]["exponent"]),
    Claim("Kepler's third law exponent, 1 sd", r"distance\^[\d.]+ ± ([\d.]+)\*\*", lambda: jupiter()["kepler_third_law"]["exponent_std"]),
    Claim("Halley: free exponent", r"on Halley's comet a force law with the exponent left free fitted ([\d.]+)",
          lambda: ranked_param("halley", "n")),
    Claim("Pendulum g", r"a pendulum given as 0.5 m long gives g = ([\d.]+) m/s² \(95%", lambda: param("pendulum", "g")),
    Claim("Pendulum g, 95% low", r"a pendulum given as 0.5 m long gives g = [\d.]+ m/s² \(95% interval ([\d.]+) to",
          lambda: param("pendulum", "g", "ci95")[0]),
    Claim("Pendulum g, 95% high", r"a pendulum given as 0.5 m long gives g = [\d.]+ m/s² \(95% interval [\d.]+ to ([\d.]+)\)",
          lambda: param("pendulum", "g", "ci95")[1]),
    Claim("Pendulum g, % above local g", r"\), ([\d.]+)% above the local ([\d.]+) at the filming site",
          lambda: pendulum_check()["measured"]["relative_error"] * 100),
    Claim("Pendulum local g", r"% above the local ([\d.]+) at the filming site", lambda: pendulum_check()["truth"]["g"]),
    Claim("Pendulum: length that explains it, mm", r"\(a ([\d.]+) mm shorter length would explain it all",
          lambda: pendulum_check()["length"]["shorter_by_mm"]),
    Claim("Pendulum check: physical-pendulum correction, mm", r"the physical-pendulum correction is ([\d.]+) mm",
          lambda: pendulum_check()["length"]["physical_pendulum_correction_mm"]),
    Claim("Pendulum check: clock rate needed, % slow", r"would need to run ([\d.]+)% slow to explain the gap",
          lambda: (pendulum_check()["clock"]["needed_to_explain"] - 1) * 100),
    Claim("Pendulum check: stopwatch reading", r"where it reads ([\d.]+) real seconds per video second",
          lambda: pendulum_check()["clock"]["real_s_per_video_s"]),
    Claim("Pendulum check: small-angle bias, %", r"would have been off by only ([\d.]+)% at this",
          lambda: pendulum_check()["amplitude"]["small_angle_bias_relative"] * 100),
    Claim("Pendulum check: length gap, mm", r"What remains is the length: ([\d.]+) mm shorter",
          lambda: pendulum_check()["length"]["shorter_by_mm"]),
    Claim("Kepler's third law exponent, 95% low", r"Kepler's third law \(exponent [\d.]+, 95% interval ([\d.]+) to",
          lambda: jupiter()["kepler_third_law"]["exponent"] - 1.96 * jupiter()["kepler_third_law"]["exponent_std"]),
    Claim("Kepler's third law exponent, 95% high", r"Kepler's third law \(exponent [\d.]+, 95% interval [\d.]+ to ([\d.]+)\)",
          lambda: jupiter()["kepler_third_law"]["exponent"] + 1.96 * jupiter()["kepler_third_law"]["exponent_std"]),
    Claim("WASP-12b summary, ms/yr", r"WASP-12b's shrinking orbit \(([\d.]+) ms per year, 95%",
          lambda: -timing("wasp12b")["period_change_ms_per_year"][0]),
    Claim("WASP-12b summary, 95% low", r"ms per year, 95% interval ([\d.]+) to [\d.]+, against the published",
          lambda: -timing("wasp12b")["period_change_ms_per_year"][0] - 1.96 * timing("wasp12b")["period_change_ms_per_year"][1]),
    Claim("WASP-12b summary, 95% high", r"ms per year, 95% interval [\d.]+ to ([\d.]+), against the published",
          lambda: -timing("wasp12b")["period_change_ms_per_year"][0] + 1.96 * timing("wasp12b")["period_change_ms_per_year"][1]),
    Claim("Io: Jupiter's GM, % from JPL", r"Jupiter's GM from Io's orbit, within ([\d.]+)% of JPL",
          lambda: abs(jupiter()["moons"][0]["gm_error_percent"])),
    Claim("Audit study: clips", r"\| (\d+) clips: real motion", lambda: load("data/audit_study/summary.json")["clips"]),
    Claim("Audit study: real clips with a law, %", r"real motion has a forecasting law (\d+)% of the time",
          lambda: audit("real")),
    Claim("Audit study: generated clips with a law, %", r"AI-generated (\d+)%; no real clip", lambda: audit("generated")),
    Claim("Audit table: real tracked", r"Object tracked by SAM 3\.1 \| (\d+) of \d+", lambda: load("data/audit_study/analysis.json")["sets"]["real"]["tracked"]["k"]),
    Claim("Audit table: wide tracked", r"Object tracked by SAM 3\.1 \| \d+ of \d+ \(\d+%\) \| (\d+) of", lambda: load("data/audit_study/analysis.json")["sets"]["real_wide"]["tracked"]["k"]),
    Claim("Audit table: generated tracked", r"Object tracked by SAM 3\.1 \| \d+ of \d+ \(\d+%\) \| \d+ of \d+ \(\d+%\) \| (\d+) of", lambda: load("data/audit_study/analysis.json")["sets"]["generated"]["tracked"]["k"]),
    Claim("Audit table: real laws", r"among tracked clips \| \*\*(\d+) of", lambda: load("data/audit_study/analysis.json")["sets"]["real"]["verified_law_among_tracked"]["k"]),
    Claim("Audit table: wide laws", r"among tracked clips \| \*\*\d+ of \d+ \(\d+%\)\*\* \| (\d+) of", lambda: load("data/audit_study/analysis.json")["sets"]["real_wide"]["verified_law_among_tracked"]["k"]),
    Claim("Audit table: generated laws", r"among tracked clips \| \*\*\d+ of \d+ \(\d+%\)\*\* \| \d+ of \d+ \(\d+%\) \| \*\*(\d+) of", lambda: load("data/audit_study/analysis.json")["sets"]["generated"]["verified_law_among_tracked"]["k"]),
    Claim("Audit table: real median error", r"share of the motion's spread\) \| \*\*([\d.]+)\*\*", lambda: load("data/audit_study/analysis.json")["sets"]["real"]["heldout_error_median"]),
    Claim("Audit table: generated median error", r"share of the motion's spread\) \| \*\*[\d.]+\*\* \| [\d.]+ \| \*\*([\d.]+)\*\*", lambda: load("data/audit_study/analysis.json")["sets"]["generated"]["heldout_error_median"]),
    Claim("Audit table: real judged", r"Accused of breaking physics \| \*\*0 of (\d+)\*\*", lambda: load("data/audit_study/analysis.json")["sets"]["real"]["judged"]["k"]),
    Claim("Audit table: wide judged", r"Accused of breaking physics \| \*\*0 of \d+\*\* \| \*\*0 of (\d+)\*\*", lambda: load("data/audit_study/analysis.json")["sets"]["real_wide"]["judged"]["k"]),
    Claim("Audit: all real judged", r"none of the (\d+) real clips it could judge", lambda: load("data/audit_study/analysis.json")["false_accusation_all_real"]["judged_real"]),
    Claim("Audit: all real upper 95%", r"at most ([\d.]+)% of judged real clips at 95%", lambda: load("data/audit_study/analysis.json")["false_accusation_all_real"]["upper_95"] * 100),
    Claim("Audit table: generated flagged", r"\*\*0 of \d+\*\* \| \*\*0 of \d+\*\* \| (\d+) of \d+", lambda: load("data/audit_study/analysis.json")["sets"]["generated"]["flagged_among_judged"]["k"]),
    Claim("Audit table: generated judged", r"\*\*0 of \d+\*\* \| \*\*0 of \d+\*\* \| \d+ of (\d+)", lambda: load("data/audit_study/analysis.json")["sets"]["generated"]["judged"]["k"]),
    Claim("Audit table: model alone, real", r"alone, from 4 frames, says \"violates physics\" \| (\d+) of", lambda: load("data/audit_study/analysis.json")["sets"]["real"]["model_alone"]["violates_physics"]),
    Claim("Audit table: Muse Spark 1.3 alone, real", r"\(closed\) alone, the same question \| (\d+) of", lambda: load("data/audit_study/analysis.json")["sets"]["real"]["model_alone_meta"]["violates_physics"]),
    Claim("Audit table: model alone, wider real", r"says \"violates physics\" \| \d+ of \d+ \| (\d+) of", lambda: load("data/audit_study/analysis.json")["sets"]["real_wide"]["model_alone"]["violates_physics"]),
    Claim("Audit table: Muse Spark 1.3 alone, wider real", r"the same question \| \d+ of \d+ \| (\d+) of", lambda: load("data/audit_study/analysis.json")["sets"]["real_wide"]["model_alone_meta"]["violates_physics"]),
    Claim("Audit table: model alone, generated", r"says \"violates physics\" \| \d+ of \d+ \|[^|]*\| (\d+) of", lambda: load("data/audit_study/analysis.json")["sets"]["generated"]["model_alone"]["violates_physics"]),
    Claim("Audit: AUC", r"\(AUC ([\d.]+); 0\.5 would mean", lambda: load("data/audit_study/analysis.json")["forecast_error_auc_generated_vs_real"]),
    Claim("Scale-free: real tuning clips flagged at defaults", r"At the physical defaults they flagged (\d+) real tuning clips",
          lambda: len(load("data/audit_study/scale_free.json")["real_tuning_clips_flagged_at_defaults"])),
    Claim("Scale-free: held-out real judged", r"they flagged no real clip \(0 of (\d+) judged\)",
          lambda: load("data/audit_study/scale_free.json")["heldout"]["real"]["old_plus_new"]["flagged_among_judged"]["n"]),
    Claim("Scale-free: held-out rated-wrong flagged", r"no additional rated-wrong clip \((\d+) of \d+, as before\)",
          lambda: load("data/audit_study/scale_free.json")["heldout"]["generated_rated_wrong"]["old_plus_new"]["flagged_among_all"]["k"]),
    Claim("Scale-free: held-out rated-wrong clips", r"no additional rated-wrong clip \(\d+ of (\d+), as before\)",
          lambda: load("data/audit_study/scale_free.json")["heldout"]["generated_rated_wrong"]["clips"]),
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
    Claim("Mercury: Sun's GM % off JPL", r"\| Mercury \| [^|]+ \| ([\d.]+)% \|", lambda: abs(orbit_more("mercury")["gm_error_percent"])),
    Claim("Mars: Sun's GM % off JPL", r"\| Mars \| [^|]+ \| ([\d.]+)% \|", lambda: abs(orbit_more("mars")["gm_error_percent"])),
    Claim("Ceres: Sun's GM % off JPL", r"\| Ceres \| [^|]+ \| ([\d.]+)% \|", lambda: abs(orbit_more("ceres")["gm_error_percent"])),
    Claim("Comet 2P/Encke: Sun's GM % off JPL", r"\| Comet 2P/Encke \| [^|]+ \| ([\d.]+)% \|", lambda: abs(orbit_more("encke")["gm_error_percent"])),
    Claim("Sun bias: late returns", r"The misses all go one way: (\d+) of \d+ came back later",
          lambda: load("data/astro/sun/results.json")["return_bias"]["late"]),
    Claim("Sun bias: sign test p", r"a sign test gives p = ([\d.]+) for chance",
          lambda: load("data/astro/sun/results.json")["return_bias"]["sign_test_p"]),
    Claim("Sun bias: no-delay LOO (d)", r"no delay misses by ([\d.]+) days on average",
          lambda: load("data/astro/sun/results.json")["return_bias"]["loo_mae_days"]["none"]),
    Claim("Sun bias: constant delay (d)", r"a constant ([\d.]+)-day delay by",
          lambda: load("data/astro/sun/results.json")["return_bias"]["constant_days"]),
    Claim("Sun bias: constant LOO (d)", r"-day delay by ([\d.]+), and a delay",
          lambda: load("data/astro/sun/results.json")["return_bias"]["loo_mae_days"]["constant"]),
    Claim("Sun bias: latitude LOO (d)", r"shrinks with latitude by ([\d.]+)\.",
          lambda: load("data/astro/sun/results.json")["return_bias"]["loo_mae_days"]["latitude"]),
    Claim("Agent cases: WASP-12b hand chi-square", r"decaying orbit wins \(held-out chi-square ([\d.]+) per event\)",
          lambda: next(h["heldout_chi2"] for h in load("data/cases_agent/wasp12b.json")["hand_chosen"] if h["law"] == "decay")),
    Claim("Agent cases: Kepler-88 N-body forecast (min)", r"N-body second planet wins \(forecast within (\d+) min\)",
          lambda: next(h["heldout_rms_min"] for h in load("data/cases_agent/kepler88b.json")["hand_chosen"] if h["law"] == "nbody")),
    Claim("Agent cases: Kepler-88 best agent error (%)", r"no law verified \(best forecast error (\d+)%\)",
          lambda: load("data/cases_agent/kepler88b.json")["agent_ranking"][0]["heldout_nrmse"] * 100),
    Claim("Agent cases: fireball alpha-beta held-out", r"held-out ([\d.]+) and [\d.]+ km/s\)",
          lambda: next(h["heldout_rms_kms"] for h in load("data/cases_agent/fireball.json")["hand_chosen"] if h["law"] == "alpha_beta")),
    Claim("Agent cases: fireball single-body held-out", r"held-out [\d.]+ and ([\d.]+) km/s\)",
          lambda: next(h["heldout_rms_kms"] for h in load("data/cases_agent/fireball.json")["hand_chosen"] if h["law"] == "single_body")),
    Claim("Agent cases: fireball agent law (%)", r"fragmenting-drag law forecast within (\d+)%",
          lambda: load("data/cases_agent/fireball.json")["agent_ranking"][0]["heldout_nrmse"] * 100),
    Claim("Agent cases: fireball constant acceleration (%)", r"constant acceleration, (\d+)% \|",
          lambda: next(r["heldout_nrmse"] for r in load("data/cases_agent/fireball.json")["agent_ranking"] if r["title"] == "Constant acceleration") * 100),
    Claim("New fall: smallest 95% region (km2)", r"each 95% region (\d+) to \d+ km²",
          lambda: min(i["hdr95"]["area_km2"] for i in load("predictions/fireballs/wa-20260814/prediction.json")["prediction"]["items"])),
    Claim("New fall: largest 95% region (km2)", r"each 95% region \d+ to (\d+) km²",
          lambda: max(i["hdr95"]["area_km2"] for i in load("predictions/fireballs/wa-20260814/prediction.json")["prediction"]["items"])),
    Claim("New fall: 5 g site latitude", r"median sites from\s+([\d.]+)° N",
          lambda: load("predictions/fireballs/wa-20260814/prediction.json")["prediction"]["items"][0]["median_lat_lon"][0]),
    Claim("New fall: 600 g site latitude", r"W for 5 g to ([\d.]+)° N",
          lambda: load("predictions/fireballs/wa-20260814/prediction.json")["prediction"]["items"][-1]["median_lat_lon"][0]),
    Claim("Robot: identified held-out RMS", r"tracks the real joints to ([\d.]+)° RMS against",
          lambda: load("data/robot/result.json")["heldout"]["identified"]["arm_total"]["rms_deg"]),
    Claim("Robot: default held-out RMS", r"° RMS against ([\d.]+)° for the published model",
          lambda: load("data/robot/result.json")["heldout"]["default"]["arm_total"]["rms_deg"]),
    Claim("Robot: ratio to default", r"for the published model, ([\d.]+) times closer",
          lambda: load("data/robot/result.json")["heldout_comparison"]["default_minus_identified"]["ratio"]),
    Claim("Robot: fair baseline RMS", r"fitted to the training episodes, reaches ([\d.]+)°",
          lambda: load("data/robot/result.json")["heldout"]["baseline_delay"]["arm_total"]["rms_deg"]),
    Claim("Robot: ratio to fair baseline", r"corrected model is ([\d.]+) times closer than that",
          lambda: load("data/robot/result.json")["heldout_comparison"]["fair_baseline_minus_identified"]["ratio"]),
    Claim("Robot: ratio to stronger baseline", r"and ([\d.]+) times closer than the published model with per-joint",
          lambda: load("data/robot/result.json")["heldout_comparison"]["stronger_baseline_minus_identified"]["ratio"]),
    Claim("Robot: episodes", r"takes (\d+) public teleoperation episodes", lambda: load("data/robot/result.json")["data"]["episodes"]),
    Claim("Robot: held-out episodes", r"open-loop in MuJoCo on (\d+) held-out episodes",
          lambda: len(load("data/robot/result.json")["data"]["test_episodes"])),
    Claim("Lab footage: clips measured", r"- (\d+) of the 16 clips gave a measured g", lambda: load("data/lab_footage/results.json")["measured"]),
    Claim("Lab footage: clips total", r"\d+ of the (\d+) clips gave a measured g", lambda: load("data/lab_footage/results.json")["clips"]),
    Claim("Lab footage: median error (%)", r"The median error\s+is ([\d.]+)%", lambda: load("data/lab_footage/results.json")["median_abs_error"] * 100),
    Claim("Lab footage: fixed offset (cm)", r"stated one by ([\d.]+) cm plus", lambda: load("data/lab_footage/results.json")["length_offset_fit"]["fixed_m"] * 100),
    Claim("Lab footage: offset per metre (%)", r"cm plus ([\d.]+)% of the stated length",
          lambda: load("data/lab_footage/results.json")["length_offset_fit"]["per_metre"] * 100),
    Claim("Lab footage: offset scatter (mm)", r"\(([\d.]+) mm scatter about that line",
          lambda: load("data/lab_footage/results.json")["length_offset_fit"]["rms_m"] * 1000),
    Claim("ISS history: files", r"over (\d+) archived NASA trajectory files", lambda: load("data/astro/satellites/history.json")["data"]["distinct_files"]),
    Claim("ISS history: burns found", r"records of each reboost .*?: (\d+) of \d+\s+recorded burns found", lambda: load("data/astro/satellites/history.json")["vs_nasa_records"]["detected"]),
    Claim("ISS history: burns covered", r"\d+ of (\d+)\s+recorded burns found", lambda: load("data/astro/satellites/history.json")["vs_nasa_records"]["burns_covered"]),
    Claim("ISS history: recall low (%)", r"recall 95% interval (\d+)% to", lambda: load("data/astro/satellites/history.json")["vs_nasa_records"]["recall_ci95"][0] * 100),
    Claim("ISS history: false alarms", r"\), (\d+) false alarm in", lambda: load("data/astro/satellites/history.json")["vs_nasa_records"]["false_alarms"]),
    Claim("ISS history: quiet days", r"false alarm in (\d+) quiet days", lambda: load("data/astro/satellites/history.json")["vs_nasa_records"]["quiet_days"]),
    Claim("ISS history: timing median (h)", r"a median\s+(\d+) hours from its recorded time", lambda: load("data/astro/satellites/history.json")["vs_nasa_records"]["timing_error_h_median"]),
    Claim("ISS history: max dv difference", r"within ([\d.]+) m/s of NASA's figure",
          lambda: max(abs(x["difference_ms"]) for x in load("data/astro/satellites/history.json")["dv_vs_nasa"])),
    Claim("Faults: type right", r"the fault type is right in (\d+) of 20", lambda: load("data/health_public/result.json")["bearings"]["faulty"]["type_accuracy"]["k"]),
    Claim("Faults: healthy alarms", r"and (\d+) of 20 healthy recordings raise an alarm", lambda: load("data/health_public/result.json")["bearings"]["healthy_false_alarms"]["k"]),
    Claim("Coast-down: Coulomb", r"two coefficients, ([\d.]+) rad/s²", lambda: load("data/health_public/result.json")["spindown"]["two_term_mean_sd"]["a_rad_s2"][0]),
    Claim("Coast-down: viscous", r"rad/s² and ([\d.]+) per second", lambda: load("data/health_public/result.json")["spindown"]["two_term_mean_sd"]["b_per_s"][0]),
    Claim("Coast-down: published Coulomb", r"match the published\s+([\d.]+) and", lambda: load("data/health_public/result.json")["spindown"]["published_mean_sd"]["a_rad_s2"][0]),
    Claim("Coast-down: smallest change (%)", r"would show a change of ([\d.]+)% in Coulomb drag",
          lambda: load("data/health_public/result.json")["spindown"]["smallest_detectable_relative_change"]["a_rad_s2"] * 100),
    Claim("Robot: MJCF alone", r"without the offsets, reach ([\d.]+)°", lambda: load("data/robot/result.json")["heldout"]["identified_mjcf_only"]["arm_total"]["rms_deg"]),
    Claim("Robot: servo laws alone", r"simulated directly reach ([\d.]+)°", lambda: load("data/robot/result.json")["heldout"]["identified_servo_laws"]["arm_total"]["rms_deg"]),
    Claim("Ephemeris: held-out chi2 low", r"held-out chi-square ([\d.]+) to [\d.]+ per\s+transit", lambda: min(x["heldout"]["chi2_per_transit"] for x in [x for x in load("data/ephemeris/results.json")["planets"] if x.get("status") == "registered"])),
    Claim("Ephemeris: held-out chi2 high", r"held-out chi-square [\d.]+ to ([\d.]+) per\s+transit", lambda: max(x["heldout"]["chi2_per_transit"] for x in [x for x in load("data/ephemeris/results.json")["planets"] if x.get("status") == "registered"])),
    Claim("Ephemeris: shift low (min)", r"timetables are off\s+by (\d+) to", lambda: min(abs(x["at_2027_01_01"]["shift_min"]) for x in [x for x in load("data/ephemeris/results.json")["planets"] if x.get("status") == "registered"])),
    Claim("Ephemeris: shift high (min)", r"timetables are off\s+by \d+ to (\d+) minutes", lambda: max(abs(x["at_2027_01_01"]["shift_min"]) for x in [x for x in load("data/ephemeris/results.json")["planets"] if x.get("status") == "registered"])),
    Claim("Ephemeris: published sigma low", r"grown to (\d+) to [\d,]+ minutes", lambda: min(x["at_2027_01_01"]["published_sigma_min"] for x in [x for x in load("data/ephemeris/results.json")["planets"] if x.get("status") == "registered"])),
    Claim("Ephemeris: published sigma high", r"grown to \d+ to ([\d,]+) minutes", lambda: max(x["at_2027_01_01"]["published_sigma_min"] for x in [x for x in load("data/ephemeris/results.json")["planets"] if x.get("status") == "registered"])),
    Claim("Ephemeris: refreshed sigma low", r"refreshed ones carry ([\d.]+) to", lambda: min(x["at_2027_01_01"]["refreshed_sigma_min"] for x in [x for x in load("data/ephemeris/results.json")["planets"] if x.get("status") == "registered"])),
    Claim("Ephemeris: refreshed sigma high", r"refreshed ones carry [\d.]+ to ([\d.]+) minutes", lambda: max(x["at_2027_01_01"]["refreshed_sigma_min"] for x in [x for x in load("data/ephemeris/results.json")["planets"] if x.get("status") == "registered"])),
    Claim("MIMII: physics fans", r"average area\s+under the curve of ([\d.]+) on fans", lambda: load("data/health_public/result.json")["sounds"]["fan_mean"]["physics_auc"]),
    Claim("MIMII: physics pumps", r"under the curve of [\d.]+ on fans and ([\d.]+) on pumps", lambda: load("data/health_public/result.json")["sounds"]["pump_mean"]["physics_auc"]),
    Claim("MIMII: template fans", r"does better \(([\d.]+)\s+and", lambda: load("data/health_public/result.json")["sounds"]["fan_mean"]["generic_template_auc"]),
    Claim("MIMII: template pumps", r"does better \([\d.]+\s+and ([\d.]+)\)", lambda: load("data/health_public/result.json")["sounds"]["pump_mean"]["generic_template_auc"]),
    Claim("MIMII: paper fans", r"autoencoder reaches ([\d.]+) and", lambda: load("data/health_public/result.json")["sounds"]["fan_mean"]["mimii_paper_autoencoder_auc_6db"]),
    Claim("MIMII: paper pumps", r"autoencoder reaches [\d.]+ and ([\d.]+) on its own split", lambda: load("data/health_public/result.json")["sounds"]["pump_mean"]["mimii_paper_autoencoder_auc_6db"]),
    Claim("Agent cases: coast-down two-term held-out (rpm)", r"Coulomb plus viscous friction wins \(held-out ([\d.]+) rpm",
          lambda: next(h["heldout_rmse_rpm"] for h in load("data/cases_agent/coastdown.json")["hand_chosen"] if h["law"] == "coulomb+viscous")),
    Claim("Agent cases: coast-down Coulomb alone (rpm)", r"Coulomb alone ([\d.]+) rpm\)",
          lambda: next(h["heldout_rmse_rpm"] for h in load("data/cases_agent/coastdown.json")["hand_chosen"] if h["law"] == "coulomb")),
    Claim("Agent cases: coast-down deceleration", r"\(Coulomb alone, ([\d.]+) rad/s²",
          lambda: -load("data/cases_agent/coastdown.json")["parameters"][0]["value"]),
    Claim("Asteroids vs JPL: night rms", r"the ATLAS night's positions sit ([\d.]+) arcsec \(rms\)", lambda: load("data/astro/asteroids/jpl_check.json")["your_night"]["residual_rms_arcsec"]),
    Claim("Asteroids vs JPL: stated sigma", r"inside the ([\d.]+) arcsec the observatory", lambda: load("data/astro/asteroids/jpl_check.json")["your_night"]["stated_sigma_arcsec"]),
    Claim("Asteroids vs JPL: forecast median", r"three\s+nights lands a median ([\d.]+) arcsec from JPL", lambda: load("data/astro/asteroids/jpl_check.json")["forecast_vs_jpl"]["median_arcsec"]),
    Claim("Asteroids vs JPL: positions", r"at the (\d+) positions measured three weeks later", lambda: load("data/astro/asteroids/jpl_check.json")["forecast_vs_jpl"]["n"]),
    Claim("Asteroids vs JPL: observations median", r"of that window sit a median ([\d.]+) arcsec from it", lambda: load("data/astro/asteroids/jpl_check.json")["observations_vs_jpl"]["median_arcsec"]),
    Claim("Spring follow-up: dry friction held-out %", r"Dry friction forecasts the held-out motion\nwithin ([\d.]+)%", lambda: next(f["heldout_nrmse"] for f in load("gallery/gallery-spring-longer/result.json")["ranking"] if f["model_id"] == "oscillator_dry") * 100),
    Claim("Spring follow-up: viscous held-out %", r"viscous damping misses it by ([\d.]+)%", lambda: next(f["heldout_nrmse"] for f in load("gallery/gallery-spring-longer/result.json")["ranking"] if f["model_id"] == "damped_harmonic") * 100),
    Claim("Spring follow-up: viscous BIC behind", r"and falls (\d+) BIC behind", lambda: spring_longer_gap()),
    Claim("Spring follow-up: omega", r"frequency agrees with the first analysis \(([\d.]+) rad/s\)", lambda: next(p["value"] for p in load("gallery/gallery-spring-longer/result.json")["parameters"] if p["name"] == "omega")),
    Claim("Fleet: own calibration", r"own\s+calibration\s+tracks\s+its\s+joints\s+to\s+([\d.]+)°\s+on\s+average", lambda: load("data/robot/fleet.json")["summary"]["own_mean_deg"]),
    Claim("Fleet: borrowed calibration", r"borrowed from another arm to ([\d.]+)°, worse in", lambda: load("data/robot/fleet.json")["summary"]["borrowed_mean_deg"]),
    Claim("Fleet: pairs worse", r"worse\s+in\s+(\d+)\s+of\s+\d+\s+pairs\s+at\s+95%", lambda: load("data/robot/fleet.json")["summary"]["pairs_significantly_worse"]),
    Claim("Fleet: pairs", r"worse\s+in\s+\d+\s+of\s+(\d+)\s+pairs\s+at\s+95%", lambda: load("data/robot/fleet.json")["summary"]["pairs"]),
    Claim("Glimmer check: held-out %", r"dry friction,\nforecasting the held-out motion within ([\d.]+)% \(Muse Spark Open", lambda: next(f["heldout_nrmse"] for f in load("data/glimmer_check/spring_longer.json")["ranking"] if f["model_id"] == "oscillator_dry") * 100),
    Claim("Glimmer check: viscous BIC behind", r"with viscous damping (\d+) BIC behind\. On the gallery", lambda: glimmer_gap()),
    Claim("ZTF: residual rms", r"centroided, lands ([\d.]+) arcsec \(rms\) from JPL", lambda: load("data/astro/asteroids/ztf.json")["residual_rms_arcsec"]),
    Claim("ZTF: WCS rms", r"own plate solutions \(median ([\d.]+) arcsec\)", lambda: load("data/astro/asteroids/ztf.json")["median_wcs_rms_arcsec"]),
    Claim("ZTF: frames measured", r"(\d+) of\s+the \d+ frames found solved", lambda: load("data/astro/asteroids/ztf.json")["frames_measured"]),
    Claim("ZTF: frames found", r"\d+ of\s+the (\d+) frames found solved", lambda: load("data/astro/asteroids/ztf.json")["frames_found"]),
    Claim("Fleet: original arm own, servo replay", r"where the original arm's own calibration reaches\s+([\d.]+)°", lambda: load("data/robot/fleet.json")["matrix"]["lerobot__svla_so100_pickplace"]["lerobot__svla_so100_pickplace"]["rms_deg"]),
    Claim("Calibrate: new arm held-out", r"its\s+own\s+laws\s+track\s+its\s+12\s+held-out\s+episodes\s+to\s+([\d.]+)°", lambda: load("data/robot/fleet/pranavsaroha__so100_medicine_0605/result.json")["heldout"]["identified_servo_laws"]["arm_total"]["rms_deg"]),
    Claim("Calibrate: new arm episodes held out", r"its\s+own\s+laws\s+track\s+its\s+(\d+)\s+held-out", lambda: len(load("data/robot/fleet/pranavsaroha__so100_medicine_0605/result.json")["data"]["test_episodes"])),
    Claim("Gripper: public model mm", r"public model has the hand ([\d.]+) mm off", lambda: load("data/robot/impact.json")["gripper"]["default"]["rms_mm"]["mean"]),
    Claim("Gripper: tuned public mm", r"delay and offsets ([\d.]+) mm, the corrected", lambda: load("data/robot/impact.json")["gripper"]["baseline_delay_offset"]["rms_mm"]["mean"]),
    Claim("Gripper: corrected mm", r"the corrected model ([\d.]+) mm \(", lambda: load("data/robot/impact.json")["gripper"]["identified"]["rms_mm"]["mean"]),
    Claim("Gripper: ratio", r"mm \([\d.]+ to [\d.]+\): ([\d.]+) times closer", lambda: load("data/robot/impact.json")["gripper_ratio_default_over_identified"]["mean"]),
    Claim("Gripper: place corrected", r"puts the hand within ([\d.]+) mm \(median\)", lambda: load("data/robot/impact.json")["gripper"]["identified"]["at_place_mm"]["median"]),
    Claim("Gripper: place public", r"\(median\) against ([\d.]+) mm for the public", lambda: load("data/robot/impact.json")["gripper"]["default"]["at_place_mm"]["median"]),
    Claim("Policy: better than baseline", r"teleoperation in (\d+) of \d+ rollouts \(", lambda: load("data/robot/policy.json")["summary"]["rollouts_own_better_than_baseline"]),
    Claim("Policy: rollouts", r"teleoperation in \d+ of (\d+) rollouts \(", lambda: load("data/robot/policy.json")["summary"]["rollouts"]),
    Claim("Policy: better than borrowed", r"borrowed calibration in (\d+) of\s+\d+\.", lambda: load("data/robot/policy.json")["summary"]["rollouts_own_better_than_borrowed"]),
    Claim("Policy: ratio min", r"\(([\d.]+) to [\d.]+ times closer per arm\)", lambda: min(a["policy"]["baseline"]["mean"] / a["policy"]["own"]["mean"] for a in load("data/robot/policy.json")["arms"])),
    Claim("Policy: ratio max", r"\([\d.]+ to ([\d.]+) times closer per arm\)", lambda: max(a["policy"]["baseline"]["mean"] / a["policy"]["own"]["mean"] for a in load("data/robot/policy.json")["arms"])),
    Claim("Policy-trained: 00ri", r"predicts the other \d+ at ([\d.]+)° and", lambda: next(x["policy_calibration"]["mean"] for x in load("data/robot/policy.json")["calibrated_on_policy"] if x["arm"].startswith("00ri"))),
    Claim("Policy-trained: julyfun", r"predicts the other \d+ at [\d.]+° and ([\d.]+)°", lambda: next(x["policy_calibration"]["mean"] for x in load("data/robot/policy.json")["calibrated_on_policy"] if x["arm"].startswith("julyfun"))),
    Claim("Drift: June repeat", r"calibration costs only ([\d.]+)° extra \(both", lambda: round(max(p["extra_deg"] for p in load("data/robot/drift.json")["pairs"] if p["days_apart"] <= 10 and p["used_on"].endswith(("0605", "0608"))), 2)),
    Claim("Drift: six months min", r"borrowed calibration costs ([\d.]+)° to [\d.]+° extra, worse in all", lambda: min(p["extra_deg"] for p in load("data/robot/drift.json")["pairs"] if p["days_apart"] > 100)),
    Claim("Drift: six months max", r"borrowed calibration costs [\d.]+° to ([\d.]+)° extra, worse in all", lambda: max(p["extra_deg"] for p in load("data/robot/drift.json")["pairs"] if p["days_apart"] > 100)),
    Claim("ALOHA: Menagerie", r"already identified on a real ALOHA\) \| ([\d.]+)°", lambda: load("data/robot/aloha/result.json")["heldout"]["default"]["arm_total"]["rms_deg"]),
    Claim("ALOHA: cups_open corrected", r"by itself: ([\d.]+)° against", lambda: load("data/robot/aloha/lerobot__aloha_static_cups_open/result.json")["heldout"]["identified"]["arm_total"]["rms_deg"]),
    Claim("ALOHA: corrected", r"\*\*Corrected by PhysicsLens\*\* \| \*\*([\d.]+)°", lambda: load("data/robot/aloha/result.json")["heldout"]["identified"]["arm_total"]["rms_deg"]),
    Claim("ALOHA: ratio", r"tracks this ALOHA ([\d.]+) times as closely", lambda: load("data/robot/aloha/result.json")["heldout_comparison"]["default_minus_identified"]["ratio"]),
    Claim("ALOHA: right wrist before", r"right wrist angle from ([\d.]+)° to", lambda: load("data/robot/aloha/result.json")["heldout"]["default"]["joints"]["right/wrist_angle"]["rms_deg"]),
    Claim("ALOHA: right wrist after", r"right wrist angle from [\d.]+° to ([\d.]+)°", lambda: load("data/robot/aloha/result.json")["heldout"]["identified"]["joints"]["right/wrist_angle"]["rms_deg"]),
    Claim("Rocket: bucket edge", r"land a median \*\*([\d.]+) s\*\* from the throttle", lambda: load("data/rocket/result.json")["summary"]["bucket_edge_error_median_s"]),
    Claim("Rocket: bucket launches", r"verified law; on (\d+) it contains the throttle bucket", lambda: load("data/rocket/result.json")["summary"]["bucket_launches"]),
    Claim("Rocket: Isp sea level", r"specific impulse (\d+) s at sea level and \d+ s in vacuum \(median", lambda: load("data/rocket/result.json")["summary"]["isp_sl_median_s"]),
    Claim("Rocket: Isp vacuum", r"specific impulse \d+ s at sea level and (\d+) s in vacuum \(median", lambda: load("data/rocket/result.json")["summary"]["isp_vac_median_s"]),
    Claim("ALOHA transfer battery", r"\| battery \(\d+ episodes\) \| [\d.]+° \| \*\*([\d.]+)°\*\*", lambda: next(t["identified"]["rms_deg"] for t in load("data/robot/aloha/transfer.json")["tasks"] if t["dataset"].endswith("battery"))),
    Claim("ALOHA transfer candy", r"\| candy \(\d+ episodes\) \| [\d.]+° \| \*\*([\d.]+)°\*\*", lambda: next(t["identified"]["rms_deg"] for t in load("data/robot/aloha/transfer.json")["tasks"] if t["dataset"].endswith("candy"))),
    Claim("ALOHA transfer tape", r"\| tape \(\d+ episodes\) \| [\d.]+° \| \*\*([\d.]+)°\*\*", lambda: next(t["identified"]["rms_deg"] for t in load("data/robot/aloha/transfer.json")["tasks"] if t["dataset"].endswith("tape"))),
    Claim("Agent rocket: launches with rocket law", r"the rocket equation with mass loss, written by itself on (\d+) of 5", lambda: sum(1 for f in sorted((ROOT / "data/cases_agent").glob("rocket-*.json")) if any(("ocket" in (x["title"] or "")) for x in json.loads(f.read_text())["agent_ranking"]))),
    Claim("Drone: textbook total", r"\| Textbook law \(what simulators use\) \| [\d.]+ \| [\d.]+ \| [\d.]+ \| ([\d.]+) \|", lambda: load("data/drone/result.json")["heldout_total_rms_n"]["textbook"]),
    Claim("Drone: ours total", r"\(4 numbers\)\*\* \| \*\*[\d.]+\*\* \| \*\*[\d.]+\*\* \| \*\*[\d.]+\*\* \| \*\*([\d.]+)\*\* \|", lambda: load("data/drone/result.json")["heldout_total_rms_n"]["physicslens"]),
    Claim("Drone: neural total", r"plus a neural network \| [\d.]+ \| [\d.]+ \| [\d.]+ \| ([\d.]+) \|", lambda: load("data/drone/result.json")["heldout_total_rms_n"]["neurobem_bem_nn"]),
    Claim("Compensator: real data", r"really sent within ([\d.]+)° RMS", lambda: load("data/robot/impact.json")["compensation"]["real_data"]["rows"]["compensator"]["mean"]),
    Claim("Compensator: delay and offsets", r"against ([\d.]+)°\s+for a fitted delay and offsets", lambda: load("data/robot/impact.json")["compensation"]["real_data"]["rows"]["delay_and_offset"]["mean"]),
    Claim("Compensator: simulation", r"planned path from\s+[\d.]+° to ([\d.]+)°", lambda: load("data/robot/impact.json")["compensation"]["simulation"]["rows"]["compensator"]["mean"]),
    Claim("Pendulum longer: g", r"a longer stretch of the same swing gives ([\d.]+);", lambda: next(p["value"] for p in load("gallery/gallery-pendulum-longer/result.json")["agreed_parameters"] if p["name"] == "g")),
    Claim("Pendulum longer: simple pendulum BIC behind", r"simple pendulum falls (\d+) BIC behind while damping", lambda: pendulum_longer_gap()),
    Claim("Pendulum longer: damping held-out %", r"while damping forecasts the held-out motion within ([\d.]+)%", lambda: next(f["heldout_nrmse"] for f in load("gallery/gallery-pendulum-longer/result.json")["ranking"] if f["model_id"] == "pendulum_linear_damping") * 100),
    Claim("Pendulum longer: simple held-out %", r"\(without it: ([\d.]+)%\)", lambda: next(f["heldout_nrmse"] for f in load("gallery/gallery-pendulum-longer/result.json")["ranking"] if f["model_id"] == "pendulum") * 100),
    Claim("PyOTE: ours at 0.13 s", r"edge errors of ([\d.]+) frames rms against", lambda: load("data/astro/occultation/pyote_compare.json")["rows"][0]["physicslens_rms_frames"]),
    Claim("PyOTE: theirs at 0.13 s", r"frames rms against\s+PyOTE's ([\d.]+) at 0.13 s", lambda: load("data/astro/occultation/pyote_compare.json")["rows"][0]["pyote_rms_frames"]),
    Claim("PyOTE: ours at 1 s", r"and ([\d.]+) against [\d.]+ at 1 s", lambda: load("data/astro/occultation/pyote_compare.json")["rows"][2]["physicslens_rms_frames"]),
    Claim("PyOTE: theirs at 1 s", r"and [\d.]+ against ([\d.]+) at 1 s", lambda: load("data/astro/occultation/pyote_compare.json")["rows"][2]["pyote_rms_frames"]),
    Claim("Summary: predictions registered", r"Where the evidence stands:\*\* .*?; (\w+) blind predictions",
          lambda: len(load_index_rows())),
    Claim("Robot video: median mm", r"through a camera fitted\s+on two training episodes, to a median ([\d.]+) mm", lambda: load("data/robot/video_check.json")["heldout"]["median_mm"]["mean"]),
    Claim("Robot video: median mm low", r"\(95% over episodes ([\d.]+) to [\d.]+ mm\)\. The fit", lambda: load("data/robot/video_check.json")["heldout"]["median_mm"]["lo"]),
    Claim("Robot video: median mm high", r"\(95% over episodes [\d.]+ to ([\d.]+) mm\)\. The fit", lambda: load("data/robot/video_check.json")["heldout"]["median_mm"]["hi"]),
    Claim("Kepler-88 family: starts", r"the fit was repeated from (\d+) starting points spread",
          lambda: timing("kepler88b")["companion_family"]["starts"]),
    Claim("Kepler-88 family: refined", r"with the best (\d+)\s+refined", lambda: timing("kepler88b")["companion_family"]["refined"]),
    Claim("Kepler-88 family: period (d)", r"one basin, refined to ([\d.]+) days",
          lambda: timing("kepler88b")["companion_family"]["Pc_range"][0]),
    Claim("Kepler-88 family: mass", r"refined to [\d.]+ days and (\d+) Earth masses",
          lambda: timing("kepler88b")["companion_family"]["mc_earth_range"][0]),
    Claim("Kepler-88 family: next fit's chi-square gap", r"\(the next is (\d+) chi-square worse\)",
          lambda: sorted(f["chi2"] for f in timing("kepler88b")["companion_family"]["all_fits"])[1]
          - timing("kepler88b")["companion_family"]["all_fits"][0]["chi2"]),
    Claim("Kepler-88 c: period (d)", r"unseen planet at\s+roughly ([\d.]+) days", lambda: timing("kepler88b")["companion"]["Pc"]),
    Claim("Kepler-88 c: period, formal sd", r"formal errors \(± ([\d.]+) days", lambda: timing("kepler88b")["companion"]["Pc_sigma"]),
    Claim("Kepler-88 c: mass (Earths)", r"roughly [\d.]+ days and (\d+) Earth masses", lambda: timing("kepler88b")["companion"]["mc_earth"]),
    Claim("Kepler-88 c: mass, formal sd", r"formal errors \(± [\d.]+ days, ± (\d+) Earth masses", lambda: timing("kepler88b")["companion"]["mc_earth_sigma"]),
    Claim("Kepler-88 c: published period in formal sds", r"the published period is (\d+) of them away",
          lambda: abs(timing("kepler88b")["companion"]["Pc"] - 22.3383) / timing("kepler88b")["companion"]["Pc_sigma"]),
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
    Claim("Winchcombe: finds inside blind 95% regions", r"computed from the simulation alone: (\d+) of the 9 lies inside",
          lambda: fireball()["summary"]["blind_inside_95"]),
    Claim("Winchcombe summary: blind 95%", r"but only (\d+) of 9 inside its blind 95% landing region",
          lambda: fireball()["summary"]["blind_inside_95"]),
    Claim("Winchcombe: farthest find across the track (m)", r"runs within (\d+) m of all \d+ recovered", lambda: fireball()["summary"]["max_across_m"]),
    Claim("Winchcombe: finds matched, release at the end", r"leave where the light ends\) places (\d+) of the 9", lambda: fireball()["summary"]["along_at_end"]),
    Claim("Winchcombe: more matched, release after the flare", r"the other (\d+) fit only if stones also left",
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
