"""Machine faults from physics on public data, scored against the datasets' own fault labels.

  python scripts/fetch_health_public.py                  (once: downloads into var/health_public/)
  python scripts/health_public_demo.py [bearings] [sounds] [spindown]    (default: all; free, local math only)
  python scripts/health_public_demo.py report                            (re-render REPORT.md only)

Writes data/health_public/result.json (every number below comes from this script) and data/health_public/REPORT.md
(rendered from result.json). Each part states its decision rule and its pass rule in code, before the labels are
read, and the labels are used only to score.

1. Bearings (University of Ottawa UORED-VAFCLS, CC BY 4.0): kinematic defect rates from the documented geometry and
   the measured shaft rate, envelope spectrum, fault type = which defect rate stands out (physics.bearing).
2. Fan and pump sound (MIMII, CC BY-SA 4.0): rotation rate and harmonic structure against each machine's normal
   clips, AUC against the anomaly labels, next to the published autoencoder baselines (physics.machine_sound).
3. Spin-down (open-circuit coast-down of an in-wheel motor, CC BY 4.0): which drag terms of
   I dw/dt = -tau_c - b w - c w^2 the data need, against the authors' published fit (physics.spindown).
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.physics import bearing, machine_sound, spindown

RAW = ROOT / "var" / "health_public"
OUT = ROOT / "data" / "health_public"
RESULT = OUT / "result.json"

# ---------------------------------------------------------------- 1. bearings

# Sehri, Dumond and Bouchard 2023 (Data in Brief 49, 109327): NSK 6203ZZ and FAFNIR 203KD, same dimensions.
UORED_GEOMETRY = bearing.BearingGeometry(n_balls=8, ball_diameter_mm=6.77, pitch_diameter_mm=28.50,
                                         contact_angle_deg=0.0)
UORED_FS = 42000.0
LABEL = {"H": "none", "I": "inner", "O": "outer", "B": "ball", "C": "cage"}
STAGE = {"0": "healthy", "1": "developing", "2": "faulty"}
# Pass rule, fixed before scoring: on the faulty-stage recordings the 4-way fault-type accuracy must have a Wilson
# 95% lower bound above 0.40 (chance is 0.25), and at most 2 of the 20 healthy recordings may raise an alarm.
BEARING_PASS = {"type_accuracy_lower95_above": 0.40, "max_false_alarms_of_20": 2}


def _one_bearing(path: str) -> dict:
    from scipy.io import loadmat
    p = Path(path)
    data = next(v for k, v in loadmat(p).items() if not k.startswith("__"))
    accel = data[:, 0]
    rpm = float(data[0, 2])  # the Hall-sensor speed is stored once, in the first row
    d = bearing.diagnose(accel, UORED_FS, UORED_GEOMETRY, rpm / 60)
    letter, _, stage = p.stem.split("_")
    return {"file": p.name, "label": LABEL[letter], "stage": STAGE[stage], "tach_rpm": rpm,
            "shaft_hz": round(d.shaft_hz, 3), "band_hz": list(d.band_hz),
            "expected_hz": {k: round(v, 2) for k, v in d.expected_hz.items()},
            "snr_db": {k: [round(x, 1) for x in v] for k, v in d.snr_db.items()},
            "present": d.present, "diagnosis": d.fault}


def _rate(k: int, n: int) -> dict:
    lo, hi = bearing.wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None, "wilson95": [round(lo, 3), round(hi, 3)]}


# Post hoc second rule (physics.bearing.BASELINE_RULE): written after the first rule failed. Reported beside the
# headline, never in the status. Synchronous speed of a 4-pole motor on 60 Hz is 30 Hz; 10% slip is the edge.
SHAFT_RANGE_HZ = (27.0, 30.0)


def _load_accel(path: Path) -> tuple[np.ndarray, float]:
    from scipy.io import loadmat
    data = next(v for k, v in loadmat(path).items() if not k.startswith("__"))
    return data[:, 0], float(data[0, 2])


def _one_baseline(args: tuple[str, str]) -> dict:
    test_path, base_path = Path(args[0]), Path(args[1])
    x, rpm = _load_accel(test_path)
    if test_path == base_path:
        # Healthy check: first half is the baseline, second half the test, so the rule cannot see itself.
        half = len(x) // 2
        x, base = x[half:], x[:half]
    else:
        base, _ = _load_accel(base_path)
    d = bearing.diagnose_against_baseline(x, base, UORED_FS, UORED_GEOMETRY, SHAFT_RANGE_HZ, rpm / 60)
    letter, _, stage = test_path.stem.split("_")
    return {"file": test_path.name, "baseline": base_path.name if test_path != base_path else "first half",
            "label": LABEL[letter], "stage": STAGE[stage], "tach_rpm": rpm, "shaft_rpm": round(d.shaft_hz * 60, 1),
            "band_hz": list(d.band_hz), "growth_db": {k: [round(v, 1) for v in g] for k, g in d.growth_db.items()},
            "grown": d.grown, "diagnosis": d.fault}


def run_bearings_baseline() -> dict:
    raw = RAW / "uored"
    jobs = []
    for f in sorted(raw.glob("*.mat")):
        bid = f.stem.split("_")[1]
        jobs.append((str(f), str(raw / f"H_{bid}_0.mat")))
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(_one_baseline, jobs))
    out: dict = {"rule": bearing.BASELINE_RULE, "post_hoc": True, "shaft_range_hz": SHAFT_RANGE_HZ, "recordings": rows}
    out.update(_score(rows))
    gaps = [abs(r["shaft_rpm"] - r["tach_rpm"]) / r["shaft_rpm"] for r in rows]
    out["tach_vs_vibration_rel_gap"] = {"median": round(float(np.median(gaps)), 4), "max": round(max(gaps), 4)}
    out["shaft_rpm_range"] = [min(r["shaft_rpm"] for r in rows), max(r["shaft_rpm"] for r in rows)]
    return out


def _score(rows: list[dict]) -> dict:
    out: dict = {}
    for stage in ("faulty", "developing"):
        sub = [r for r in rows if r["stage"] == stage]
        out[stage] = {
            "type_accuracy": _rate(sum(r["diagnosis"] == r["label"] for r in sub), len(sub)),
            "detected_any_fault": _rate(sum(r["diagnosis"] != "none" for r in sub), len(sub)),
            "per_type": {t: _rate(sum(r["diagnosis"] == t for r in sub if r["label"] == t),
                                  sum(r["label"] == t for r in sub)) for t in bearing.FAULTS},
            "confusion": {t: {p: sum(r["diagnosis"] == p for r in sub if r["label"] == t)
                              for p in (*bearing.FAULTS, "none")} for t in bearing.FAULTS},
        }
    healthy = [r for r in rows if r["stage"] == "healthy"]
    out["healthy_false_alarms"] = _rate(sum(r["diagnosis"] != "none" for r in healthy), len(healthy))
    out["healthy_alarm_types"] = {t: sum(r["diagnosis"] == t for r in healthy) for t in bearing.FAULTS}
    return out


def run_bearings() -> dict:
    files = sorted((RAW / "uored").glob("*.mat"))
    if len(files) != 60:
        raise SystemExit("Expected 60 UORED .mat files; run scripts/fetch_health_public.py bearings first.")
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(_one_bearing, map(str, files)))
    out: dict = {"rule": bearing.DIAGNOSIS_RULE, "pass_rule": BEARING_PASS,
                 "geometry": UORED_GEOMETRY.__dict__, "recordings": rows}
    out.update(_score(rows))
    gap = [abs(r["shaft_hz"] * 60 - r["tach_rpm"]) / r["tach_rpm"] for r in rows]
    out["shaft_rate_vs_tacho_max_rel"] = round(max(gap), 4)
    fa = out["healthy_false_alarms"]["k"]
    lo = out["faulty"]["type_accuracy"]["wilson95"][0]
    out["passed"] = bool(lo > BEARING_PASS["type_accuracy_lower95_above"]
                         and fa <= BEARING_PASS["max_false_alarms_of_20"])
    out["post_hoc_baseline_rule"] = run_bearings_baseline()
    return out


# ---------------------------------------------------------------- 2. fan and pump sound

# Published AUCs at 6 dB SNR, autoencoder baseline, Purohit et al. 2019 (MIMII paper, arXiv:1909.09347, Table 3):
# all anomalous clips plus as many normal clips per machine, trained on the remaining normal clips.
MIMII_PAPER_AUC = {"fan": {"id_00": 0.75, "id_02": 0.99, "id_04": 0.92, "id_06": 0.99},
                   "pump": {"id_00": 0.84, "id_02": 0.45, "id_04": 0.99, "id_06": 0.94}}
# DCASE 2020 Task 2 baseline (autoencoder, development set, single channel, its own noise mix; mean of 10 runs).
# Cited for context only: that dataset is CC BY-NC-SA and is not used here.
DCASE2020_BASELINE_AUC = {"fan": {"id_00": 0.5441, "id_02": 0.7340, "id_04": 0.6161, "id_06": 0.7392},
                          "pump": {"id_00": 0.6715, "id_02": 0.6153, "id_04": 0.8833, "id_06": 0.7455}}


def _wav(path: Path) -> tuple[float, np.ndarray]:
    from scipy.io import wavfile
    fs, x = wavfile.read(path)
    return float(fs), (x[:, 0] if x.ndim == 2 else x).astype(float)  # channel 1 only, as the DCASE task does


def _sound_clip(path: str) -> tuple[machine_sound.Spectrum, np.ndarray]:
    fs, x = _wav(Path(path))
    return machine_sound.spectrum(x, fs), machine_sound.band_levels(x, fs)


def _sound_machine(item: tuple[str, dict]) -> tuple[str, dict]:
    from sklearn.covariance import LedoitWolf
    key, sel = item
    machine, mid = key.split("/")
    load = {g: [_sound_clip(str(RAW / "mimii" / n)) for n in names] for g, names in sel.items()}
    ref_spec = [s for s, _ in load["normal_ref"]]
    f_ref = machine_sound.reference_rotation(ref_spec)
    feats = {g: [machine_sound.clip_features(s, f_ref) for s, _ in v] for g, v in load.items()}
    ref = machine_sound.robust_reference(feats["normal_ref"])
    pos = np.array([machine_sound.anomaly_score(r, ref) for r in feats["anomalous"]])
    neg = np.array([machine_sound.anomaly_score(r, ref) for r in feats["normal_test"]])
    auc = machine_sound.auc_delong(pos, neg)
    # Generic comparison detector, learned from the same 60 normal clips: 64 log band levels, Gaussian
    # (Ledoit-Wolf shrinkage), Mahalanobis distance. Not physics; it shows what a plain template gets here.
    bands = {g: np.array([b for _, b in v]) for g, v in load.items()}
    lw = LedoitWolf().fit(bands["normal_ref"])
    gen = machine_sound.auc_delong(lw.mahalanobis(bands["anomalous"]), lw.mahalanobis(bands["normal_test"]))
    # Post hoc, for explanation only (not used by the score): which physical quantity separates best.
    per_feature = {}
    for k in machine_sound.FEATURES:
        zp = np.array([abs(machine_sound.z_scores(r, ref)[k]) for r in feats["anomalous"]])
        zn = np.array([abs(machine_sound.z_scores(r, ref)[k]) for r in feats["normal_test"]])
        per_feature[k] = round(machine_sound.auc_delong(zp, zn)[0], 3)
    rot_shift = float(np.median([r["rotation_hz"] for r in feats["anomalous"]]) - ref["rotation_hz"][0])
    print(f"sounds: {key} physics AUC {auc[0]:.3f} [{auc[1]:.3f}, {auc[2]:.3f}], generic {gen[0]:.3f}", flush=True)
    return key, {
        "n_anomalous": len(pos), "n_normal_test": len(neg), "n_normal_ref": len(ref_spec),
        "reference_comb_hz": round(f_ref, 2),
        "physics_auc": [round(v, 3) for v in auc], "generic_template_auc": [round(v, 3) for v in gen],
        "mimii_paper_autoencoder_auc_6db": MIMII_PAPER_AUC[machine][mid],
        "dcase2020_baseline_auc": DCASE2020_BASELINE_AUC[machine][mid],
        "post_hoc_feature_auc": per_feature,
        "median_anomalous_rate_shift_hz": round(rot_shift, 3),
    }


def run_sounds() -> dict:
    sel_path = RAW / "mimii" / "selection.json"
    if not sel_path.exists():
        raise SystemExit("MIMII selection missing; run scripts/fetch_health_public.py mimii first.")
    selection = json.loads(sel_path.read_text())
    out: dict = {"features": list(machine_sound.FEATURES),
                 "score": "root mean square of robust z-scores (median, 1.4826 MAD) against 60 normal clips",
                 "machines": {}}
    with ProcessPoolExecutor(max_workers=8) as pool:  # one machine per worker; they share nothing
        out["machines"] = dict(pool.map(_sound_machine, selection.items()))
    for machine in ("fan", "pump"):
        ms = [v for k, v in out["machines"].items() if k.startswith(machine)]
        out[f"{machine}_mean"] = {
            "physics_auc": round(float(np.mean([m["physics_auc"][0] for m in ms])), 3),
            "generic_template_auc": round(float(np.mean([m["generic_template_auc"][0] for m in ms])), 3),
            "mimii_paper_autoencoder_auc_6db": round(float(np.mean([m["mimii_paper_autoencoder_auc_6db"]
                                                                     for m in ms])), 3),
            "dcase2020_baseline_auc": round(float(np.mean([m["dcase2020_baseline_auc"] for m in ms])), 3)}
    return out


# ---------------------------------------------------------------- 3. spin-down

# Long et al. 2026, Zenodo 10.5281/zenodo.23231539 v2, Results.md: mean of five runs (sd across runs).
PUBLISHED_SPIN = {"a_rad_s2": (13.380, 0.380), "b_per_s": (0.0790, 0.0028)}
SPIN_MIN_RPM = 50.0  # below this the encoder wheel and stiction dominate; the authors fit from 50 rpm too


def run_spindown() -> dict:
    import pandas as pd
    path = RAW / "spindown" / "bench_spindown_lifted.csv"
    if not path.exists():
        raise SystemExit("Spin-down record missing; run scripts/fetch_health_public.py spindown first.")
    df = pd.read_csv(path)
    runs = []
    for col in df.columns[1:]:
        keep = df[col] >= SPIN_MIN_RPM
        t, w = df["t_s"][keep].to_numpy(), df[col][keep].to_numpy() * 2 * np.pi / 60
        fits = spindown.fit_spindown(t, w)
        chosen = spindown.select_law(fits)
        rpm = 60 / (2 * np.pi)
        runs.append({"run": col, "points": len(t), "selected": chosen.model, "laws": [
            {"model": f.model, "params": {k: round(v, 5) for k, v in f.params.items()},
             "stderr": {k: round(v, 5) for k, v in f.stderr.items()}, "rmse_rpm": round(f.rmse * rpm, 2),
             "heldout_rmse_rpm": round(f.heldout_rmse * rpm, 2), "bic": round(f.bic, 1)} for f in fits]})
    two = [next(x for x in r["laws"] if x["model"] == "coulomb+viscous") for r in runs]
    a = np.array([x["params"]["a"] for x in two])
    b = np.array([x["params"]["b"] for x in two])
    summary = {"a_rad_s2": [round(a.mean(), 3), round(a.std(ddof=1), 3)],
               "b_per_s": [round(b.mean(), 4), round(b.std(ddof=1), 4)]}
    agrees = all(abs(summary[k][0] - PUBLISHED_SPIN[k][0]) < PUBLISHED_SPIN[k][1] for k in PUBLISHED_SPIN)
    # Repeatability sets the smallest drag change a later coast-down could flag (3 run-to-run sd).
    detectable = {k: round(3 * summary[k][1] / summary[k][0], 3) for k in summary}
    return {"runs": runs, "selected_laws": [r["selected"] for r in runs], "two_term_mean_sd": summary,
            "published_mean_sd": PUBLISHED_SPIN, "agrees_with_published": agrees,
            "smallest_detectable_relative_change": detectable}


# ---------------------------------------------------------------- report

def status(result: dict) -> str:
    b = result.get("bearings", {})
    return "proven" if b.get("passed") else "not proven"


def _fmt(r: dict) -> str:
    return f"{r['k']}/{r['n']} ({r['rate']:.0%}, 95% {r['wilson95'][0]:.0%} to {r['wilson95'][1]:.0%})"


def report(result: dict) -> str:
    L = ["# Machine faults from physics on public data", "",
         "Generated by `scripts/health_public_demo.py` from `result.json`. Sources and licences: `SOURCES.md`.", "",
         (f"Status: **{result['status']}** (rule: bearing fault-type accuracy on the faulty-stage recordings with a "
         "Wilson 95% lower bound above 40%, chance being 25%, and at most 2 of 20 healthy recordings alarmed)."), ""]
    b = result.get("bearings")
    if b:
        L += ["## 1. Bearings: which defect rate stands out (UORED-VAFCLS, 60 recordings)", "",
              f"Rule (fixed before scoring): {b['rule']}", "",
              (f"Shaft rate from the 1x vibration line agrees with the Hall-sensor speed within "
              f"{b['shaft_rate_vs_tacho_max_rel']:.2%} on every recording."), ""]
        for stage in ("faulty", "developing"):
            s = b[stage]
            L += [(f"**{stage.capitalize()} stage (20 bearings, 5 per fault type)**: type correct "
                  f"{_fmt(s['type_accuracy'])}; any fault detected {_fmt(s['detected_any_fault'])}."), "",
                  "| true type | correct | diagnosed as outer / inner / ball / cage / none |", "|---|---|---|"]
            for t in bearing.FAULTS:
                c = s["confusion"][t]
                L.append(f"| {t} | {_fmt(s['per_type'][t])} | {c['outer']} / {c['inner']} / {c['ball']} / "
                         f"{c['cage']} / {c['none']} |")
            L.append("")
        L += [(f"**Healthy recordings, false alarms**: {_fmt(b['healthy_false_alarms'])}; "
              f"alarm types {b['healthy_alarm_types']}."), "",
              f"Pass rule met: **{b['passed']}**.", ""]
        ph = b.get("post_hoc_baseline_rule")
        if ph:
            L += ["### Post hoc second rule (written after the first failed; not used for the status)", "",
                  f"Rule: {ph['rule']}", "",
                  (f"Why it was tried: the Hall-sensor speed stored once per file differs from the measured 1x "
                  f"vibration line by {ph['tach_vs_vibration_rel_gap']['median']:.1%} (median) and up to "
                  f"{ph['tach_vs_vibration_rel_gap']['max']:.0%}, and a healthy recording can already show a defect "
                  "series (the motor's own internal bearings have the same geometry). Thresholds were not changed. "
                  f"Measured shaft speeds span {ph['shaft_rpm_range'][0]} to {ph['shaft_rpm_range'][1]} rpm; the "
                  "lowest values (some cage-bearing recordings, near the bottom of the slip range) suggest the "
                  "strongest line there is not the 1x line, which is not resolved."), ""]
            for stage in ("faulty", "developing"):
                st = ph[stage]
                L.append(f"- {stage}: type correct {_fmt(st['type_accuracy'])}; any fault detected "
                         f"{_fmt(st['detected_any_fault'])}; per type " + ", ".join(
                             f"{t} {st['per_type'][t]['k']}/{st['per_type'][t]['n']}" for t in bearing.FAULTS) + ".")
            L += [f"- healthy (second half against first half): false alarms {_fmt(ph['healthy_false_alarms'])}.",
                  "", ("It removes the false alarms but does not find the fault type either. On this dataset, "
                  "envelope analysis at the kinematic defect rates does not identify the fault type; one possible "
                  "reason, not tested here, is that the run-to-failure damage of degreased bearings is distributed "
                  "wear rather than a single localized pit, and distributed wear has no single impact rate."), ""]
    s = result.get("sounds")
    if s:
        L += ["## 2. Fan and pump sound: rotation and harmonics against the machine's normal (MIMII, 6 dB)", "",
              (f"Score: {s['score']}. Features: {', '.join(s['features'])}. Per machine: 40 anomalous and 40 normal "
              "test clips, 60 normal reference clips (a fixed seeded subset). The published AUCs use all anomalous "
              "clips and a different split, so the comparison is indicative, not like for like."), "",
              ("| machine | physics AUC (95%) | generic template AUC (95%) | MIMII paper AE, 6 dB | DCASE 2020 "
              "baseline | best single physical quantity (post hoc) |"), "|---|---|---|---|---|---|"]
        for k, m in s["machines"].items():
            best = max(m["post_hoc_feature_auc"].items(), key=lambda kv: kv[1])
            pa, ga = m["physics_auc"], m["generic_template_auc"]
            L.append(f"| {k} | {pa[0]:.2f} ({pa[1]:.2f} to {pa[2]:.2f}) | {ga[0]:.2f} ({ga[1]:.2f} to {ga[2]:.2f}) "
                     f"| {m['mimii_paper_autoencoder_auc_6db']:.2f} | {m['dcase2020_baseline_auc']:.2f} | "
                     f"{best[0]} {best[1]:.2f} |")
        for machine in ("fan", "pump"):
            mm = s[f"{machine}_mean"]
            L.append(f"| **{machine} mean** | {mm['physics_auc']:.2f} | {mm['generic_template_auc']:.2f} | "
                     f"{mm['mimii_paper_autoencoder_auc_6db']:.2f} | {mm['dcase2020_baseline_auc']:.2f} | |")
        L.append("")
        ms = s["machines"]
        wins = [k for k, m in ms.items() if m["physics_auc"][0] >= m["generic_template_auc"][0]]
        loses = [k for k, m in ms.items() if m["physics_auc"][2] < m["generic_template_auc"][0]]
        edge = [k for k, m in ms.items() if m["reference_comb_hz"] > 0.97 * machine_sound.F0_RANGE_HZ[1]]
        shifts = ", ".join(f"{k} {m['median_anomalous_rate_shift_hz']:+.2f} Hz" for k, m in ms.items())
        L += ["Reading the table:", "",
              (f"- The physics score matches or beats the generic template on {', '.join(wins) or 'no machine'}; "
              f"the template is better, beyond the physics score's 95% interval, on {', '.join(loses) or 'none'}. "
              "A learned spectral template wins on average here."),
              ("- Where physics helps is the explanation: it says which physical quantity moved. Median shift of the "
              f"tracked comb rate in anomalous clips against normal: {shifts}. For fan/id_00 the rate shift alone "
              "separates the classes (post hoc AUC of rotation_hz in the table), which reads as a speed change, "
              "the kind of fault (voltage change, unbalance under load) the dataset lists for fans."),
              (f"- Limit: on {', '.join(edge) or 'none'} the reference comb sits at the top of the 15 to 100 Hz "
              "search range, so it is a strong tonal comb, not a confirmed rotation rate; those rows rest mostly on "
              "the broadband floors."),
              ("- The template's near-perfect AUCs on this subset are higher than the published autoencoders "
              "(which use all anomalous clips and a different split); both detectors here see clips from the same "
              "recording sessions, so these numbers are likely optimistic for a new installation."), ""]
    p = result.get("spindown")
    if p:
        L += ["## 3. Spin-down law of an in-wheel motor (5 coast-downs, CC BY 4.0)", "",
              (f"Selected law per run (simplest law not beaten on both BIC and held-out forecast by 10%): "
              f"{', '.join(p['selected_laws'])}."), "",
              (f"Coulomb + viscous fit, mean and run-to-run sd: tau_c/I = {p['two_term_mean_sd']['a_rad_s2'][0]} +- "
              f"{p['two_term_mean_sd']['a_rad_s2'][1]} rad/s^2, b/I = {p['two_term_mean_sd']['b_per_s'][0]} +- "
              f"{p['two_term_mean_sd']['b_per_s'][1]} 1/s; published "
              f"{p['published_mean_sd']['a_rad_s2'][0]} +- {p['published_mean_sd']['a_rad_s2'][1]} and "
              f"{p['published_mean_sd']['b_per_s'][0]} +- {p['published_mean_sd']['b_per_s'][1]}. "
              f"Agrees within the published spread: {p['agrees_with_published']}."), "",
              (f"Smallest change a later coast-down could flag (3 run-to-run sd): Coulomb drag "
              f"{p['smallest_detectable_relative_change']['a_rad_s2']:.1%}, viscous drag "
              f"{p['smallest_detectable_relative_change']['b_per_s']:.1%}."), "",
              ("| run | points | Coulomb only: held-out rmse (rpm) | + viscous: tau_c/I, b/I, held-out rmse | "
              "+ quadratic: c/I |"), "|---|---|---|---|---|"]
        for r in p["runs"]:
            c1, c2, c3 = r["laws"]
            L.append(f"| {r['run']} | {r['points']} | {c1['heldout_rmse_rpm']} | {c2['params']['a']:.3f}, "
                     f"{c2['params']['b']:.4f}, {c2['heldout_rmse_rpm']} | {c3['params']['c']:.1e} |")
        L.append("")
    return "\n".join(L) + "\n"


def _finite(v):
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_finite(x) for x in v]
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


if __name__ == "__main__":
    parts = sys.argv[1:] or ["bearings", "sounds", "spindown"]
    result = json.loads(RESULT.read_text()) if RESULT.exists() else {}
    runners = {"bearings": run_bearings, "sounds": run_sounds, "spindown": run_spindown}
    for part in parts:
        if part == "report":  # re-render REPORT.md from result.json without recomputing
            continue
        print(f"running {part}")
        result[part] = runners[part]()
    result["status"] = status(result)
    OUT.mkdir(parents=True, exist_ok=True)
    # NaN is not JSON: a harmonic beyond the spectrum has no level, so it is written as null.
    RESULT.write_text(json.dumps(_finite(result), indent=1, default=float) + "\n")
    (OUT / "REPORT.md").write_text(report(result))
    print(f"status: {result['status']}; wrote {RESULT.relative_to(ROOT)} and REPORT.md")
