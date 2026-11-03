"""Scale-free audit checks, tuned on half of the study's clips and scored on the other half.

The four checks (backend/physicslens/physics/audit.py, check_scale_free) compare free fall with itself, so no
length scale is needed: one acceleration through a fall, a parabolic path, no energy gained with gravity taken from
the flight's own curvature, each bounce no higher than the fall into it. Free: it replays every clip's saved
measurements (var/study/runs/<id>/audit_input.npz), no model call.

Fixed before any result was looked at:
- Split: sha256("audit-scale-free-v1:" + clip id); an even first byte puts the clip in the tuning half, odd in the
  held-out half.
- Labels: real footage (Physics-IQ) and generated video (VideoPhy-2). A generated clip breaks physics by people's
  judgement when its human physical-commonsense score is below 4 (VideoPhy-2 counts 4 and 5 as physically sound).
- Thresholds, from the tuning half only: each check's practical size is the smallest on a grid (the physical default
  in audit.py times 1 to 50) that flags no real tuning clip, then 20% larger as a margin. ACCEL_CHANGE is set first,
  since the other checks reuse it.
- Wiring rule: the checks enter the audit's verdict only if, on the held-out half, (a) no more real clips are flagged
  than by the existing audit alone, and (b) more generated clips that people rated physically wrong are flagged.

  python scripts/audit_scale_free.py     # writes data/audit_study/scale_free.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_study import MANIFEST, RESULTS, WORK, rows_of, wilson
from physicslens.physics import audit as A

OUT = ROOT / "data" / "audit_study" / "scale_free.json"
SALT = "audit-scale-free-v1:"
MARGIN = 1.2
# The physical defaults the tuning starts from (audit.py ships the tuned values).
DEFAULTS = {"ACCEL_CHANGE": 0.25, "PATH_RESID": 0.02, "SF_ENERGY_GAIN": 0.15, "BOUNCE_GAIN": 1.1}
GRID = np.geomspace(1.0, 50.0, 80)
# (constant in audit.py, check title it governs); ACCEL_CHANGE first, the others reuse it.
TUNED = (("ACCEL_CHANGE", "Free fall keeps one acceleration"), ("PATH_RESID", "Free-flight path is a parabola"),
         ("SF_ENERGY_GAIN", "Falling objects don't gain energy"), ("BOUNCE_GAIN", "Each bounce no higher than the last"))
WIRING_RULE = ("in the verdict only if, on the held-out half, no more real clips are flagged than by the existing audit "
               "alone, and more generated clips that people rated physically wrong are flagged")


def half(clip_id: str) -> str:
    return "tuning" if hashlib.sha256((SALT + clip_id).encode()).digest()[0] % 2 == 0 else "heldout"


def label(clip: dict[str, Any]) -> str:
    if clip["set"] == "real":
        return "real"
    return "generated_wrong" if clip.get("human_pc") is not None and clip["human_pc"] < 4 else "generated_sound"


def judgeable(a: A.AuditInput) -> tuple[bool, float]:
    """The audit's own preconditions: the scale-free checks run only where the rest of the audit may judge."""
    if len(a.traj) < 3:
        return False, 0.0
    sigma = A.noise_sigma(a.traj.xy)
    return not A.preconditions(a, sigma), sigma


def scale_free_checks(a: A.AuditInput) -> list[A.Check]:
    ok, sigma = judgeable(a)
    if not ok:
        return []
    return A.check_scale_free(a, sigma)


def flags_real(rows: list[dict[str, Any]], title: str) -> bool:
    return any(c.title == title and c.status == "violation"
               for r in rows if r["label"] == "real" for c in scale_free_checks(r["input"]))


def tune(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for name, title in TUNED:
        default = DEFAULTS[name]
        base = 1.0 if name == "BOUNCE_GAIN" else 0.0  # the bounce limit is a ratio whose physical bound is 1
        chosen = None
        for m in GRID:
            setattr(A, name, base + (default - base) * m)
            if not flags_real(rows, title):
                chosen = base + (default - base) * m
                break
        value = base + (chosen - base) * MARGIN if chosen is not None else float("inf")
        if chosen is not None and m == GRID[0]:
            value = default  # the physical default already flags no real tuning clip: keep it, no margin needed
        setattr(A, name, value)
        out[name] = {"default": default, "tuned": value if np.isfinite(value) else None, "title": title,
                     "note": None if np.isfinite(value) else "no grid value left the real tuning clips unflagged: the check never flags"}
    return out


def share(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "lo95": lo, "hi95": hi}


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    groups = {"real": ["real"], "generated": ["generated_wrong", "generated_sound"],
              "generated_rated_wrong": ["generated_wrong"], "generated_rated_sound": ["generated_sound"]}
    for gname, labels in groups.items():
        rs = [r for r in rows if r["label"] in labels]
        g: dict[str, Any] = {"clips": len(rs)}
        for mode in ("old", "old_plus_new"):
            v = Counter(r[mode]["verdict"] for r in rs)
            judged = v["violates_physics"] + v["consistent"]
            g[mode] = {"verdicts": dict(v), "judgeable": share(judged, len(rs)),
                       "flagged_among_judged": share(v["violates_physics"], judged),
                       "flagged_among_all": share(v["violates_physics"], len(rs))}
        per_check: dict[str, Counter] = {}
        for r in rs:
            for c in r["new_checks"]:
                per_check.setdefault(c["title"], Counter())["cant_judge" if c["status"] == "skipped" else c["status"]] += 1
            if not r["new_checks"]:
                per_check.setdefault("(audit preconditions not met)", Counter())["cant_judge"] += 1
        g["new_checks"] = {k: dict(c) for k, c in per_check.items()}
        out[gname] = g
    return out


def published() -> dict[str, Any]:
    """The whole-study numbers already published (old audit), for comparison only."""
    s = json.loads((ROOT / "data" / "audit_study" / "summary.json").read_text())
    real = [x for x in s["sources"] if x["key"] == "real"]
    gen = [x for x in s["sources"] if x["key"] != "real"]
    tot = lambda xs, k: sum(int(x[k]) for x in xs)
    return {"note": "all 149 clips, existing audit, as in the README; not a held-out measurement",
            "real": {"judged": tot(real, "judged"), "flagged": tot(real, "flagged"),
                     "flagged_among_judged": share(tot(real, "flagged"), tot(real, "judged"))},
            "generated": {"judged": tot(gen, "judged"), "flagged": tot(gen, "flagged"),
                          "flagged_among_judged": share(tot(gen, "flagged"), tot(gen, "judged"))}}


def main() -> None:
    clips = {c["id"]: c for c in json.loads(MANIFEST.read_text())["clips"]}
    rows, missing = [], Counter()
    ran = {r["id"] for r in rows_of(RESULTS)}  # the 149 clips the study ran; the rest of the manifest never ran
    for cid, c in sorted((k, v) for k, v in clips.items() if k in ran):
        saved = WORK / "runs" / cid / "audit_input.npz"
        if not saved.exists():
            missing[f"{half(cid)}:{label(c)}"] += 1  # never downloaded, or SAM found no object: no measurement
            continue
        rows.append({"id": cid, "half": half(cid), "label": label(c), "generator": c["generator"],
                     "human_pc": c.get("human_pc"), "input": A.load_input(saved)})
    shipped = {name: getattr(A, name) for name, _ in TUNED}
    shipped_in_verdict = A.SCALE_FREE_IN_VERDICT

    tuning = [r for r in rows if r["half"] == "tuning"]
    heldout = [r for r in rows if r["half"] == "heldout"]
    for name, value in DEFAULTS.items():
        setattr(A, name, value)
    at_default = [{"id": r["id"], "flags": [c.text for c in scale_free_checks(r["input"]) if c.status == "violation"]}
                  for r in tuning if r["label"] == "real"]
    at_default = [x for x in at_default if x["flags"]]  # what the physical defaults would accuse on real tuning clips
    thresholds = tune(tuning)
    for r in rows:
        a = r.pop("input")
        r["old"] = A.audit(a, scale_free=False)
        r["old_plus_new"] = A.audit(a, scale_free=True)
        r["new_checks"] = [{"title": c.title, "status": c.status, "value": c.value, "z": c.z, "text": c.text}
                           for c in scale_free_checks(a)]

    held = score(heldout)
    real_old = held["real"]["old"]["verdicts"].get("violates_physics", 0)
    real_new = held["real"]["old_plus_new"]["verdicts"].get("violates_physics", 0)
    wrong_old = held["generated_rated_wrong"]["old"]["verdicts"].get("violates_physics", 0)
    wrong_new = held["generated_rated_wrong"]["old_plus_new"]["verdicts"].get("violates_physics", 0)
    wire = real_new <= real_old and wrong_new > wrong_old

    changed = [{"id": r["id"], "half": r["half"], "label": r["label"], "generator": r["generator"],
                "human_pc": r["human_pc"], "old": r["old"]["verdict"], "old_plus_new": r["old_plus_new"]["verdict"],
                "new_violations": [c["text"] for c in r["new_checks"] if c["status"] == "violation"]}
               for r in rows if r["old"]["verdict"] != r["old_plus_new"]["verdict"]
               or any(c["status"] == "violation" for c in r["new_checks"])]
    result = {
        "wording": "The audit flags physically impossible motion when it can judge; a clean audit is not proof a clip is real.",
        "split": {"rule": f"sha256('{SALT}' + clip id), even first byte: tuning; odd: held-out",
                  "tuning": dict(Counter(r["label"] for r in tuning)), "heldout": dict(Counter(r["label"] for r in heldout)),
                  "no_saved_measurement": dict(missing)},
        "labels": {"generated_rated_wrong": "VideoPhy-2 human physical-commonsense score below 4",
                   "generated_rated_sound": "score 4 or 5", "real": "Physics-IQ real footage"},
        "threshold_rule": f"smallest of default x {GRID[0]:g} to {GRID[-1]:g} flagging no real tuning clip, times {MARGIN}",
        "thresholds": thresholds, "real_tuning_clips_flagged_at_defaults": at_default,
        "shipped_constants": shipped, "constants_match": all(np.isclose(shipped[k], v["tuned"] if v["tuned"] is not None else np.inf) for k, v in thresholds.items()),
        "wiring_rule": WIRING_RULE,
        "wire_in_verdict": wire, "shipped_in_verdict": shipped_in_verdict,
        "heldout": held, "tuning": score(tuning),
        "published_for_comparison": published(),
        "clips_changed_or_flagged_by_new_checks": changed,
    }
    OUT.write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(json.dumps({k: result[k] for k in ("split", "thresholds", "constants_match", "wire_in_verdict")}, indent=1, default=float))
    for g in ("real", "generated", "generated_rated_wrong"):
        for mode in ("old", "old_plus_new"):
            h = held[g][mode]
            print(f"heldout {g:22} {mode:13} judgeable {h['judgeable']['k']}/{h['judgeable']['n']} "
                  f"flagged {h['flagged_among_judged']['k']}/{h['flagged_among_judged']['n']}")
        print(f"   new checks: {held[g]['new_checks']}")
    for c in changed:
        print(c)


if __name__ == "__main__":
    main()
