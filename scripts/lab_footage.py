"""How accurate is it on real footage? Many labelled clips, each with an independent truth, and the coverage of the
95% intervals (a 95% interval should contain the truth about 95 times in 100: the real rate is reported).

  python scripts/lab_footage.py fetch                 # Delfys75 (Kaggle, MIT) and two Wikimedia Commons clips
  python scripts/lab_footage.py run --max-usd 4       # PAID: SAM 3.1 + Muse Spark Open on every clip (cached by clip)
  python scripts/lab_footage.py report                # writes data/lab_footage/results.json, no calls

Sources (licences in data/lab_footage/SOURCES.md):
- Delfys75 (Castaneda Garcia et al., CVPR 2025; Kaggle jaswar/physical-parameter-prediction, MIT): pendulums with
  string lengths measured with a tape (45, 90 and 150 cm, plus or minus 0.1 cm), five takes each, and balls dropped
  from 20 cm with known radii (the scale).
- Wikimedia Commons, Mathe-AC Lernwerkstatt Aachen (CC BY-SA 4.0): a 0.6 m pendulum with a ruler in view.

The truth for g is local gravity at the lab (physics/gravity.py), not 9.81. Clips are never committed; only the
numbers are.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.physics.gravity import AACHEN, Site, local_g

CACHE = ROOT / "var" / "lab_footage"
OUT = ROOT / "data" / "lab_footage"
KAGGLE = "jaswar/physical-parameter-prediction"
# The dataset's authors are at TU Delft (the name is Delft + physics); the lab's exact place is not stated. Across
# the Netherlands normal gravity changes by under 0.02%, a tenth of the tightest interval here.
DELFT = Site("Delft, Netherlands (author affiliation)", 52.0, 0.0, "lab not stated; g varies under 0.02% across the country")
COMMONS = {"fadenpendel-60cm": ("https://commons.wikimedia.org/wiki/Special:FilePath/Fadenpendelgeschwindigkeit_(2_grad).webm",
                                0.60, AACHEN)}
# The dataset's damping label for its pendulums (output_selected/parameters.json): mean and spread over the takes.
DELFYS_DAMPING = {"mean": 0.05924, "std": 0.00776, "note": "the dataset's -0.0592 read as the coefficient of the angular velocity"}
# Seconds of each Delfys75 pendulum take to analyse: enough swings for a held-out quarter, few enough frames to stay cheap.
TRIM_S = 10.0


def clips() -> list[dict]:
    """Every clip with its truth, from the dataset's own labels."""
    out = []
    root = CACHE / "delfys75"
    for length_cm in (45, 90, 150):
        for take in range(1, 6):
            v = next(iter(root.rglob(f"pendulum/pendulum_{length_cm}/{take:02d}/video.mp4")), None)
            if v:
                out.append({"id": f"delfys-pendulum-{length_cm}-{take}", "video": v, "kind": "pendulum",
                            "length_m": length_cm / 100, "length_sigma_m": 0.001, "site": DELFT, "source": "Delfys75"})
    for name, (_url, length, site) in COMMONS.items():
        v = CACHE / "commons" / f"{name}.mp4"
        if v.exists():
            out.append({"id": f"commons-{name}", "video": v, "kind": "pendulum", "length_m": length,
                        "length_sigma_m": 0.005, "site": site, "source": "Wikimedia Commons"})
    return out


def fetch() -> None:
    (CACHE / "delfys75").mkdir(parents=True, exist_ok=True)
    if not any((CACHE / "delfys75").rglob("video.mp4")):
        # The archive may already be here (downloaded where the Kaggle login lives and copied over): unpack it.
        zips = list((CACHE / "delfys75").glob("*.zip"))
        if not zips:
            subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE, "-p", str(CACHE / "delfys75")], check=True)
            zips = list((CACHE / "delfys75").glob("*.zip"))
        for z in zips:
            subprocess.run(["unzip", "-q", "-o", str(z), "-d", str(CACHE / "delfys75")], check=True)
    (CACHE / "commons").mkdir(parents=True, exist_ok=True)
    for name, (url, _l, _s) in COMMONS.items():
        webm, mp4 = CACHE / "commons" / f"{name}.webm", CACHE / "commons" / f"{name}.mp4"
        if not mp4.exists():
            subprocess.run(["curl", "-sL", "-A", "PhysicsLens research", "-o", str(webm), url], check=True)
            # SAM rejects WebM: H.264 in MP4, as the upload path does.
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(webm), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-an", str(mp4)], check=True)
    for c in clips():
        print(c["id"], c["video"])


def trimmed(c: dict) -> Path:
    """The first TRIM_S seconds after the first second (the release), re-encoded at 30 fps."""
    out = CACHE / "trimmed" / f"{c['id']}.mp4"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1", "-t", str(TRIM_S), "-i", str(c["video"]), "-vf", "fps=30",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(out)], check=True)
    return out


def run(max_usd: float) -> None:
    os.environ.setdefault("APP_ENV", "dev")
    from physicslens.costguard import CostGuardClient
    from physicslens.pipeline import RunOptions, analyze
    from physicslens.settings import load_settings

    settings = load_settings()
    if settings.mock_models:
        sys.exit("run needs MOCK_MODELS=0: it calls SAM 3.1 and Muse Spark Open for real.")
    client = CostGuardClient(settings)

    def spent() -> float:
        import sqlite3
        with sqlite3.connect(ROOT / "var" / "ledger.db") as conn:
            return float(conn.execute("SELECT COALESCE(SUM(est_cost_usd), 0) FROM spend WHERE status IN "
                                      "('committed', 'failed_billed') AND run_id LIKE 'lab-%'").fetchone()[0])
    start = spent()
    for c in clips():
        d = CACHE / "runs" / c["id"]
        if (d / "result.json").exists():
            continue
        if spent() - start > max_usd:
            print(f"budget: stopping at ${spent() - start:.3f}", flush=True)
            break
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            r = analyze(client, trimmed(c), d, f"lab-{c['id']}-{int(t0)}",
                        RunOptions(audience="dev", pendulum_length_m=c["length_m"], object_hint="the pendulum bob"),
                        lambda m: None)
        except Exception as e:  # noqa: BLE001  one clip's failure is a result too, recorded with its reason
            r = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
        (d / "result.json").write_text(json.dumps(r, indent=1, default=str))
        print(f"{c['id']}: {r['status']} ({time.time() - t0:.0f} s, ${spent() - start:.3f} so far)", flush=True)


def g_of(r: dict) -> dict | None:
    for p in (r.get("parameters") or []) + (r.get("agreed_parameters") or []):
        if p["name"] == "g":
            return p
    return None


def report() -> dict:
    rows = []
    for c in clips():
        f = CACHE / "runs" / c["id"] / "result.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        g = g_of(r)
        truth = local_g(c["site"])
        row = {"clip": c["id"], "source": c["source"], "length_m": c["length_m"], "truth_g": truth, "site": c["site"].name,
               "status": r.get("status"), "reason": r.get("reason"), "law": (r.get("equation") or {}).get("title")}
        if g:
            # The stated length's own error widens what the truth should be compared with (g scales with L).
            lo, hi = g["ci95"]
            rel = c["length_sigma_m"] / c["length_m"]
            row.update(g=g["value"], ci95=[lo, hi], error=g["value"] / truth - 1, inside=bool(lo <= truth <= hi),
                       inside_with_length=bool(lo * (1 - 1.96 * rel) <= truth <= hi * (1 + 1.96 * rel)),
                       # g is the stated length times the measured omega^2: the length that would give the local g. If
                       # one offset fits every length, the stated lengths miss a fixed part (the bob's radius, the pivot).
                       length_offset_m=c["length_m"] * (truth / g["value"] - 1))
        gam = next((f["params"].get("gamma") for f in r.get("ranking", []) if f.get("model_id") == "pendulum_linear_damping"), None)
        if c["source"] == "Delfys75" and gam is not None:
            # The dataset also labels damping (mean -0.059 over its takes, spread 0.008): read as the coefficient of
            # the angular velocity, it is our gamma with the opposite sign. Reported beside g, not scored.
            row["gamma"] = gam
            row["gamma_label"] = DELFYS_DAMPING
        rows.append(row)
    measured = [r for r in rows if "g" in r]
    out = {"generated": "scripts/lab_footage.py", "clips": len(rows), "measured": len(measured),
           "coverage_95": sum(r["inside"] for r in measured) / len(measured) if measured else None,
           "coverage_95_with_length": sum(r["inside_with_length"] for r in measured) / len(measured) if measured else None,
           "median_abs_error": sorted(abs(r["error"]) for r in measured)[len(measured) // 2] if measured else None,
           "length_offset_by_length": {f"{L:g}": [r["length_offset_m"] for r in measured if r["length_m"] == L]
                                       for L in sorted({r["length_m"] for r in measured})},
           "rows": rows}
    if len({r["length_m"] for r in measured}) >= 2:
        # Offset = a + b * L: a fixed part (the bob's radius, the pivot) and a part growing with length (a timing or
        # scale error makes g wrong by a fixed fraction, so the "missing length" grows with L).
        import numpy as np
        L = np.array([r["length_m"] for r in measured])
        off = np.array([r["length_offset_m"] for r in measured])
        b, a = np.polyfit(L, off, 1)
        out["length_offset_fit"] = {"fixed_m": float(a), "per_metre": float(b),
                                    "rms_m": float(np.sqrt(np.mean((off - (a + b * L)) ** 2)))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "fetch":
        fetch()
    elif cmd == "run":
        run(float(sys.argv[sys.argv.index("--max-usd") + 1]))
    else:
        report()
