"""Physics audit at scale: how often does AI-generated video break physics, next to real footage of the same kinds of motion?

Two public sets, analyzed by the same pipeline with the same settings:
- generated: VideoPhy-2 test set (MIT licence), clips from seven video generators, each rated 1 to 5 by people for
  physical commonsense ("pc"). Only captions describing motion PhysicsLens can follow are used (drops, throws,
  bounces, swings, rolls, slides).
- real: Physics-IQ (CC BY 4.0), real footage filmed with fixed cameras, from its public bucket.

  python scripts/audit_study.py select --generated 400 --real 100     # writes data/audit_study/manifest.json (free)
  python scripts/audit_study.py run --limit 20 --workers 3 --max-usd 5     # SPENDS MONEY: SAM 3.1 + Muse Spark Open
  python scripts/audit_study.py baseline --max-usd 2                  # SPENDS MONEY: the model alone, from 4 frames
  python scripts/audit_study.py report                                # free: tables from what has run
  python scripts/audit_study.py replay                                # free: today's audit rules on stored runs

Runs use the full pipeline a judge gets by default (--profile light: one hypothesis sample, one tool step, no
referee, recorded in each result). Clips are downloaded into var/study (not tracked); the manifest and the per-clip
verdicts are tracked so anyone can check them. A clip already run is never re-run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import sqlite3
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import zip_longest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "audit_study"
WORK = ROOT / "var" / "study"
MANIFEST = DATA / "manifest.json"
RESULTS = WORK / "results.jsonl"
BASELINE = WORK / "baseline.jsonl"

VIDEOPHY2_CSV = "https://huggingface.co/datasets/videophysics/videophy2_test/resolve/main/videophy2_test.csv"
PIQ_LIST = "https://storage.googleapis.com/storage/v1/b/physics-iq-benchmark/o?prefix={prefix}&maxResults=1000"
PIQ_FILE = "https://storage.googleapis.com/physics-iq-benchmark/{name}"
MOTION = re.compile(r"\b(drop(s|ped)?|falls?|falling|thrown|throws?|tossed|tosses|bounc\w*|swing\w*|pendulum|rolls?|"
                    r"rolling|launch\w*|projectile|slides?|sliding)\b", re.I)
# Physics-IQ scenes with one object in free fall, rolling, swinging or colliding: the motion the audit can measure.
PIQ_SCENES = ("ball-and-block-fall", "ball-hits-nothing", "ball-rolls-off", "ball-ramp", "mirror-ball-fall",
              "single-cradle", "ball-in-sand", "balls-collide", "ball-rolls-on-glass", "roll-front-box",
              "solid-ball-peakaboo", "duck-falls-in-box", "two-balls-pass", "ball-hits-duck", "roll-behind-box",
              "roll-in-box", "ball-train", "double-cradle")
MAX_SECONDS = 10.0
# "full" is exactly what a judge's run gets (three hypothesis samples, agent tool steps, the referee); "light" makes a
# clip cheap and says so in its decision trail.
PROFILES = {"full": {}, "light": {"hypothesis_samples": 1, "max_tool_steps": 1, "referee": False}}


def fetch(url: str, timeout: float = 120, attempts: int = 4) -> bytes:
    """Retries with a growing pause: a network or DNS blip once failed hundreds of downloads in a row."""
    for k in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except OSError:
            if k == attempts - 1:
                raise
            time.sleep(30 * (k + 1))
    raise AssertionError("unreachable")


def quote_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    # "+" stays as it is: the dataset's links use it for spaces in the stored file names (Sora's clips 403 otherwise).
    return urllib.parse.urlunsplit(parts._replace(path=urllib.parse.quote(parts.path, safe="/+")))


def select(n_generated: int, n_real: int, seed: int) -> None:
    rng = random.Random(seed)
    rows = list(csv.DictReader(io.StringIO(fetch(VIDEOPHY2_CSV).decode())))
    seen: set[str] = set()
    by_gen: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        if r["video_url"] and r["video_url"] not in seen and MOTION.search(r["caption"]) and r["pc"]:
            seen.add(r["video_url"])
            by_gen[r["model_name"]].append(r)
    gens = sorted(by_gen)
    # Even across generators, and within a generator spread over the human scores, so both questions have data.
    per = {g: 0 for g in gens}
    left = n_generated
    while left > 0 and any(per[g] < len(by_gen[g]) for g in gens):
        for g in gens:
            if left and per[g] < len(by_gen[g]):
                per[g] += 1
                left -= 1
    clips: list[dict[str, Any]] = []
    for g in gens:
        by_pc: dict[str, list[dict[str, str]]] = defaultdict(list)
        for r in by_gen[g]:
            by_pc[r["pc"]].append(r)
        for pool in by_pc.values():
            rng.shuffle(pool)
        picked: list[dict[str, str]] = []
        while len(picked) < per[g]:
            for pc in sorted(by_pc):
                if by_pc[pc] and len(picked) < per[g]:
                    picked.append(by_pc[pc].pop())
        for r in picked:
            clips.append({"id": f"vp2-{g}-{hashlib.sha1(r['video_url'].encode()).hexdigest()[:8]}", "set": "generated",
                          "source": "VideoPhy-2 test", "license": "MIT", "generator": g, "url": quote_url(r["video_url"]),
                          "caption": r["caption"], "human_pc": int(r["pc"]), "human_sa": int(r["sa"]) if r["sa"] else None,
                          "is_hard": r["is_hard"] == "1"})
    names: list[str] = []
    for take in ("take-1", "take-2"):
        listing = json.loads(fetch(PIQ_LIST.format(prefix=f"full-videos/{take}/30FPS/")))
        names += [o["name"] for o in listing.get("items", [])]
    by_scene: dict[str, list[str]] = defaultdict(list)
    for n in sorted(names):
        scene = re.sub(r"^.*trimmed-", "", n).rsplit(".", 1)[0]
        if scene in PIQ_SCENES:
            by_scene[scene].append(n)
    real: list[str] = []
    while len(real) < n_real and any(by_scene.values()):
        for s in PIQ_SCENES:
            if by_scene.get(s) and len(real) < n_real:
                real.append(by_scene[s].pop(0))
    for n in real:
        clips.append({"id": "piq-" + Path(n).stem.split("_")[0] + "-" + ("t1" if "take-1" in n else "t2") + "-"
                      + re.search(r"perspective-([a-z]+)", n).group(1), "set": "real", "source": "Physics-IQ",
                      "license": "CC BY 4.0 (Google DeepMind)", "generator": "real", "url": PIQ_FILE.format(name=n),
                      "caption": re.sub(r"^.*trimmed-", "", n).rsplit(".", 1)[0].replace("-", " "),
                      "human_pc": None, "human_sa": None, "is_hard": False})
    DATA.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"seed": seed, "clips": clips}, indent=1) + "\n")
    print(f"{len(clips)} clips: {Counter(c['generator'] for c in clips)}")


def prepare(clip: dict[str, Any]) -> Path:
    """Download once, then a clean H.264 copy of at most MAX_SECONDS (generators write odd codecs and lengths)."""
    raw = WORK / "raw" / (clip["id"] + ".mp4")
    out = WORK / "clips" / (clip["id"] + ".mp4")
    if out.exists():
        return out
    raw.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        raw.write_bytes(fetch(clip["url"]))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-t", str(MAX_SECONDS), "-an", "-c:v", "libx264",
                    "-threads", os.environ.get("FFMPEG_THREADS", "2"),
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
    return out


def spent(run_id: str | None = None) -> float:
    with sqlite3.connect(ROOT / "var" / "ledger.db") as conn:
        q = "SELECT COALESCE(SUM(est_cost_usd), 0) FROM spend WHERE status IN ('committed', 'failed_billed')"
        return float(conn.execute(q + (" AND run_id = ?" if run_id else ""), (run_id,) if run_id else ()).fetchone()[0])


def run_one(clip: dict[str, Any], profile: str = "full") -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "backend"))
    from physicslens.costguard import CostGuardClient
    from physicslens.pipeline import RunOptions, analyze
    from physicslens.settings import load_settings

    t0 = time.time()
    row: dict[str, Any] = {"id": clip["id"], "set": clip["set"], "generator": clip["generator"],
                           "human_pc": clip["human_pc"], "profile": profile}
    try:
        video = prepare(clip)
        client = CostGuardClient(load_settings())
        client.limits["agent"] = {**(client.limits.get("agent") or {}), **PROFILES[profile]}
        # One name per attempt: the per-run cap is for one analysis, not every retry of the same clip added up.
        run_id = f"study-{clip['id']}-{int(time.time())}"
        run_dir = WORK / "runs" / clip["id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        result = analyze(client, video, run_dir, run_id, RunOptions(audience="dev"))
        (run_dir / "result.json").write_text(json.dumps(result, indent=1))
        audit = result.get("audit") or {}
        row.update({
            "status": result.get("status"), "family": result.get("family"),
            "equation": (result.get("equation") or {}).get("title"),
            "audit": audit.get("verdict"), "headline": audit.get("headline"),
            "violations": sorted({v["check"] for v in audit.get("violations", [])}),
            "providers": sorted({f"{c['provider']}:{c['model']}" for c in result.get("llm_calls") or []}),
            "spend_usd": round(spent(run_id), 5),
        })
    except Exception as e:  # one clip failing must not stop the study; the failure is a row of its own
        row.update({"status": "error", "audit": "error", "error": f"{type(e).__name__}: {e}"[:400]})
    row["seconds"] = round(time.time() - t0, 1)
    return row


def rows_of(path: Path) -> list[dict[str, Any]]:
    """Every row, the latest per clip winning (a clip retried after a refused budget appears twice)."""
    latest: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                latest[r["id"]] = r
    return list(latest.values())


def done_ids(path: Path, retry_errors: bool = False) -> set[str]:
    # A clip refused for budget (a daily cap reached mid-batch) is not a result: it always runs again.
    def retry(err: str) -> bool:  # refused for budget, or the network was down: not a result, run it again
        err = err.lower()
        return any(k in err for k in ("busy", "budget", "urlopen error", "name or service", "timed out", "connection",
                                      "open-weight model failed"))
    return {r["id"] for r in rows_of(path)
            if not (retry(r.get("error") or "") or (retry_errors and r.get("status") == "error"))}


def run(limit: int, workers: int, max_usd: float, only: str | None, profile: str, retry_errors: bool = False,
        generator: str | None = None) -> None:
    clips = json.loads(MANIFEST.read_text())["clips"]
    done = done_ids(RESULTS, retry_errors)
    todo = [c for c in clips if c["id"] not in done and (only is None or c["set"] == only)
            and (generator is None or c["generator"] == generator)]
    # Interleave the sets, and the generators within the generated set, so that stopping anywhere (a pilot, a
    # spending limit) leaves a balanced sample rather than one generator's clips.
    by_gen: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in todo:
        if c["set"] == "generated":
            by_gen[c["generator"]].append(c)
    gen = [c for group in zip_longest(*by_gen.values()) for c in group if c is not None]
    real = [c for c in todo if c["set"] == "real"]
    order = [c for pair in zip(gen, real) for c in pair] + gen[len(real):] + real[len(gen):]
    order = order[:limit]
    start = spent()
    est = 0.25 if profile == "full" else 0.15  # per clip, for the in-flight allowance; the ledger has the real figure afterwards
    WORK.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, RESULTS.open("a") as out:
        pending: dict[Any, dict[str, Any]] = {}
        queue = list(order)
        while queue or pending:
            while queue and len(pending) < workers:
                if spent() - start + est * (len(pending) + 1) > max_usd:
                    print(f"budget: stopping new clips at ${spent() - start:.3f} (limit ${max_usd:.2f})", flush=True)
                    queue = []
                    break
                c = queue.pop(0)
                pending[pool.submit(run_one, c, profile)] = c
            if not pending:
                break
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for f in finished:
                pending.pop(f)
                row = f.result()
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(f"{row['id']}: {row['seconds']} s, {row.get('status')} {row.get('equation') or ''} | audit "
                      f"{row.get('audit')} {row.get('violations') or ''} ${row.get('spend_usd', 0):.3f} {row.get('error') or ''}",
                      flush=True)
    print(f"spent this batch: ${spent() - start:.3f}")


BASELINE_SYSTEM = """You are shown frames from a short video with the time of each frame. Some videos are real
footage and some are generated by AI. Judge only the motion of the main moving object: does it obey ordinary
physics (gravity, momentum, energy, no teleporting or morphing)? Answer "violates_physics" if you see a clear
violation, "consistent" if the motion looks physically right, "cant_judge" if the frames don't show enough."""


def baseline(max_usd: float) -> None:
    """The model alone, from the same keyframes: the comparison that shows what measuring adds."""
    sys.path.insert(0, str(ROOT / "backend"))
    from typing import Literal

    from pydantic import Field
    from physicslens.agent.llm import ask_json
    from physicslens.agent.schemas import Strict
    from physicslens.costguard import CostGuardClient
    from physicslens.perception import media
    from physicslens.settings import load_settings

    class Judgement(Strict):
        verdict: Literal["violates_physics", "consistent", "cant_judge"]
        reason: str = Field(max_length=300)

    client = CostGuardClient(load_settings())
    cfg = client.limits.get("media") or {}
    done = done_ids(BASELINE)
    ran = done_ids(RESULTS)
    start = spent()
    with BASELINE.open("a") as out:
        for c in json.loads(MANIFEST.read_text())["clips"]:
            if c["id"] in done or c["id"] not in ran:
                continue
            if spent() - start > max_usd:
                print("budget reached")
                break
            video = WORK / "clips" / (c["id"] + ".mp4")
            info = media.probe(video)
            n = int(cfg.get("llm_keyframes_max", 4))
            frames = media.keyframes(video, n, int(cfg.get("llm_image_long_side_px", 768)))
            dur = info.frames / info.fps
            times = [round(i * dur / max(n - 1, 1), 2) for i in range(len(frames))]
            try:
                j, call = ask_json(client, task="llm_alone", system=BASELINE_SYSTEM, schema_model=Judgement,
                                   text=f"Frame times in seconds: {times}.", run_id=f"study-baseline-{c['id']}", images=frames)
                row = {"id": c["id"], "verdict": j.verdict, "reason": j.reason, "model": f"{call.provider}:{call.model}"}
            except Exception as e:  # recorded, not hidden
                row = {"id": c["id"], "verdict": "error", "reason": f"{type(e).__name__}: {e}"[:300]}
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(row["id"], row["verdict"], flush=True)


def replay() -> None:
    """The current audit rules on every stored run (from its saved audit input): free, no model call."""
    sys.path.insert(0, str(ROOT / "backend"))
    from physicslens.physics.audit import audit, load_input

    changed = 0
    with RESULTS.open("a") as out:
        for r in rows_of(RESULTS):
            saved = WORK / "runs" / r["id"] / "audit_input.npz"
            if not saved.exists():
                continue
            a = audit(load_input(saved))
            new = {**r, "audit": a["verdict"], "headline": a["headline"],
                   "violations": sorted({v["check"] for v in a["violations"]}), "replayed": True}
            if (new["audit"], new["violations"]) != (r.get("audit"), r.get("violations")):
                changed += 1
                print(f"{r['id']}: {r.get('audit')} -> {new['audit']} {new['violations'] or ''}")
            out.write(json.dumps(new) + "\n")
    print(f"replayed; {changed} verdicts changed")


def _error_kind(err: str) -> str:
    e = err.lower()
    if "http error" in e or "urlopen" in e or "name or service" in e:
        return "download"
    if "open-weight" in e or "busy" in e or "budget" in e:
        return "model service"
    return "tracking"


def _auc(pos: list[float], neg: list[float]) -> float | None:
    """Probability that a random value from pos exceeds one from neg (ties count half): the Mann-Whitney AUC."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def final_rows() -> list[dict[str, Any]]:
    """Every clip's row under the current audit rules (replayed from its saved input, free), with the best law's
    held-out error and the model-alone verdict."""
    sys.path.insert(0, str(ROOT / "backend"))
    from physicslens.physics.audit import audit, load_input

    base = {r["id"]: r for r in rows_of(BASELINE)} if BASELINE.exists() else {}
    out = []
    for r in rows_of(RESULTS):
        row = dict(r)
        saved = WORK / "runs" / r["id"] / "audit_input.npz"
        if saved.exists():
            a = audit(load_input(saved))
            row.update(audit=a["verdict"], headline=a["headline"], violations=sorted({v["check"] for v in a["violations"]}))
        res_path = WORK / "runs" / r["id"] / "result.json"
        if res_path.exists() and row.get("status") != "error":
            res = json.loads(res_path.read_text())
            best = (res.get("ranking") or [{}])[0]
            row["heldout_nrmse"] = best.get("heldout_nrmse")
        row["model_alone"] = (base.get(r["id"]) or {}).get("verdict")
        out.append(row)
    return out


def analysis() -> dict[str, Any]:
    """Every metric of the study, from the stored runs, under the current audit rules (replayed, free)."""

    per: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final_rows():
        per[row["set"]].append(row)

    def share(k: int, n: int) -> dict[str, Any]:
        lo, hi = wilson(k, n)
        return {"k": k, "n": n, "rate": k / n if n else None, "lo": lo, "hi": hi}

    out: dict[str, Any] = {"sets": {}, "generators": {}, "human_score": {}}
    for name, rows in per.items():
        c = Counter(r["audit"] for r in rows)
        judged = c["violates_physics"] + c["consistent"]
        kinds = Counter(_error_kind(r.get("error") or "") for r in rows if r["audit"] == "error")
        # A clip that never downloaded, or that a model service failed on, says nothing about tracking.
        analysed = [r for r in rows if r["audit"] != "error" or _error_kind(r.get("error") or "") == "tracking"]
        tracked = [r for r in rows if r["audit"] != "error"]
        laws = Counter(r.get("status") for r in tracked)
        nrmse = sorted(float(r["heldout_nrmse"]) for r in tracked if r.get("heldout_nrmse") is not None)
        spend = [float(r["spend_usd"]) for r in rows if r.get("spend_usd")]
        secs = [float(r["seconds"]) for r in rows if r.get("seconds") and r.get("status") != "error"]
        out["sets"][name] = {
            "clips": len(rows), "verdicts": dict(c),
            "judged": share(judged, len(rows)),
            "flagged_among_judged": share(c["violates_physics"], judged),
            "errors": dict(kinds),
            "tracked": share(len(tracked), len(analysed)),
            "verified_law_among_tracked": share(laws["accepted"], len(tracked)),
            "heldout_error_median": nrmse[len(nrmse) // 2] if nrmse else None,
            "heldout_errors": nrmse,
            "cost_per_clip_usd": sum(spend) / len(spend) if spend else None,
            "seconds_per_clip_median": sorted(secs)[len(secs) // 2] if secs else None,
            "model_alone": dict(Counter(r["model_alone"] for r in rows if r["model_alone"])),
        }
    for r in per.get("generated", []):
        g = out["generators"].setdefault(r["generator"], Counter())
        g[r["audit"]] += 1
    out["generators"] = {g: {**dict(c), "flagged_among_judged": share(c["violates_physics"], c["violates_physics"] + c["consistent"])}
                         for g, c in out["generators"].items()}
    gen = per.get("generated", [])
    for pc in sorted({r["human_pc"] for r in gen if r["human_pc"] is not None}):
        c = Counter(r["audit"] for r in gen if r["human_pc"] == pc)
        out["human_score"][pc] = {**dict(c), "flagged_among_judged": share(c["violates_physics"], c["violates_physics"] + c["consistent"])}
    judged_gen = [r for r in gen if r["audit"] in ("violates_physics", "consistent")]
    flagged_pc = [6 - r["human_pc"] for r in judged_gen if r["audit"] == "violates_physics"]
    passed_pc = [6 - r["human_pc"] for r in judged_gen if r["audit"] == "consistent"]
    # Do flagged clips have worse human physics scores? (6 - score, so higher means people saw it as less physical.)
    out["flags_vs_human_auc"] = _auc(flagged_pc, passed_pc)
    real_err = out["sets"].get("real", {}).get("heldout_errors", [])
    gen_err = out["sets"].get("generated", {}).get("heldout_errors", [])
    # Does the best law forecast generated motion worse than real motion? AUC over the held-out errors.
    out["forecast_error_auc_generated_vs_real"] = _auc(gen_err, real_err)
    real = out["sets"].get("real")
    if real:
        n = real["judged"]["k"]
        out["false_accusation"] = {"accused": real["verdicts"].get("violates_physics", 0), "judged_real": n,
                                   "upper_95": wilson(real["verdicts"].get("violates_physics", 0), n)[1] if n else None}
    (DATA / "analysis.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    return out


def report() -> None:
    clips = {c["id"]: c for c in json.loads(MANIFEST.read_text())["clips"]}
    rows = final_rows()
    base = {r["id"]: r for r in (json.loads(x) for x in BASELINE.read_text().splitlines() if x.strip())} \
        if BASELINE.exists() else {}
    table: dict[str, Counter] = defaultdict(Counter)
    by_pc: dict[int, Counter] = defaultdict(Counter)
    for r in rows:
        table[r["generator"]][r["audit"]] += 1
        if r["human_pc"] is not None:
            by_pc[r["human_pc"]][r["audit"]] += 1
    print(f"{'source':14} {'n':>4} {'violates':>9} {'consistent':>11} {'cant judge':>11} {'error':>6}  flagged among judged")
    for g in sorted(table, key=lambda g: (g == "real", g)):
        t = table[g]
        judged = t["violates_physics"] + t["consistent"]
        print(f"{g:14} {sum(t.values()):4} {t['violates_physics']:9} {t['consistent']:11} {t['cant_judge']:11} "
              f"{t['error']:6}  {t['violates_physics'] / judged:.0%}" if judged else f"{g:14} {sum(t.values()):4}  (none judged)")
    print("\nBy human physical-commonsense score (generated clips):")
    for pc in sorted(by_pc):
        t = by_pc[pc]
        judged = t["violates_physics"] + t["consistent"]
        print(f"  pc={pc}: n={sum(t.values())}, judged={judged}, flagged={t['violates_physics']}"
              + (f" ({t['violates_physics'] / judged:.0%})" if judged else ""))
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "results.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "set", "generator", "caption", "human_pc", "status", "family", "equation", "audit",
                    "violations", "headline", "model_alone", "license", "url"])
        for r in rows:
            c = clips.get(r["id"], {})
            w.writerow([r["id"], r["set"], r["generator"], c.get("caption"), r["human_pc"], r.get("status"), r.get("family"),
                        r.get("equation"), r.get("audit"), ";".join(r.get("violations") or []), r.get("headline") or r.get("error"),
                        (base.get(r["id"]) or {}).get("verdict"), c.get("license"), c.get("url")])
    print(f"\nwrote {DATA / 'results.csv'} ({len(rows)} clips)")
    summary = {**summarize(rows, clips, base), "analysis": analysis()}
    for path in (DATA / "summary.json", ROOT / "frontend" / "public" / "audit_study.json"):
        path.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {DATA / 'summary.json'} and the site's copy")


GENERATOR_NAMES = {"real": "Real footage (Physics-IQ)", "cogvideo": "CogVideoX", "cosmos": "Cosmos", "hunyuan": "HunyuanVideo",
                   "ray2": "Ray2", "sora": "Sora", "videocrafter": "VideoCrafter", "wan": "Wan"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a share k/n that stays sensible for small n and for 0 or n (unlike k/n +- 2 sd)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(rows: list[dict[str, Any]], clips: dict[str, dict[str, Any]], base: dict[str, Any]) -> dict[str, Any]:
    def group(rs: list[dict[str, Any]]) -> dict[str, Any]:
        c = Counter(r["audit"] for r in rs)
        judged = c["violates_physics"] + c["consistent"]
        lo, hi = wilson(c["violates_physics"], judged)
        return {"n": len(rs), "judged": judged, "flagged": c["violates_physics"], "consistent": c["consistent"],
                "cant_judge": c["cant_judge"], "error": c["error"],
                "rate": c["violates_physics"] / judged if judged else None, "lo": lo, "hi": hi}

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pc: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_source[r["generator"]].append(r)
        if r["human_pc"] is not None:
            by_pc[r["human_pc"]].append(r)
    sources = [{"key": g, "label": GENERATOR_NAMES.get(g, g), **group(rs)}
               for g, rs in sorted(by_source.items(), key=lambda kv: (kv[0] != "real", kv[0]))]
    examples = []
    for r in rows:
        if r["set"] == "generated" and r["audit"] == "violates_physics" and len(examples) < 12:
            c = clips.get(r["id"], {})
            examples.append({"id": r["id"], "generator": GENERATOR_NAMES.get(r["generator"], r["generator"]),
                             "caption": c.get("caption"), "human_pc": r["human_pc"], "headline": r.get("headline"),
                             "url": c.get("url")})
    alone = Counter((base.get(r["id"]) or {}).get("verdict") for r in rows if r["id"] in base)
    return {
        "clips": len(rows), "sources": sources,
        "by_human_score": [{"pc": pc, **group(rs)} for pc, rs in sorted(by_pc.items())],
        "examples": examples,
        "model_alone": dict(alone) if alone else None,
        "licences": {"generated": "VideoPhy-2 test set, MIT", "real": "Physics-IQ, CC BY 4.0 (Google DeepMind)"},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--generated", type=int, default=150)
    s.add_argument("--real", type=int, default=50)
    s.add_argument("--seed", type=int, default=7)
    r = sub.add_parser("run")
    r.add_argument("--limit", type=int, required=True)
    r.add_argument("--workers", type=int, default=2, help="clips at once; 2 keeps a 16 GB laptop responsive")
    r.add_argument("--max-usd", type=float, required=True)
    r.add_argument("--only", choices=("generated", "real"))
    r.add_argument("--profile", choices=tuple(PROFILES), default="full")
    r.add_argument("--retry-errors", action="store_true", help="run clips that failed before once more")
    r.add_argument("--generator", help="only this generator's clips (e.g. sora), or real")
    b = sub.add_parser("baseline")
    b.add_argument("--max-usd", type=float, required=True)
    sub.add_parser("report")
    sub.add_parser("replay", help="apply the current audit rules to every stored run (free)")
    sub.add_parser("analysis", help="every metric of the study, under the current rules (free)")
    a = ap.parse_args()
    os.environ.setdefault("APP_ENV", "demo")  # open-weight reasoning only, as judges get it
    # A batch shares the machine with its owner: few threads per process, and the lowest CPU priority, so the
    # desktop stays usable (four parallel clips with default threading ran a 16 GB laptop out of memory).
    for name, value in (("PHYSICSLENS_DECODE_WORKERS", "2"), ("FFMPEG_THREADS", "2"), ("OMP_NUM_THREADS", "1"),
                        ("OPENBLAS_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1")):
        os.environ.setdefault(name, value)
    os.nice(10)
    if a.cmd == "select":
        select(a.generated, a.real, a.seed)
    elif a.cmd == "run":
        run(a.limit, a.workers, a.max_usd, a.only, a.profile, a.retry_errors, a.generator)
    elif a.cmd == "baseline":
        baseline(a.max_usd)
    elif a.cmd == "replay":
        replay()
    elif a.cmd == "analysis":
        print(json.dumps({k: v for k, v in analysis().items()}, indent=1, default=float)[:6000])
    else:
        report()


if __name__ == "__main__":
    main()
