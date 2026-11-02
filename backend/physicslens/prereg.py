"""Pre-registered predictions: a forecast written down, hashed and timestamped before the answer exists.

A prediction made after the answer is known can always be bent to fit it. So each one is a file,
predictions/<case>/<id>/prediction.json, holding its inputs (with sources), the forecast with intervals, the rule that
will score it and the code version that made it, and its SHA-256 goes into predictions/INDEX.md. Pushing the folder
makes GitHub's history the timestamp. A prediction is never edited: a corrected one is a new id, and the old one stays
visible. Later, outcomes/<case>/<id>.json scores it against what happened, and the score never touches the
prediction file.

Local files only, no network.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "predictions"
OUTCOMES = ROOT / "outcomes"
INDEX = PRED / "INDEX.md"
INDEX_HEAD = ("# Pre-registered predictions\n\nEach row: a prediction file and its SHA-256, written when the prediction was "
              "made. Recompute with `sha256sum predictions/<case>/<id>/prediction.json`; a mismatch means the file was "
              "changed after it was registered. Scores live in `outcomes/`, never in the prediction.\n\n"
              "| Created (UTC) | Case | Id | Scored after (UTC) | SHA-256 |\n|---|---|---|---|---|\n")
REQUIRED = ("title", "question", "inputs", "prediction", "scoring")


class PreregError(ValueError):
    pass


def code_version() -> dict[str, Any]:
    """The commit the prediction was made from, and whether the tree had uncommitted changes (stated, not hidden)."""
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10,
                                  check=False).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"commit": git("rev-parse", "HEAD") or None, "uncommitted_changes": bool(git("status", "--porcelain"))}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_for(case: str, pid: str, root: Path = PRED) -> Path:
    for part in (case, pid):
        if not part or not all(c.isalnum() or c in "-_." for c in part) or part.startswith("."):
            raise PreregError(f"bad name {part!r}: letters, digits, '-', '_' and '.' only")
    return root / case / pid / "prediction.json"


def register(case: str, pid: str, body: dict[str, Any], root: Path = PRED, now: datetime | None = None) -> Path:
    """Write a new prediction and its hash. Refuses to overwrite: corrections get a new id."""
    missing = [k for k in REQUIRED if k not in body]
    if missing:
        raise PreregError(f"prediction lacks {', '.join(missing)}")
    if "score_after_utc" not in body["scoring"]:
        raise PreregError("scoring needs score_after_utc: when reality can first answer")
    path = path_for(case, pid, root)
    if path.exists():
        raise PreregError(f"{path} exists: a prediction is never edited; register a corrected one under a new id")
    created = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    doc = {"id": pid, "case": case, "created_utc": created, "code_version": code_version(), **body, "status": "open"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=False, default=float) + "\n")
    index = root / "INDEX.md"
    if not index.exists():
        index.write_text(INDEX_HEAD)
    with index.open("a") as f:
        f.write(f"| {created} | {case} | {pid} | {body['scoring']['score_after_utc']} | `{sha256(path)}` |\n")
    return path


def index_rows(root: Path = PRED) -> list[dict[str, str]]:
    index = root / "INDEX.md"
    if not index.exists():
        return []
    rows = []
    for line in index.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[4].startswith("`"):
            rows.append({"created_utc": cells[0], "case": cells[1], "id": cells[2], "score_after_utc": cells[3],
                         "sha256": cells[4].strip("`")})
    return rows


def verify(root: Path = PRED) -> list[str]:
    """Every registered prediction still matches the hash written when it was made. Returns the problems."""
    problems = []
    for r in index_rows(root):
        p = path_for(r["case"], r["id"], root)
        if not p.exists():
            problems.append(f"{r['case']}/{r['id']}: file missing")
        elif sha256(p) != r["sha256"]:
            problems.append(f"{r['case']}/{r['id']}: changed after it was registered")
    return problems


def score(case: str, pid: str, outcome: dict[str, Any], root: Path = PRED, outcomes: Path = OUTCOMES,
          now: datetime | None = None) -> Path:
    """Record what happened next to a prediction (hit or miss per item), without touching the prediction."""
    pred = path_for(case, pid, root)
    if not pred.exists():
        raise PreregError(f"no prediction {case}/{pid}")
    if "items" not in outcome or "summary" not in outcome:
        raise PreregError("an outcome needs items (per predicted quantity) and a summary")
    doc = json.loads(pred.read_text())
    when = now or datetime.now(UTC)
    if when.strftime("%Y-%m-%dT%H:%M:%SZ") < doc["scoring"]["score_after_utc"]:
        raise PreregError("too early: the prediction's own scoring date has not come")
    out = outcomes / case / f"{pid}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"id": pid, "case": case, "prediction_sha256": sha256(pred),
                               "predicted_utc": doc["created_utc"], "scored_utc": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
                               **outcome}, indent=1, default=float) + "\n")
    return out


def summary(root: Path = PRED, outcomes: Path = OUTCOMES) -> dict[str, Any]:
    """Every prediction with its score if any, and the running tally. Misses stay in the list."""
    items = []
    for r in index_rows(root):
        p = path_for(r["case"], r["id"], root)
        doc = json.loads(p.read_text()) if p.exists() else {}
        o = outcomes / r["case"] / f"{r['id']}.json"
        out = json.loads(o.read_text()) if o.exists() else None
        items.append({**r, "title": doc.get("title"), "question": doc.get("question"),
                      "prediction": doc.get("prediction"), "scoring": doc.get("scoring"),
                      "intact": bool(p.exists() and sha256(p) == r["sha256"]), "outcome": out})
    scored = [i for i in items if i["outcome"]]
    hits = sum(i["outcome"].get("hits", 0) for i in scored)
    misses = sum(i["outcome"].get("misses", 0) for i in scored)
    return {"predictions": items, "open": len(items) - len(scored), "scored": len(scored), "hits": hits,
            "misses": misses}
