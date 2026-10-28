"""Baseline: what does the reasoning model say about the physics on its own, from the same frames, with no measurement?

For each clip it sends the reasoning model the keyframes PhysicsLens itself uses, with their timestamps, and asks for
the governing equation and its parameters, several times to show the spread. Every raw answer is saved verbatim, next
to the truth (when known) and PhysicsLens's verified result, so the table shows whatever the numbers say.

  python scripts/llm_alone.py --clip gallery/gallery-pendulum/clip.mp4 --result gallery/gallery-pendulum/result.json \
      --truth '{"g": 9.81, "period_s": 1.419}' --note "0.5 m string" --runs 5 --out benchmark/llm_alone

Uses the configured reasoning provider (REASONING_PROVIDER=glimmer or together) through the cost guard. It makes paid
calls when the provider is paid: say what it will cost and get approval first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.agent.llm import ask_json
from physicslens.agent.schemas import Strict
from physicslens.costguard import CostGuardClient
from physicslens.perception import media
from physicslens.settings import load_settings

SYSTEM = """You are shown frames from a video of a moving object, with the time of each frame. Say which differential
equation governs the motion and estimate its physical parameters (for example g in m/s^2, a length in m, the period in
s, a damping rate in 1/s, a coefficient of restitution). Give your best numerical estimates even if unsure."""


class GuessParam(Strict):
    name: str = Field(max_length=40)
    value: float
    unit: str = Field(max_length=20)


class LLMGuess(Strict):
    motion: str = Field(max_length=200)
    equation: str = Field(max_length=300)
    parameters: list[GuessParam] = Field(max_length=8)
    period_s: float | None
    reasoning_summary: str = Field(max_length=600)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--result", help="PhysicsLens result.json for the same clip")
    ap.add_argument("--truth", default="{}", help='known answers as JSON, e.g. {"g": 9.81}')
    ap.add_argument("--note", default="", help="what is known about the setup, given to the model as to PhysicsLens")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not 1 <= a.runs <= 10:
        sys.exit("Use 1 to 10 runs.")

    settings = load_settings()
    client = CostGuardClient(settings)
    clip = Path(a.clip)
    info = media.probe(clip)
    media_cfg = client.limits.get("media") or {}
    count = int(media_cfg.get("llm_keyframes_max", 4))
    frames = media.keyframes(clip, count, int(media_cfg.get("llm_image_long_side_px", 768)))
    duration = info.frames / info.fps
    times = [round(i * duration / max(count - 1, 1), 3) for i in range(len(frames))]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    answers = []
    for k in range(a.runs):
        # The attempt number keeps each run out of the response cache: the point is the spread of fresh answers.
        text = (f"Frame times in seconds: {times}. Clip length {duration:.2f} s. {a.note} "
                f"(independent attempt {k + 1} of {a.runs})")
        guess, call = ask_json(client, task="llm_alone", system=SYSTEM, text=text, schema_model=LLMGuess,
                               run_id=f"llm-alone-{clip.stem}-{k}", images=frames)
        answers.append({"attempt": k + 1, "model": call.model, "provider": call.provider, "answer": guess.model_dump(),
                        "reasoning": call.reasoning})
        print(f"attempt {k + 1}: {guess.equation} {[(p.name, p.value, p.unit) for p in guess.parameters]}")
    physicslens = None
    if a.result:
        r = json.loads(Path(a.result).read_text())
        physicslens = {"status": r.get("status"), "equation": (r.get("equation") or {}).get("title"),
                       "parameters": [{k: p.get(k) for k in ("name", "value", "std", "unit")} for p in r.get("parameters", [])]}
    record = {"clip": str(clip), "note": a.note, "truth": json.loads(a.truth), "frame_times_s": times,
              "llm_alone": answers, "physicslens": physicslens}
    (out / f"{clip.parent.name or clip.stem}.json").write_text(json.dumps(record, indent=1))
    print(f"saved {out / f'{clip.parent.name or clip.stem}.json'}")


if __name__ == "__main__":
    main()
