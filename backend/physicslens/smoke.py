"""`make smoke`: the only command that spends money. One tiny real call per model, capped by smoke_cap_usd."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from .costguard import CostGuardClient, CostGuardError, ProviderCallFailed
from .settings import (
    BASETEN_REASONING_MODEL,
    META_CONTRIBUTOR_MODEL,
    META_TEXT_MODEL,
    SAM_MODEL,
    load_budget,
    load_settings,
)

SMOKE_CLIP = Path("samples/smoke_ball.mp4")
FIXTURES = Path("tests/fixtures")
BASETEN_CANDIDATES = (BASETEN_REASONING_MODEL, "meta-models/muse-spark-open")


def clip_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets,duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    if float(stream["duration"]) > 3.0:
        sys.exit(f"{path} is longer than 3 s; the smoke test only uses a tiny clip.")
    return int(stream["nb_read_packets"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    if settings.mock_models:
        sys.exit("make smoke must run with MOCK_MODELS=0.")
    cap = float(load_budget(settings.config_dir)["smoke_cap_usd"])
    client = CostGuardClient(settings)
    run_id = f"smoke-{int(time.time())}"
    total = 0.0

    print("1) Meta: list models (free)")
    meta_models = client.list_models("meta_api")
    for m in (META_TEXT_MODEL, META_CONTRIBUTOR_MODEL, SAM_MODEL, "muse-voice-transcribe-1.0"):
        print(f"   {m:<32} {'available' if m in meta_models else 'NOT LISTED'}")

    print("2) Meta: one tiny Muse Spark call")
    model = settings.meta_text_model()
    try:
        r = client.meta_text(input="Reply with the single word: ok", model=model, run_id=run_id,
                             effort="minimal", max_output_tokens=200, use_cache=False, run_cap_usd=cap)
    except ProviderCallFailed as e:
        if model != META_CONTRIBUTOR_MODEL or e.status_code != 404:
            raise
        print(f"   {model} not available to this account (404); set USE_CONTRIBUTOR_MODEL=0. Trying {META_TEXT_MODEL}.")
        model = META_TEXT_MODEL
        r = client.meta_text(input="Reply with the single word: ok", model=model, run_id=run_id,
                             effort="minimal", max_output_tokens=200, use_cache=False, run_cap_usd=cap)
    total += r.cost_usd
    print(f"   {model}: {r.text.strip()!r}  usage={r.usage.as_dict()}  cost=${r.cost_usd:.5f}")

    print("3) Meta: one SAM 3.1 video call on a <=3 s clip")
    if not SMOKE_CLIP.is_file():
        sys.exit(f"Missing {SMOKE_CLIP}.")
    frames = clip_frames(SMOKE_CLIP)
    file_id, video_hash = client.upload_video(SMOKE_CLIP)
    try:
        r = client.sam_video(prompt="red ball", file_id=file_id, video_hash=video_hash, est_frames=frames,
                             metadata={"mask_encoding": "one_bit", "include_confidence": "true"},
                             run_id=run_id, run_cap_usd=cap)
    finally:
        client.delete_file(file_id, video_hash)
    total += r.cost_usd
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "sam_smoke_response.json").write_text(json.dumps(r.raw, indent=2))
    print(f"   {len(r.text.splitlines())} output lines, frames billed={r.usage.video_frames}, cost=${r.cost_usd:.5f}")
    print(f"   raw response saved to {FIXTURES / 'sam_smoke_response.json'}")

    print("4) Baseten: Muse Spark Open model name (free list call)")
    try:
        names = client.list_models("baseten")
        for m in BASETEN_CANDIDATES:
            print(f"   {m:<36} {'available' if m in names else 'NOT LISTED'}")
        if not any(m in names for m in BASETEN_CANDIDATES):
            print("   FAIL: neither Muse Spark Open model name is listed.")
    except CostGuardError as e:
        print(f"   skipped: {e}")
    print("   (Baseten generation is skipped until its price in config/prices.yaml is verified.)")

    print(f"\nSmoke total (ledger estimate): ${total:.5f}  (cap ${cap:.2f})")
    print("Now compare with the Meta dashboard Usage page; the dashboard is the truth.")


if __name__ == "__main__":
    main()
