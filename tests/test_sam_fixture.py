"""Checks against a real SAM 3.1 response on samples/smoke_ball.mp4 (synthetic pendulum with known positions)."""

import json
import math
import re
from pathlib import Path

import meta_sam_parser as msp
import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "sam_smoke_response.json"
LINE = re.compile(r"<(\d+)f>(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);[^|]*\|><\|mask;[^|]*data=(\d+),(\d+),([^|]*)\|>")


def true_center(frame: int) -> tuple[float, float]:
    # Must match scripts/make_smoke_clip.py
    theta = 0.5 * math.cos(2 * math.pi * frame / 30)
    return 320 + 250 * math.sin(theta), 40 + 250 * math.cos(theta)


def frame_lines() -> list[str]:
    raw = json.loads(FIXTURE.read_text())
    text = "".join(c.get("text", "") for item in raw["output"] for c in item.get("content") or [])
    return text.splitlines()


def test_every_frame_tracked_and_billed():
    raw = json.loads(FIXTURE.read_text())
    assert raw["usage"]["video_frames_processed"] == 60
    assert sorted(int(LINE.match(line).group(1)) for line in frame_lines()) == list(range(60))


@pytest.mark.parametrize("index", [0, 13, 29, 47, 59])
def test_masks_are_box_local(index):
    m = LINE.match(frame_lines()[index])
    frame = int(m.group(1))
    x1, y1 = int(m.group(3)), int(m.group(4))
    height, width, payload = int(m.group(7)), int(m.group(8)), m.group(9)
    # The decoder expects the payload WITH its encoding marker ('!' one_bit, '~' lossless).
    mask = msp.SegmentationMask(encoding="one_bit" if payload[0] == "!" else "lossless",
                                payload=payload, width=width, height=height)
    raster = np.frombuffer(msp.decode_mask_to_raster(mask), np.uint8).reshape(height, width)
    ys, xs = np.nonzero(raster)
    tx, ty = true_center(frame)
    assert abs(x1 + xs.mean() - tx) < 1.5 and abs(y1 + ys.mean() - ty) < 1.5


def test_a_corrupted_mask_is_dropped_not_measured():
    """Validation moved out of the parser into the single strict decode: a bad payload must still never become data."""
    from physicslens.perception.sam_track import tracks_from_text
    lines = [ln for ln in frame_lines() if ln.strip()]
    good = tracks_from_text("\n".join(lines))
    m = re.search(r"data=(\d+),(\d+),!(\S+?)\|>", lines[5])
    corrupt = lines[5].replace(m.group(3), m.group(3)[::-1])  # same length and marker, wrong content
    bad = tracks_from_text("\n".join([*lines[:5], corrupt, *lines[6:]]))
    oid = next(iter(good))
    assert 5 in good[oid].frames and 5 not in bad[oid].frames
    assert len(bad[oid].frames) == len(good[oid].frames) - 1
