"""SAM 3.1 output text -> per-object pixel tracks. Parsing uses the official meta_sam_parser package."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import meta_sam_parser as msp
import numpy as np

log = logging.getLogger("physicslens.perception")


class TrackingError(RuntimeError):
    pass


@dataclass(frozen=True)
class ObjectTrack:
    object_id: str
    frames: np.ndarray  # (N,) frame indices of the SAM clip, ascending
    centers_px: np.ndarray  # (N, 2) mask centroid (col, row), full-frame pixel index coordinates
    angles: np.ndarray  # (N,) orientation of the mask's long axis, radians
    long_px: np.ndarray  # (N,) min-area-rectangle sides: used to calibrate scale from reference objects
    short_px: np.ndarray
    confidence: np.ndarray
    boxes: np.ndarray | None = None  # (N, 4) SAM's own box per frame (left, top, right, bottom), for drawing checks


def parse_sam_text(text: str, completed: bool = True) -> list[msp.SegmentationMaskRecord]:
    parser = msp.video_segmentation_format().create_parser()
    parser.push(text, emit=False)
    outcome = msp.CompletedOutcome() if completed else msp.IncompleteOutcome(reason="eof")
    result = parser.finish(outcome).result
    for d in result.diagnostics:
        log.warning("SAM output diagnostic: %s", d)
    if result.outcome.status != "completed":
        log.warning("SAM output incomplete: %s", result.outcome)
    # Malformed lines surface as diagnostics, never as records, so they can't leak in as data.
    return [r for r in result.records if r.kind == "mask"]


def mask_raster(record: msp.SegmentationMaskRecord) -> np.ndarray:
    m = record.mask
    return np.frombuffer(msp.decode_mask_to_raster(m), np.uint8).reshape(m.height, m.width)


def _shape(raster: np.ndarray, left: int, top: int) -> tuple[float, float, float, float, float] | None:
    ys, xs = np.nonzero(raster)
    if len(xs) < 3:
        return None
    # Masks are box-local: verified against a synthetic clip with known positions (tests/test_sam_fixture.py).
    cx, cy = left + xs.mean(), top + ys.mean()
    (_, _), (w, h), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
    cov = np.cov(np.vstack([xs, ys]))
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, np.argmax(evals)]
    angle = float(np.arctan2(major[1], major[0]))
    return float(cx), float(cy), max(w, h) + 1.0, min(w, h) + 1.0, angle  # +1: pixel extents, not center distances


def tracks_from_records(records: list[msp.SegmentationMaskRecord]) -> dict[str, ObjectTrack]:
    per_object: dict[str, dict[int, tuple]] = {}
    for r in records:
        shape = _shape(mask_raster(r), r.bounds.left, r.bounds.top)
        if shape is None:
            continue
        # Keyed by frame index and object id, never by line order: frames can arrive out of order.
        b = r.bounds
        per_object.setdefault(r.object_id, {})[r.frame.frame_index] = (
            *shape, r.confidence or 0.0, b.left, b.top, b.right, b.bottom)

    tracks = {}
    for oid, by_frame in per_object.items():
        frames = np.array(sorted(by_frame))
        rows = np.array([by_frame[f] for f in frames])
        tracks[oid] = ObjectTrack(oid, frames, rows[:, 0:2], rows[:, 4], rows[:, 2], rows[:, 3], rows[:, 5], rows[:, 6:10])
    return tracks


def tracks_from_text(text: str, completed: bool = True) -> dict[str, ObjectTrack]:
    tracks = tracks_from_records(parse_sam_text(text, completed))
    if not tracks:
        raise TrackingError("SAM found no matching object in the clip.")
    return tracks
