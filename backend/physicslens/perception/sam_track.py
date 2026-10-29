"""SAM 3.1 output text -> per-object pixel tracks. Parsing uses the official meta_sam_parser package."""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import cv2
import meta_sam_parser as msp
import numpy as np
from meta_sam_parser import _segmentation

log = logging.getLogger("physicslens.perception")

# The official decoder is pure Python and strict (it re-encodes every mask to check it is canonical), and its time
# grows with mask area: a 450-frame clip took 4 to 5 minutes on one core, and 59 frames of a truck filling half
# the image over a minute. Masks decode independently, so big responses are spread over processes; small ones are
# not worth the pool's start-up (about a second).
PARALLEL_MIN_PIXELS = 400_000


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
    area_px: np.ndarray | None = None  # (N,) mask pixel count: a ball's apparent diameter for the 3D depth mode


def _structural_check(mask: msp.SegmentationMask) -> bytes:
    """Stand-in for the parser's own full decode while parsing: the same header checks, no decoding.

    The parser decodes every mask once just to validate it, serially, and we decode each again to measure it: on a
    59-frame truck clip that validation alone took about 80 s. Here the full strict decode happens exactly once, in
    parallel, in _mask_shape, and a mask that fails it is dropped there, as the parser would have dropped it.
    """
    marker = {"one_bit": "!", "lossless": "~"}.get(mask.encoding)
    if marker is None or not isinstance(mask.payload, str) or not mask.payload.startswith(marker):
        raise msp.InvalidSegmentationMaskError(f"Unsupported or malformed {mask.encoding} mask payload.")
    if not (isinstance(mask.width, int) and isinstance(mask.height, int) and mask.width > 0 and mask.height > 0):
        raise msp.InvalidSegmentationMaskError("Mask dimensions must be positive integers.")
    return b""


_PARSE_LOCK = threading.Lock()  # the swap below is process-wide, so parses take turns (each takes milliseconds)


def parse_sam_text(text: str, completed: bool = True) -> list[msp.SegmentationMaskRecord]:
    with _PARSE_LOCK:
        original = _segmentation.decode_mask_to_raster
        _segmentation.decode_mask_to_raster = _structural_check
        try:
            parser = msp.video_segmentation_format().create_parser()
            parser.push(text, emit=False)
            outcome = msp.CompletedOutcome() if completed else msp.IncompleteOutcome(reason="eof")
            result = parser.finish(outcome).result
        finally:
            _segmentation.decode_mask_to_raster = original
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


def _mask_shape(job: tuple[str, str, int, int, int, int]) -> tuple[tuple[float, ...] | None, float]:
    """Decode one mask (strictly, with the official checks) and measure it. Top level so a worker process can run it.

    A mask the official decoder rejects measures as nothing: it is dropped, never treated as data."""
    encoding, payload, width, height, left, top = job
    m = msp.SegmentationMask(encoding=encoding, payload=payload, width=width, height=height)
    try:
        raster = np.frombuffer(msp.decode_mask_to_raster(m), np.uint8).reshape(height, width)
    except msp.InvalidSegmentationMaskError:
        return None, 0.0
    return _shape(raster, left, top), float(np.count_nonzero(raster))


def _measure_all(records: list[msp.SegmentationMaskRecord]) -> list[tuple[tuple[float, ...] | None, float]]:
    jobs = [(r.mask.encoding, r.mask.payload, r.mask.width, r.mask.height, r.bounds.left, r.bounds.top) for r in records]
    workers = min(os.cpu_count() or 1, 16)
    if sum(j[2] * j[3] for j in jobs) < PARALLEL_MIN_PIXELS or workers < 2:
        return [_mask_shape(j) for j in jobs]
    # spawn, not fork: the server process has threads, and forking a threaded process can deadlock.
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(_mask_shape, jobs, chunksize=max(1, len(jobs) // (workers * 4))))


def tracks_from_records(records: list[msp.SegmentationMaskRecord]) -> dict[str, ObjectTrack]:
    per_object: dict[str, dict[int, tuple]] = {}
    for r, (shape, area) in zip(records, _measure_all(records), strict=True):
        if shape is None:
            continue
        # Keyed by frame index and object id, never by line order: frames can arrive out of order.
        b = r.bounds
        per_object.setdefault(r.object_id, {})[r.frame.frame_index] = (
            *shape, r.confidence or 0.0, b.left, b.top, b.right, b.bottom, area)

    tracks = {}
    for oid, by_frame in per_object.items():
        frames = np.array(sorted(by_frame))
        rows = np.array([by_frame[f] for f in frames])
        tracks[oid] = ObjectTrack(oid, frames, rows[:, 0:2], rows[:, 4], rows[:, 2], rows[:, 3], rows[:, 5], rows[:, 6:10],
                                  rows[:, 10])
    return tracks


def tracks_from_text(text: str, completed: bool = True) -> dict[str, ObjectTrack]:
    tracks = tracks_from_records(parse_sam_text(text, completed))
    if not tracks:
        raise TrackingError("SAM found no matching object in the clip.")
    return tracks
