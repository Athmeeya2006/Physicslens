"""Moving objects across a sequence of frames: what is not a star, not a hot pixel and not a cosmic ray.

Each frame's sources are detected; a source that sits at the same place (within a pixel or two) in most frames is a
star or a hot pixel and is removed; what is left is linked into tracks of constant rate in pixels per second, from
pairs of detections in different frames predicting the rest. A detection seen in one frame only (a cosmic ray) never
makes a track: a track needs detections in min_frames frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    frames: list[int]  # frame index per detection
    xy: np.ndarray  # (n, 2) pixel positions
    t_s: np.ndarray  # (n,) seconds from the first frame
    rate_px_s: tuple[float, float]
    rms_px: float

    def to_dict(self) -> dict:
        return {"frames": self.frames, "xy": self.xy.round(3).tolist(), "t_s": self.t_s.round(4).tolist(),
                "rate_px_per_s": [round(v, 4) for v in self.rate_px_s], "rms_px": round(self.rms_px, 4)}


def static_mask(detections: list[np.ndarray], radius_px: float = 2.0, min_fraction: float = 0.5) -> list[np.ndarray]:
    """Per frame, True for detections that recur at the same pixel in at least min_fraction of the frames."""
    from scipy.spatial import cKDTree
    n = len(detections)
    trees = [cKDTree(d) if len(d) else None for d in detections]
    out = []
    for d in detections:
        if not len(d):
            out.append(np.zeros(0, bool))
            continue
        hits = np.zeros(len(d))
        for tr in trees:
            if tr is not None:
                dist, _ = tr.query(d, distance_upper_bound=radius_px)
                hits += np.isfinite(dist)
        out.append(hits >= max(2, min_fraction * n))
    return out


def link(detections: list[np.ndarray], times_s: np.ndarray, tolerance_px: float = 2.0, min_frames: int = 3,
         max_rate_px_s: float = 1e4) -> list[Track]:
    """Constant-rate tracks through the non-static detections. Each pair of detections in two different frames defines
    a rate; the hypothesis that collects most other detections within tolerance wins, its detections are removed, and
    the search repeats."""
    times_s = np.asarray(times_s, dtype=float)
    pool = [(k, i, d[i]) for k, d in enumerate(detections) for i in range(len(d))]
    used: set[tuple[int, int]] = set()
    tracks: list[Track] = []
    while True:
        free = [(k, i, p) for k, i, p in pool if (k, i) not in used]
        best: list[tuple[int, int, np.ndarray]] = []
        for a in range(len(free)):
            ka, _, pa = free[a]
            for b in range(a + 1, len(free)):
                kb, _, pb = free[b]
                if kb == ka:
                    continue
                dt = times_s[kb] - times_s[ka]
                rate = (pb - pa) / dt
                if np.hypot(*rate) > max_rate_px_s:
                    continue
                members = []
                for k in range(len(detections)):
                    cand = [(kk, ii, pp) for kk, ii, pp in free if kk == k]
                    if not cand:
                        continue
                    pred = pa + rate * (times_s[k] - times_s[ka])
                    dists = [np.hypot(*(pp - pred)) for _, _, pp in cand]
                    j = int(np.argmin(dists))
                    if dists[j] <= tolerance_px:
                        members.append(cand[j])
                if len(members) > len(best):
                    best = members
        if len(best) < min_frames:
            return tracks
        frames = [k for k, _, _ in best]
        xy = np.array([p for _, _, p in best])
        t = times_s[frames] - times_s[0]
        A = np.column_stack([np.ones_like(t), t])
        coef, *_ = np.linalg.lstsq(A, xy, rcond=None)
        rms = float(np.sqrt(np.mean(np.sum((A @ coef - xy) ** 2, axis=1))))
        tracks.append(Track(frames, xy, t, (float(coef[1, 0]), float(coef[1, 1])), rms))
        used |= {(k, i) for k, i, _ in best}


def find_moving(detections: list[np.ndarray], times_s: np.ndarray, static_radius_px: float = 2.0,
                tolerance_px: float = 2.0, min_frames: int = 3) -> list[Track]:
    """Remove what does not move, then link what does."""
    masks = static_mask(detections, static_radius_px)
    movers = [d[~m] if len(d) else d for d, m in zip(detections, masks, strict=True)]
    return link(movers, times_s, tolerance_px, min_frames)
