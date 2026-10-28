"""3D from one phone, spheres only: a ball's apparent size gives its distance through a pinhole camera.

Z = f D / d, with f the focal length in pixels, D the ball's real diameter and d its diameter in the image; the
image position then gives X = (u - cx) Z / f and Y = (v - cy) Z / f. Only a sphere works: its outline is a disc of
the same size from every side, so a change in d is a change in distance and nothing else.

Pure local math on the tracked masks. Camera frame: X right, Y up (image rows point down), Z away from the
camera, meters. The principal point is taken as the image centre and gravity as the image's down direction,
which holds for a phone held upright and level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..sigproc.cleanup import fill_short_gaps
from .coords import Coordinates
from .fit import fit_model
from .models import MODELS_3D
from .uncertainty import bootstrap
from .verify import assess, judge_candidates

# Floors on the per-frame noise. SAM's mask edge moves by about a pixel from frame to frame; the diameter's
# own frame-to-frame scatter is used instead when it is larger.
DIAMETER_NOISE_PX = 1.0
CENTER_NOISE_PX = 1.0
# A ball's mask is round. Streaked by motion blur it stretches along the motion, and its "diameter" is then
# neither the long nor the short side.
MAX_BLUR_ASPECT = 1.3
# A sudden loss of area against the neighbouring frames means a hand or the frame edge covers part of the ball;
# a ball flying away shrinks smoothly, which the neighbours follow.
MAX_AREA_DROP = 0.30
NEIGHBOURS = 3
MAX_GAP = 3  # rejected frames bridged by interpolation, as in the 2D track; longer gaps split the track
MIN_FRAMES = 20
SPHERE_WORDS = ("ball", "sphere")
CANDIDATES = ("free_fall_3d", "linear_drag_3d", "quadratic_drag_3d")


class DepthError(ValueError):
    """The 3D mode can't run on this clip; the message says why and is shown as-is."""


def focal_from_reference(size_px: float, distance_m: float, size_m: float) -> float:
    """One-time focal length calibration: f_px = d_px * Z / D.

    Film something of known size flat and facing the camera at a measured distance (an A4 sheet, long side
    0.297 m, with a tape measure from the phone's lens to the sheet), at the same zoom and resolution as the
    throws; phones switch lenses between zoom levels, and each lens has its own f. size_px is the sheet's long
    side in pixels on a frame of that video.
    """
    if size_px <= 0 or distance_m <= 0 or size_m <= 0:
        raise DepthError("The size in pixels, the distance and the real size must all be positive.")
    return size_px * distance_m / size_m


def equivalent_diameter(area_px: np.ndarray | None, long_px: np.ndarray, short_px: np.ndarray) -> np.ndarray:
    """Diameter of the disc with the mask's area: uses every mask pixel, so it is steadier than any one side."""
    if area_px is not None:
        return np.sqrt(4.0 * np.asarray(area_px, dtype=float) / np.pi)
    # No mask area stored (tracks built before it was): the mask's minimum-area rectangle sides, which for a disc
    # both equal the diameter.
    return (np.asarray(long_px, dtype=float) + np.asarray(short_px, dtype=float)) / 2


@dataclass(frozen=True)
class FrameCheck:
    keep: np.ndarray  # (N,) bool
    blurred: int
    occluded: int


def check_frames(diameter: np.ndarray, long_px: np.ndarray, short_px: np.ndarray, boxes: np.ndarray | None = None,
                 size: tuple[int, int] | None = None) -> FrameCheck:
    """Reject motion-blurred frames (elongated mask) and partly hidden ones (area drop, or cut by the frame edge)."""
    blurred = np.asarray(long_px, dtype=float) / np.maximum(np.asarray(short_px, dtype=float), 1e-9) > MAX_BLUR_ASPECT
    area = np.asarray(diameter, dtype=float) ** 2
    occluded = np.zeros(len(area), dtype=bool)
    for i in range(len(area)):
        lo, hi = max(0, i - NEIGHBOURS), min(len(area), i + NEIGHBOURS + 1)
        ref = [area[j] for j in range(lo, hi) if j != i and not blurred[j]]
        if ref and area[i] < (1 - MAX_AREA_DROP) * float(np.median(ref)):
            occluded[i] = True
    if boxes is not None and size is not None:
        w, h = size
        b = np.asarray(boxes, dtype=float)
        occluded |= (b[:, 0] <= 0) | (b[:, 1] <= 0) | (b[:, 2] >= w) | (b[:, 3] >= h)
    occluded &= ~blurred
    return FrameCheck(~(blurred | occluded), int(blurred.sum()), int(occluded.sum()))


def diameter_noise_px(diameter: np.ndarray) -> float:
    """Frame-to-frame scatter of the diameter (robust, from second differences), never below the 1 px floor."""
    if len(diameter) < 5:
        return DIAMETER_NOISE_PX
    d2 = np.diff(np.asarray(diameter, dtype=float), 2)
    # A second difference of white noise has variance 6 sigma^2; the median absolute deviation ignores outliers.
    sigma = 1.4826 * float(np.median(np.abs(d2 - np.median(d2)))) / np.sqrt(6.0)
    return max(DIAMETER_NOISE_PX, sigma)


def back_project(centers_px: np.ndarray, diameter_px: np.ndarray, size_m: float, focal_px: float, cx: float, cy: float,
                 sigma_d_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N, 3) positions (X right, Y up, Z away), per-frame depth error and per-frame lateral error, meters."""
    c = np.asarray(centers_px, dtype=float)
    d = np.asarray(diameter_px, dtype=float)
    z = focal_px * size_m / d
    xyz = np.column_stack([(c[:, 0] - cx) * z / focal_px, -(c[:, 1] - cy) * z / focal_px, z])
    # Relative depth error equals relative diameter error: 1 px on a 40 px ball is 2.5% of the distance.
    return xyz, z * sigma_d_px / d, z * CENTER_NOISE_PX / focal_px


def skip_reason(*, focal_px: float | None, size_m: float | None, moving_object: str, family: str) -> str | None:
    """Why the 3D mode can't run on this result, or None if it can try."""
    if focal_px is None:
        return ("3D was not used: it needs the camera's focal length in pixels, and video files from phones don't record "
                "one in a form that can be read. Measure it once (film an A4 sheet flat at a known distance) and enter it. "
                "The 2D analysis above is unaffected.")
    if size_m is None:
        return "3D was not used: it needs the ball's real diameter (enter the object's size)."
    if not any(w in moving_object.lower() for w in SPHERE_WORDS):
        return (f"3D was not used: depth from apparent size only works for a ball, whose outline is the same from every "
                f"side, and the moving object was read as {moving_object!r}.")
    if family != "projectile":
        return f"3D was not used: its laws are for a thrown or falling ball, and the motion was fitted as {family}."
    return None


def analyze(frames: np.ndarray, centers_px: np.ndarray, long_px: np.ndarray, short_px: np.ndarray,
            area_px: np.ndarray | None, *, fps: float, width: int, height: int, focal_px: float, size_m: float,
            boxes: np.ndarray | None = None, bootstrap_reps: int = 40) -> dict[str, Any]:
    """Filter frames, back-project to 3D, fit the three 3D projectile laws and verify them like the 2D ones."""
    frames = np.asarray(frames, dtype=int)
    diameter = equivalent_diameter(area_px, long_px, short_px)
    check = check_frames(diameter, long_px, short_px, boxes, (width, height))
    rejected = {"blurred": check.blurred, "occluded": check.occluded}
    if check.keep.sum() < MIN_FRAMES:
        raise DepthError(f"3D was not used: only {int(check.keep.sum())} frames show the ball sharp and whole "
                         f"({check.blurred} blurred, {check.occluded} partly hidden); at least {MIN_FRAMES} are needed.")
    kept = np.column_stack([np.asarray(centers_px, dtype=float)[check.keep], diameter[check.keep]])
    seg_frames, seg = max(fill_short_gaps(frames[check.keep], kept, max_gap=MAX_GAP), key=lambda s: len(s[0]))
    if len(seg_frames) < MIN_FRAMES:
        raise DepthError(f"3D was not used: the longest unbroken stretch of usable frames is {len(seg_frames)} long; "
                         f"at least {MIN_FRAMES} are needed.")
    filled = ~np.isin(seg_frames, frames[check.keep])

    sigma_d = diameter_noise_px(seg[:, 2])
    xyz, sigma_z, sigma_lat = back_project(seg[:, :2], seg[:, 2], size_m, focal_px, (width - 1) / 2, (height - 1) / 2,
                                           sigma_d)
    # Weighted least squares through the coordinates: depth is scaled down by how much noisier it is.
    z_scale = float(min(1.0, np.median(sigma_lat) / np.median(sigma_z)))
    t = (seg_frames - seg_frames[0]) / fps
    q = xyz * np.array([1.0, 1.0, z_scale])
    coords = Coordinates("projectile3d", t, q, ("x", "y", "z"), {"z_scale": z_scale})
    fits = [fit_model(MODELS_3D[m], coords) for m in CANDIDATES]
    verdict = assess(coords, fits)
    judged = judge_candidates(verdict)

    out: dict[str, Any] = {
        "status": verdict.status, "reason": verdict.reason, "focal_px": focal_px, "diameter_m": size_m,
        "diameter_noise_px": sigma_d, "frames_used": len(seg_frames), "frames_interpolated": int(filled.sum()),
        "frames_rejected": rejected, "z_weight": z_scale,
        "depth_range_m": [float(xyz[:, 2].min()), float(xyz[:, 2].max())],
        # Depth travel against its own noise: a throw across the image moves too little in depth for drag along it to show.
        "depth_travel_vs_error": float(np.ptp(xyz[:, 2]) / np.median(sigma_z)),
        "track": [[int(f), round(float(ti), 4), *(round(float(v), 4) for v in p), round(float(sz), 4), round(float(sl), 5), bool(fl)]
                  for f, ti, p, sz, sl, fl in zip(seg_frames, t, xyz, sigma_z, sigma_lat, filled, strict=True)],
        "track_columns": ["frame", "t_s", "x_m", "y_m", "z_m", "z_err_m", "xy_err_m", "interpolated"],
        "ranking": [{"model_id": f.model_id, "title": MODELS_3D[f.model_id].title, "params": f.params, "bic": f.bic,
                     "heldout_nrmse": f.heldout_nrmse, "status": judged[f.model_id][0], "why": judged[f.model_id][1]}
                    for f in verdict.ranked],
    }
    lead = verdict.winner or (verdict.contenders[0] if verdict.contenders else None)
    if lead is not None:
        spec = MODELS_3D[lead.model_id]
        est = bootstrap(spec, coords, lead, reps=bootstrap_reps)
        out["equation"] = {"model_id": spec.id, "title": spec.title, "latex": spec.latex}
        out["parameters"] = [{"name": p.name, "latex": p.latex, "unit": p.unit, "value": est[p.name].value,
                              "std": est[p.name].std, "ci95": [est[p.name].ci_low, est[p.name].ci_high]} for p in spec.params]
        out["g"] = next(p for p in out["parameters"] if p["name"] == "g")
        # The fitted path in camera meters (depth unscaled), for checking against the measured track.
        out["fit_track"] = [[round(float(v), 4) for v in p] for p in lead.sim / np.array([1.0, 1.0, z_scale])]
    return out


def for_run(track: Any, flight_frames: np.ndarray, *, fps: float, width: int, height: int, focal_px: float | None,
            size_m: float | None, moving_object: str, family: str) -> dict[str, Any]:
    """The 3D result for one analysis, or a 'skipped' entry whose reason is shown next to the 2D result.

    track is the moving object's perception.sam_track.ObjectTrack; flight_frames the frames the 2D projectile fit
    used (one flight between bounces), so both analyses describe the same stretch of motion.
    """
    why = skip_reason(focal_px=focal_px, size_m=size_m, moving_object=moving_object, family=family)
    if why is not None:
        return {"status": "skipped", "reason": why}
    assert focal_px is not None and size_m is not None
    sel = np.isin(track.frames, flight_frames)
    pick = (lambda a: None if a is None else np.asarray(a)[sel])
    try:
        return analyze(track.frames[sel], track.centers_px[sel], track.long_px[sel], track.short_px[sel],
                       pick(track.area_px), fps=fps, width=width, height=height, focal_px=focal_px, size_m=size_m,
                       boxes=pick(track.boxes))
    except DepthError as e:
        return {"status": "skipped", "reason": str(e)}
