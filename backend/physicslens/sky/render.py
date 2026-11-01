"""One renderer for every orbit video PhysicsLens watches (the Apophis, Halley and Jupiter demos and the scenarios).

Each body is a disc drawn at 8x and shrunk with area averaging, so positions are sub-pixel accurate; the video is
H.264, yuv420p, crf 16, fast start (what SAM 3.1 accepts and tracks well). render.json records everything needed to
turn pixels back into the run's units: start time, frame step, pixels per unit, the origin's pixel and the plane basis.
A body can be simulated but hidden from the video (Le Verrier's case: Neptune pulls, only Uranus is drawn).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SUPERSAMPLE = 8
BACKGROUND = (18, 14, 10)  # BGR
TEXT = (200, 200, 200)
NOTE = (150, 150, 150)
TRACKING_NOISE_PX = 2.0  # SAM 3.1 centroid error measured on the rendered Apophis video (README, orbit mode)


def plane_basis(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Rows e1, e2, n: n along the mean angular momentum of the track, e1 the frame's x axis projected into the
    plane, so the orbit is seen face on."""
    h = np.cross(pos, vel).mean(axis=0)
    n = h / np.linalg.norm(h)
    e1 = np.array([1.0, 0.0, 0.0]) - n[0] * n
    e1 /= np.linalg.norm(e1)
    return np.vstack([e1, np.cross(n, e1), n])


@dataclass
class Drawn:
    name: str
    px: np.ndarray  # (frames, 2) pixel centres, or (2,) for a body fixed in the frame
    radius_px: float
    colour: tuple[int, int, int]  # BGR
    hidden: bool = False  # simulated, pulling, but not drawn


@dataclass
class Frame:
    size: int
    fps: int
    captions_top: list[str]  # one per frame
    captions_bottom: list[str] = field(default_factory=list)  # fixed lines, last one at the bottom


def project(pos: np.ndarray, basis: np.ndarray, px_per_unit: float, origin_px: tuple[float, float]) -> np.ndarray:
    """3D positions to pixels in the plane: x right, y up (rows grow downward)."""
    xy = np.asarray(pos) @ basis[:2].T
    return np.column_stack([origin_px[0] + xy[:, 0] * px_per_unit, origin_px[1] - xy[:, 1] * px_per_unit])


def render_frames(bodies: list[Drawn], frame: Frame) -> list[np.ndarray]:
    import cv2
    count = len(frame.captions_top)
    size, sub = frame.size, SUPERSAMPLE
    out = []
    for k in range(count):
        big = np.zeros((size * sub, size * sub, 3), np.uint8)
        big[:] = BACKGROUND
        for b in bodies:
            if b.hidden:
                continue
            c = b.px if b.px.ndim == 1 else b.px[k]
            cv2.circle(big, (int(c[0] * sub), int(c[1] * sub)), int(b.radius_px * sub), b.colour, -1, cv2.LINE_AA)
        img = cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA)
        cv2.putText(img, frame.captions_top[k], (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT, 2, cv2.LINE_AA)
        for j, line in enumerate(reversed(frame.captions_bottom)):
            cv2.putText(img, line, (24, size - 24 - 30 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.62, NOTE, 1, cv2.LINE_AA)
        out.append(img)
    return out


def write_video(frames: list[np.ndarray], out: Path, fps: int, threads: int | None = None) -> Path:
    import cv2
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_name(out.stem + "_raw.mp4")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for img in frames:
        vw.write(img)
    vw.release()
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16"]
    cmd += ["-threads", str(threads)] if threads else []
    subprocess.run([*cmd, "-movflags", "+faststart", str(out)], check=True)
    raw.unlink()
    return out


def render(bodies: list[Drawn], frame: Frame, out: Path, meta: dict, threads: int | None = None,
           meta_path: Path | None = None) -> Path:
    """Draw, encode, and write the geometry the analysis needs (<out>.json unless meta_path says otherwise)."""
    write_video(render_frames(bodies, frame), out, frame.fps, threads)
    meta = {**meta, "frames": len(frame.captions_top), "fps": frame.fps}
    if any(b.hidden for b in bodies):
        meta["hidden"] = [b.name for b in bodies if b.hidden]
    (meta_path or out.with_suffix(".json")).write_text(json.dumps(meta, indent=1, default=float))
    return out


@dataclass(frozen=True)
class Detectability:
    max_px: float
    noise_px: float
    visible: bool
    message: str


def detectability(base_px: np.ndarray, full_px: np.ndarray, noise_px: float = TRACKING_NOISE_PX) -> Detectability:
    """Whether a video tracker could see a perturbation: the largest separation (px) between the target's path
    without it and with it, against the tracker's noise. Visible means at least 3 times the noise, the level the
    audit and the fits treat as a real signal; anything less a video cannot show, whatever the physics."""
    sep = np.linalg.norm(np.asarray(full_px) - np.asarray(base_px), axis=1)
    worst = float(sep.max()) if len(sep) else 0.0
    visible = worst >= 3 * noise_px
    if visible:
        msg = (f"The perturbation moves the dot by up to {worst:.1f} px over this clip; the tracker's noise is about "
               f"{noise_px:.0f} px: a video can see it.")
    else:
        need = 3 * noise_px / max(worst, 1e-9)
        times = f"{need:.1f}" if need < 10 else f"{need:,.0f}"
        msg = (f"The perturbation moves the dot by at most {worst:.2f} px over this clip; the tracker's noise is about "
               f"{noise_px:.0f} px: a video cannot see it. It would take about {times} times the zoom (or a clip long "
               f"enough for the drift to grow that much).")
    return Detectability(worst, noise_px, visible, msg)
