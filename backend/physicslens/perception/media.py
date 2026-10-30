"""Local video handling with ffmpeg/ffprobe/OpenCV. No network."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    width: int  # displayed size, after any rotation metadata
    height: int
    fps: float
    duration_s: float
    frames: int
    codec: str


def probe(path: Path) -> VideoInfo:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,duration,nb_read_packets,codec_name:stream_side_data=rotation",
             "-show_entries", "format=duration", "-of", "json", str(path)],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        raise MediaError(f"Could not read video: {e.stderr.strip()[:200]}") from e
    data = json.loads(out.stdout)
    if not data.get("streams"):
        raise MediaError("No video stream found.")
    s = data["streams"][0]
    rate = s.get("avg_frame_rate") if s.get("avg_frame_rate") not in (None, "0/0") else s.get("r_frame_rate")
    fps = float(Fraction(rate))
    rotation = next((int(sd.get("rotation", 0)) for sd in s.get("side_data_list", []) if "rotation" in sd), 0)
    w, h = int(s["width"]), int(s["height"])
    if abs(rotation) % 180 == 90:
        w, h = h, w
    duration = float(s.get("duration") or data.get("format", {}).get("duration") or 0)
    return VideoInfo(w, h, fps, duration, int(s.get("nb_read_packets") or round(duration * fps)), s.get("codec_name", ""))


def prepare_sam_clip(src: Path, dst: Path, *, max_fps: float, max_seconds: float, start_s: float = 0.0,
                     max_long_side: int = 1080) -> VideoInfo:
    """Re-encode to what SAM accepts and bills least for: H.264 MP4, rotation applied, capped fps/length/size.

    Re-encoding also normalizes phone rotation metadata, so frame pixels and prompt coordinates agree.
    """
    info = probe(src)
    fps = min(info.fps, max_fps)
    scale = f"scale='if(gt(iw,ih),min({max_long_side},iw),-2)':'if(gt(iw,ih),-2,min({max_long_side},ih))'"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start_s:.3f}", "-i", str(src), "-t", f"{max_seconds:.3f}",
           "-an", "-vf", f"fps={fps:.6f},{scale}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           # 0 = ffmpeg's choice (every core); batch jobs set fewer so several encodes don't swamp the machine.
           "-threads", os.environ.get("FFMPEG_THREADS", "0"), str(dst)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise MediaError(f"ffmpeg failed: {e.stderr.strip()[:200]}") from e
    return probe(dst)


def read_frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise MediaError(f"Could not read frame {index}.")
    return frame


def keyframe_indices(frames: int, count: int) -> list[int]:
    """Frame numbers of the keyframes, in order; the first is frame 0."""
    idx = np.linspace(0, max(frames - 1, 0), num=min(count, max(frames, 1))).round().astype(int)
    return list(dict.fromkeys(idx.tolist()))


def motion_energy(path: Path, size: int = 64) -> np.ndarray | None:
    """Mean absolute change between consecutive frames, on small grey copies: where in the clip something moves."""
    cap = cv2.VideoCapture(str(path))
    out: list[float] = []
    prev = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            s = size / max(h, w)
            g = cv2.cvtColor(cv2.resize(frame, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None:
                out.append(float(np.mean(np.abs(g - prev))))
            prev = g
    finally:
        cap.release()
    return np.array(out) if out else None


# Motion confined to less than this share of the clip gets the keyframes; beyond it, even spacing already sees it.
ACTIVE_SPAN_MAX = 0.6


def clip_keyframe_indices(path: Path, count: int) -> list[int]:
    """Frame 0 (the opening view, and SAM's default start), then the rest spread over the span where things move.

    In phone videos and benchmark clips the object often sits still for seconds and moves for one: evenly spaced
    keyframes then show only the still scene, and the reasoning model names whatever is in view (a real clip: a
    still red ball instead of the orange one that rolls past for a second). Camera shake moves every frame, so the
    active span covers the clip and the spacing stays even.
    """
    frames = probe(path).frames
    even = keyframe_indices(frames, count)
    energy = motion_energy(path)
    if energy is None or len(energy) < 2 * count or count < 2:
        return even
    e = energy - float(np.median(energy))
    if e.max() <= 0:
        return even
    active = np.flatnonzero(e > 0.25 * e.max())
    lo, hi = int(active[0]), int(active[-1]) + 1
    if hi - lo > ACTIVE_SPAN_MAX * frames:
        return even
    # A brief impact is a few frames: widen to show the moments around it, not four copies of one instant.
    width = max(3 * count, round(0.15 * frames))
    if hi - lo < width:
        mid = (lo + hi) // 2
        lo, hi = max(0, mid - width // 2), min(frames - 1, mid + width // 2)
    picks = np.linspace(lo, min(hi, frames - 1), num=count - 1).round().astype(int)
    return list(dict.fromkeys([0, *picks.tolist()]))


def keyframes(path: Path, count: int, long_side: int) -> list[bytes]:
    """Up to `count` JPEGs, the first being frame 0 (the frame SAM prompts refer to); see clip_keyframe_indices."""
    out = []
    for i in clip_keyframe_indices(path, count):
        frame = read_frame(path, i)
        h, w = frame.shape[:2]
        s = min(1.0, long_side / max(h, w))
        if s < 1.0:
            frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise MediaError("JPEG encoding failed.")
        out.append(buf.tobytes())
    return out


def save_prepared(clip: Path, info: VideoInfo, frames_jpeg: list[bytes], out_dir: Path) -> None:
    """Freeze a prepared clip and its keyframes, so another machine analyzes exactly these bytes.

    Cache keys hash the clip and keyframe bytes, and ffmpeg/OpenCV builds differ between machines, so a GPU
    server replaying cached SAM results must start from the same files, not re-encode its own.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(clip, out_dir / "clip.mp4")
    for i, jpg in enumerate(frames_jpeg):
        (out_dir / f"keyframe_{i}.jpg").write_bytes(jpg)
    (out_dir / "info.json").write_text(json.dumps(asdict(info)))


def load_prepared(src_dir: Path, run_dir: Path) -> tuple[Path, VideoInfo, list[bytes]]:
    clip = run_dir / "clip.mp4"
    shutil.copyfile(src_dir / "clip.mp4", clip)
    info = VideoInfo(**json.loads((src_dir / "info.json").read_text()))
    frames = [p.read_bytes() for p in sorted(src_dir.glob("keyframe_*.jpg"), key=lambda p: int(p.stem.split("_")[1]))]
    return clip, info, frames


def frame_rate_varies(path: Path, tolerance: float = 0.25, max_share: float = 0.05) -> bool:
    """Whether the source video's frame intervals vary (phones often record a variable frame rate).

    The analyzed clip is resampled to a constant rate, which can shift a frame's time by half a frame; that is
    harmless for fitting a law but matters when judging whether motion is physical at all.
    """
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                              "-of", "csv=p=0", str(path)], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise MediaError(f"Could not read frame times: {e.stderr.strip()[:200]}") from e
    times = np.sort([float(x) for x in out.stdout.split() if x.strip() not in ("", "N/A")])
    steps = np.diff(times)
    if len(steps) < 10:
        return False
    med = float(np.median(steps))
    return med > 0 and float(np.mean(np.abs(steps - med) > tolerance * med)) > max_share
