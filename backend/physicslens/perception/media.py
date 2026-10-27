"""Local video handling with ffmpeg/ffprobe/OpenCV. No network."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
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
           "-an", "-vf", f"fps={fps:.6f},{scale}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
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


def keyframes(path: Path, count: int, long_side: int) -> list[bytes]:
    """Up to `count` JPEGs spread over the clip, the first being frame 0 (the frame SAM prompts refer to)."""
    info = probe(path)
    idx = np.linspace(0, max(info.frames - 1, 0), num=min(count, max(info.frames, 1))).round().astype(int)
    out = []
    for i in dict.fromkeys(idx.tolist()):
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
