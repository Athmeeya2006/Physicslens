"""The clip's soundtrack as mono 48 kHz samples, with ffmpeg. No network."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .media import MediaError

SAMPLE_RATE = 48_000


def has_audio(path: Path) -> bool:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                              "-of", "csv=p=0", str(path)], check=True, capture_output=True, text=True, timeout=30)
    except subprocess.CalledProcessError as e:
        raise MediaError(f"ffprobe failed: {e.stderr.strip()[:200]}") from e
    return bool(out.stdout.strip())


def extract_wav(video: Path, dst: Path, start_s: float = 0.0, seconds: float | None = None) -> Path | None:
    """Mono 48 kHz 16-bit WAV of the clip's sound, or None when the clip has no audio track (screen recordings,
    many exported clips). start_s and seconds match the trim applied to the video, so both share one clock."""
    if not video.exists() or not has_audio(video):
        return None
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start_s:.3f}", "-i", str(video)]
    if seconds is not None:
        cmd += ["-t", f"{seconds:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
    except subprocess.CalledProcessError as e:
        raise MediaError(f"ffmpeg could not extract the sound: {e.stderr.strip()[:200]}") from e
    return dst


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Samples scaled to [-1, 1] and the sample rate."""
    sr, data = wavfile.read(path)
    x = data.astype(float)
    if data.dtype == np.int16:
        x /= 32768.0
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, int(sr)


def read_audio(video: Path, work_dir: Path, start_s: float = 0.0, seconds: float | None = None) -> tuple[np.ndarray, int] | None:
    wav = extract_wav(video, work_dir / "audio.wav", start_s, seconds)
    return None if wav is None else load_wav(wav)
