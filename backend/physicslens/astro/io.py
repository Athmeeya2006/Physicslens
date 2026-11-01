"""Frames with times: FITS files, image sequences and video, each frame stamped in UTC with an uncertainty.

Time sources, most trusted first:
  fits     DATE-OBS (plus half of EXPTIME for the exposure mid-point); TIMEUNC or 0.1 s unless the header says more
  gps      a video whose frames were GPS-stamped (the user says so and gives the stamp of frame 0)
  exif     DateTimeOriginal of each image (whole seconds, so 0.5 s)
  user     a start time typed by the person + frame index / fps: low trust, 2 s unless they give their own
Every later result carries the source and the uncertainty (provenance()), and timing fits add it in quadrature.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.time import Time, TimeDelta

TRUST = {"fits": "high", "gps": "high", "exif": "medium", "user": "low"}
DEFAULT_SIGMA_S = {"fits": 0.1, "gps": 0.01, "exif": 0.5, "user": 2.0}


class AstroIOError(ValueError):
    pass


@dataclass
class Frame:
    data: np.ndarray  # 2D float image (counts)
    time: Time  # mid-exposure, UTC scale
    time_sigma_s: float
    time_source: str  # fits | gps | exif | user
    header: dict[str, Any] = field(default_factory=dict)
    path: str | None = None

    @property
    def trust(self) -> str:
        return TRUST[self.time_source]

    @property
    def tdb(self) -> Time:
        """The same instant in TDB, the time scale dynamics is integrated in."""
        return self.time.tdb


def _time(value: str | Time, scale: str = "utc") -> Time:
    try:
        t = value if isinstance(value, Time) else Time(value, scale=scale)
    except ValueError as e:
        raise AstroIOError(f"not a time astropy can read: {value!r} ({e})") from e
    return t.utc


def read_fits(path: Path | str, hdu: int | None = None) -> Frame:
    """One FITS image. DATE-OBS is the exposure start unless the header says otherwise (DATE-AVG, MJD-AVG win)."""
    with fits.open(path) as hdul:
        h = next((x for x in ([hdul[hdu]] if hdu is not None else hdul) if x.data is not None and x.data.ndim == 2), None)
        if h is None:
            raise AstroIOError(f"{path}: no 2D image in the file")
        header, data = h.header, np.asarray(h.data, dtype=float)
    if "DATE-AVG" in header:
        t = _time(header["DATE-AVG"], header.get("TIMESYS", "utc").lower())
    elif "MJD-AVG" in header:
        t = Time(float(header["MJD-AVG"]), format="mjd", scale=header.get("TIMESYS", "utc").lower()).utc
    elif "DATE-OBS" in header:
        t = _time(header["DATE-OBS"], header.get("TIMESYS", "utc").lower())
        exptime = float(header.get("EXPTIME", header.get("EXPOSURE", 0.0)) or 0.0)
        t = t + TimeDelta(exptime / 2, format="sec")
    else:
        raise AstroIOError(f"{path}: no DATE-OBS, DATE-AVG or MJD-AVG: the frame has no time")
    sigma = float(header.get("TIMEUNC", DEFAULT_SIGMA_S["fits"]))
    return Frame(data, t, sigma, "fits", {k: header[k] for k in header if k}, str(path))


def read_fits_sequence(paths: Iterable[Path | str]) -> list[Frame]:
    frames = [read_fits(p) for p in paths]
    return sorted(frames, key=lambda f: f.time.jd)


def exif_time(path: Path | str) -> Time | None:
    """DateTimeOriginal (local camera time, assumed UTC unless OffsetTimeOriginal is present)."""
    from PIL import Image
    try:
        exif = Image.open(path).getexif().get_ifd(0x8769)  # the Exif sub-directory
    except OSError:
        return None
    stamp = exif.get(0x9003)  # DateTimeOriginal "YYYY:MM:DD HH:MM:SS"
    if not stamp:
        return None
    iso = stamp.replace(":", "-", 2).replace(" ", "T")
    offset = exif.get(0x9011)  # OffsetTimeOriginal "+05:30"
    t = Time(iso, scale="utc")
    if offset:
        sign = -1 if offset.startswith("-") else 1
        hh, mm = offset.lstrip("+-").split(":")
        t = t - TimeDelta(sign * (int(hh) * 3600 + int(mm) * 60), format="sec")
    return t


def read_images(paths: Iterable[Path | str], start_utc: str | None = None, interval_s: float | None = None,
                user_sigma_s: float | None = None) -> list[Frame]:
    """Image files (JPEG, PNG, TIFF) as grey frames: EXIF times where every image has one, else the typed start
    time plus a fixed interval (low trust)."""
    import cv2
    paths = [Path(p) for p in paths]
    stamps = [exif_time(p) for p in paths]
    use_exif = all(s is not None for s in stamps)
    if not use_exif and (start_utc is None or interval_s is None):
        raise AstroIOError("these images carry no EXIF time: give the start time (UTC) and the interval between them")
    frames = []
    for k, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise AstroIOError(f"{p}: not an image")
        if use_exif:
            frames.append(Frame(img.astype(float), stamps[k], DEFAULT_SIGMA_S["exif"], "exif", path=str(p)))
        else:
            t = _time(start_utc) + TimeDelta(k * float(interval_s), format="sec")
            frames.append(Frame(img.astype(float), t, user_sigma_s or DEFAULT_SIGMA_S["user"], "user", path=str(p)))
    return frames


def video_times(n_frames: int, fps: float, start_utc: str, gps: bool = False, sigma_s: float | None = None) -> tuple[Time, float, str]:
    """Frame times of a video: frame 0 at start_utc, then 1/fps apart. GPS-stamped (high trust) only when the
    person says so; otherwise a typed start time is low trust."""
    if not (fps > 0 and np.isfinite(fps)):
        raise AstroIOError("the frame rate must be a positive number")
    source = "gps" if gps else "user"
    t = _time(start_utc) + TimeDelta(np.arange(n_frames) / fps, format="sec")
    return t, float(sigma_s if sigma_s is not None else DEFAULT_SIGMA_S[source]), source


def read_video(path: Path | str, start_utc: str, fps: float | None = None, gps: bool = False,
               sigma_s: float | None = None, max_frames: int = 2000) -> list[Frame]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        rate = fps or cap.get(cv2.CAP_PROP_FPS)
        images = []
        while len(images) < max_frames:
            ok, img = cap.read()
            if not ok:
                break
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float))
    finally:
        cap.release()
    if not images:
        raise AstroIOError(f"{path}: no frames could be read")
    times, sigma, source = video_times(len(images), float(rate), start_utc, gps, sigma_s)
    return [Frame(img, times[k], sigma, source, path=str(path)) for k, img in enumerate(images)]


def provenance(frames: list[Frame], data_source: str, licence: str) -> dict[str, Any]:
    """What every astro result states about its data and its clock."""
    sources = sorted({f.time_source for f in frames})
    return {"data_source": data_source, "licence": licence, "frames": len(frames),
            "time_source": "+".join(sources), "time_trust": min((TRUST[s] for s in sources), key=["low", "medium", "high"].index),
            "time_sigma_s": float(max(f.time_sigma_s for f in frames)),
            "start_utc": frames[0].time.isot, "end_utc": frames[-1].time.isot}
