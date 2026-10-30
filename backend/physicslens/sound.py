"""The microphone as a second, independent sensor for a recorded run: what the soundtrack measures, and whether
it agrees with the camera. Local math only; any problem here becomes a note, never a failed analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .perception.audio import read_audio
from .physics.coords import Coordinates
from .physics.fit import noise_sigma
from .sigproc.acoustics import (
    AcousticError,
    detect_onsets,
    dominant_frequency,
    frequency_ridge,
    fuse,
    mean_rate,
    restitution_from_onsets,
    spin_down_from_sound,
)
from .sigproc.cleanup import differentiate

log = logging.getLogger("physicslens.sound")

MAX_ONSETS_SHOWN = 40


def unavailable(note: str) -> dict[str, Any]:
    return {"available": False, "note": note}


def listen(video: Path, work_dir: Path, seconds: float, slow_motion: float, family: str, result: dict[str, Any],
           blade_count: int | None = None, rotation: Coordinates | None = None,
           rotation_frames: np.ndarray | None = None, fps: float | None = None) -> dict[str, Any]:
    """result["audio"] for a finished run. Never raises: the camera result must not depend on the soundtrack."""
    try:
        return _listen(video, work_dir, seconds, slow_motion, family, result, blade_count, rotation, rotation_frames, fps)
    except Exception as e:  # audio is optional evidence: any failure is logged and shown as a note, never fatal
        log.exception("Audio analysis failed")
        return unavailable(f"The soundtrack could not be analyzed ({type(e).__name__}: {str(e)[:120]}).")


def _listen(video: Path, work_dir: Path, seconds: float, slow_motion: float, family: str, result: dict[str, Any],
            blade_count: int | None, rotation: Coordinates | None, rotation_frames: np.ndarray | None,
            fps: float | None) -> dict[str, Any]:
    if slow_motion != 1.0:
        # Phones slow the soundtrack down with the picture, or drop it; either way its clock is not real time.
        kind = "Slow-motion" if slow_motion > 1 else "Time-lapse"
        return unavailable(f"{kind} clip: its soundtrack does not run in real time, so it was not used.")
    audio = read_audio(video, work_dir, 0.0, seconds)
    if audio is None:
        return unavailable("The clip has no sound track.")
    x, sr = audio
    if x.size == 0 or not np.any(x):
        return unavailable("The sound track is silent.")
    out: dict[str, Any] = {"available": True, "sample_rate": sr, "duration_s": x.size / sr, "notes": [],
                           "onsets": [], "restitution": None, "tone": None, "spin_down": None, "comparisons": []}

    onsets = detect_onsets(x, sr)
    out["onsets"] = [round(float(t), 5) for t in onsets.times[:MAX_ONSETS_SHOWN]]
    out["onset_count"] = int(onsets.times.size)
    out["onset_sigma_s"] = onsets.sigma_t

    if family == "projectile" and onsets.times.size >= 3:
        try:
            r = restitution_from_onsets(onsets.times, onsets.sigma_t)
            out["restitution"] = {
                "e_mean": r.e_mean, "e_std": r.e_std_mean,
                "impacts": [{"t": float(t), "e": float(e), "e_std": float(s)}
                            for t, e, s in zip(r.impact_times, r.e, r.e_std, strict=True)],
            }
        except AcousticError as e:
            out["notes"].append(f"Impacts: {e}")
        bounces = result.get("bounces")
        if out["restitution"] and bounces and bounces.get("e_std") is not None:
            out["comparisons"].append({"quantity": "restitution", "label": "Coefficient of restitution e", "unit": "",
                                       **fuse((bounces["e_mean"], bounces["e_std"]),
                                              (out["restitution"]["e_mean"], out["restitution"]["e_std"]))})

    try:
        tone = dominant_frequency(x, sr)
        out["tone"] = {"freq_hz": tone.freq_hz, "halfwidth_hz": tone.halfwidth_hz, "snr_db": tone.snr_db}
    except AcousticError as e:
        out["notes"].append(f"Tone: {e}")

    if family == "rotation":
        try:
            out["spin_down"] = spin_down_from_sound(x, sr, blade_count)
        except AcousticError as e:
            out["notes"].append(f"Spin-down: {e}")
        if out["spin_down"] is not None:
            _compare_spin(out, result, blade_count, x, sr, rotation, rotation_frames, fps)
    if blade_count and out["tone"]:
        out["tone"]["rotor_rate_hz"] = out["tone"]["freq_hz"] / blade_count
    return out


def _compare_spin(out: dict[str, Any], result: dict[str, Any], blade_count: int | None, x: np.ndarray, sr: int,
                  rotation: Coordinates | None, rotation_frames: np.ndarray | None, fps: float | None) -> None:
    sd = out["spin_down"]
    camera_law = (result.get("equation") or {}).get("model_id")
    if sd.get("law") and camera_law:
        sd["camera_law"] = camera_law
        sd["laws_agree"] = camera_law == sd["law"]["model_id"]
    if not blade_count or rotation is None or rotation_frames is None or not fps:
        return
    # Mean rotation rate over the stretch both sensors cover, on the clip's clock (the soundtrack starts with frame 0).
    t_cam = np.asarray(rotation_frames, dtype=float) / fps
    a, b = max(t_cam[0], sd["t_start"]), min(t_cam[-1], sd["t_end"])
    if b - a < 0.5:
        out["notes"].append("Spin rate: the camera and the microphone overlap for less than half a second.")
        return
    i, j = int(np.searchsorted(t_cam, a)), int(np.searchsorted(t_cam, b, side="right")) - 1
    phi = rotation.q[:, 0]
    smooth = differentiate(phi[:, None], fps).pos[:, 0]
    sigma_phi = noise_sigma(phi, smooth, fps)
    span = t_cam[j] - t_cam[i]
    cam_rate = abs(smooth[j] - smooth[i]) / span / (2 * np.pi)
    cam_std = np.sqrt(2) * sigma_phi / span / (2 * np.pi)
    ridge = frequency_ridge(x, sr)
    f_mean, f_std = mean_rate(ridge.t, ridge.f, t_cam[i], t_cam[j], ridge.resolution_hz)
    out["comparisons"].append({"quantity": "rotation_rate", "label": f"Mean spin rate, {t_cam[i]:.1f} to {t_cam[j]:.1f} s",
                               "unit": "turns/s", **fuse((float(cam_rate), float(cam_std)),
                                                         (f_mean / blade_count, f_std / blade_count))})


def audio_trail_step(audio: dict[str, Any]) -> dict[str, Any] | None:
    """A decision-trail entry, only when the microphone measured something."""
    if not audio.get("available"):
        return None
    bits: list[str] = []
    if audio.get("restitution"):
        r = audio["restitution"]
        bits.append(f"From {len(r['impacts']) + 2} impact sounds, flight-time ratios give e = {r['e_mean']:.3f} ± {r['e_std']:.3f} "
                    "(no scale or g needed).")
    elif audio.get("onset_count"):
        bits.append(f"Heard {audio['onset_count']} impact sounds.")
    if audio.get("spin_down") and audio["spin_down"].get("law"):
        sd = audio["spin_down"]
        bits.append(f"The tone falls from {sd['f_start_hz']:.1f} to {sd['f_end_hz']:.1f} Hz; the same spin-down fit "
                    f"picks {sd['law']['title'].lower()}.")
    if audio.get("tone"):
        t = audio["tone"]
        bits.append(f"Strongest steady tone {t['freq_hz']:.2f} ± {t['halfwidth_hz']:.2f} Hz.")
    items = []
    for c in audio.get("comparisons", []):
        cam, mic = c["camera"], c["microphone"]
        verdict = (f"agree within {c['z']:.1f} sigma" if c["agree"] else f"disagree by {c['z']:.1f} sigma, so both are reported")
        items.append({"model_id": c["quantity"], "title": c["label"], "status": "accepted" if c["agree"] else "rejected",
                      "why": f"camera {cam['value']:.3f} ± {cam['std']:.3f}, microphone {mic['value']:.3f} ± {mic['std']:.3f}: {verdict}"})
    if not bits and not items:
        return None
    return {"step": "audio", "provider": None, "model": "acoustic analysis", "text": " ".join(bits), "items": items}
