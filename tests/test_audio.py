"""The microphone as a second sensor, on synthetic sound with known answers (no paid calls)."""

import subprocess

import numpy as np
import pytest
from physicslens.perception.audio import extract_wav, load_wav, read_audio
from physicslens.sigproc.acoustics import (
    AcousticError,
    detect_onsets,
    dominant_frequency,
    fuse,
    restitution_from_onsets,
    spin_down_from_sound,
)
from physicslens.sound import audio_trail_step, listen
from scipy.integrate import solve_ivp
from scipy.io import wavfile

SR = 48_000


def bounce_sound(restitutions: list[float], first_flight: float = 0.6, t0: float = 0.25, noise: float = 0.003,
                 seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Ringing clicks at the impacts of a vertical bounce, quieter each time; returns (samples, impact times)."""
    times, t, T = [t0], t0, first_flight
    for e in [1.0, *restitutions]:
        T *= e
        t += T
        times.append(t)
    rng = np.random.default_rng(seed)
    x = rng.normal(0, noise, int((times[-1] + 0.3) * SR))
    ring = np.arange(int(0.03 * SR))
    for i, ti in enumerate(times):
        n0 = int(ti * SR)
        tt = (ring - (ti * SR - n0)) / SR
        x[n0:n0 + ring.size] += 0.6 * 0.85**i * np.sin(2 * np.pi * 3200 * tt) * np.exp(-tt / 0.004) * (tt >= 0)
    # A low hum and a voice-band murmur under the clicks: the high-pass must ignore both.
    tt = np.arange(x.size) / SR
    x += 0.05 * np.sin(2 * np.pi * 120 * tt) + 0.02 * np.sin(2 * np.pi * 300 * tt) * np.sin(2 * np.pi * 0.7 * tt)
    return x, np.array(times)


def test_bounce_clicks_recover_restitution():
    truth = [0.82, 0.8, 0.78, 0.76, 0.74, 0.72]
    x, times = bounce_sound(truth)
    on = detect_onsets(x, SR)
    assert on.times.size == times.size
    r = restitution_from_onsets(on.times, on.sigma_t)
    assert r.e.size == len(truth)
    assert np.max(np.abs(r.e - truth)) < 0.005
    # The stated errors are honest: every bounce lies within 3 of its own standard deviations.
    assert np.all(np.abs(r.e - truth) < 3 * r.e_std)
    np.testing.assert_allclose(r.impact_times, times[1:-1], atol=0.002)
    assert min(truth) < r.e_mean < max(truth)


def test_unrelated_sounds_around_the_bounces_are_left_out():
    x, _ = bounce_sound([0.8, 0.8, 0.8, 0.8])
    # A loud clap a second before the drop and one well after the ball has settled.
    lead = np.zeros(SR)
    lead[1000:1200] = np.random.default_rng(1).normal(0, 0.5, 200)
    tail = np.zeros(int(1.5 * SR))
    tail[-2000:-1800] = np.random.default_rng(2).normal(0, 0.5, 200)
    on = detect_onsets(np.concatenate([lead, x, tail]), SR)
    r = restitution_from_onsets(on.times, on.sigma_t)
    assert r.e.size == 4 and np.max(np.abs(r.e - 0.8)) < 0.005


def test_too_few_impacts_is_a_clear_error():
    with pytest.raises(AcousticError):
        restitution_from_onsets(np.array([0.5, 1.1]), 0.0005)


def test_hum_frequency_within_its_stated_interval():
    rng = np.random.default_rng(3)
    t = np.arange(int(2.5 * SR)) / SR
    x = 0.1 * np.sin(2 * np.pi * 117.3 * t) + 0.04 * np.sin(2 * np.pi * 234.6 * t + 1) + rng.normal(0, 0.05, t.size)
    tone = dominant_frequency(x, SR)
    assert tone.halfwidth_hz == pytest.approx(1 / 2.5)
    assert abs(tone.freq_hz - 117.3) <= tone.halfwidth_hz


def test_noise_has_no_tone():
    with pytest.raises(AcousticError):
        dominant_frequency(np.random.default_rng(4).normal(0, 1, SR), SR)


def chirp(c: float, k: float, q: float, f0: float = 360.0, seconds: float = 4.0, seed: int = 5) -> np.ndarray:
    """A blade-pass whine whose frequency follows df/dt = -c - k f - q f^2 (the spin-down laws on the tone)."""
    t = np.arange(int(seconds * SR)) / SR
    f = solve_ivp(lambda _t, y: [-c - k * y[0] - q * y[0] ** 2], (0, seconds), [f0], t_eval=t, rtol=1e-10).y[0]
    rng = np.random.default_rng(seed)
    return 0.2 * np.sin(2 * np.pi * np.cumsum(f) / SR) + rng.normal(0, 0.05, t.size)


@pytest.mark.parametrize("ckq,law", [
    ((40.0, 0, 0), "spin_dry"),
    ((0, 0.35, 0), "spin_viscous"),
    ((0, 0, 0.0015), "spin_air"),
])
def test_spin_down_chirp_identifies_the_friction_law(ckq, law):
    out = spin_down_from_sound(chirp(*ckq), SR)
    assert out["status"] == "accepted" and out["law"]["model_id"] == law
    assert out["rotor_params"] == []  # no blade count: the tone's own coefficients only
    truth = {"c_f": ckq[0], "k": ckq[1], "q_f": ckq[2]}
    for p in out["acoustic_params"]:
        assert p["value"] == pytest.approx(truth[p["name"]], rel=0.03)


def test_blade_count_converts_to_the_rotor():
    blades = 4
    out = spin_down_from_sound(chirp(40.0, 0, 0), SR, blade_count=blades)
    (c,) = out["rotor_params"]
    # The tone falls 40 Hz/s, so a 4-blade rotor slows by 10 turns/s per second = 20 pi rad/s^2.
    assert c["name"] == "c" and c["value"] == pytest.approx(2 * np.pi * 40.0 / blades, rel=0.03)
    assert out["rotor_start_rate_hz"] == pytest.approx(out["f_start_hz"] / blades)


def test_steady_tone_is_not_a_spin_down():
    t = np.arange(3 * SR) / SR
    with pytest.raises(AcousticError, match="nothing is spinning down"):
        spin_down_from_sound(0.2 * np.sin(2 * np.pi * 300 * t), SR)


def test_fusion_combines_only_agreeing_values():
    ok = fuse((0.80, 0.02), (0.81, 0.01))
    assert ok["agree"] and ok["fused"]["std"] < 0.01
    assert 0.80 < ok["fused"]["value"] < 0.81
    bad = fuse((0.80, 0.01), (0.60, 0.01))
    assert not bad["agree"] and bad["fused"] is None and bad["z"] > 3


def _video(path, audio: bool) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=2"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=2"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"] + (["-c:a", "aac"] if audio else []) + [str(path)]
    subprocess.run(cmd, check=True, capture_output=True)


def test_clip_without_audio_track_returns_none(tmp_path):
    clip = tmp_path / "silent.mp4"
    _video(clip, audio=False)
    assert extract_wav(clip, tmp_path / "a.wav") is None
    assert read_audio(clip, tmp_path) is None
    audio = listen(clip, tmp_path, 2.0, 1.0, "projectile", {})
    assert audio == {"available": False, "note": "The clip has no sound track."}
    assert audio_trail_step(audio) is None


def test_clip_with_audio_is_resampled_to_48k_mono(tmp_path):
    clip = tmp_path / "tone.mp4"
    _video(clip, audio=True)
    wav = extract_wav(clip, tmp_path / "a.wav")
    x, sr = load_wav(wav)
    assert sr == 48_000 and x.ndim == 1
    audio = listen(clip, tmp_path, 2.0, 1.0, "pendulum", {})
    assert audio["available"] and abs(audio["tone"]["freq_hz"] - 440) <= audio["tone"]["halfwidth_hz"]
    step = audio_trail_step(audio)
    assert step["step"] == "audio" and step["provider"] is None and step["model"] == "acoustic analysis"


def test_slow_motion_soundtrack_is_not_used(tmp_path):
    clip = tmp_path / "tone.mp4"
    _video(clip, audio=True)
    assert listen(clip, tmp_path, 2.0, 8.0, "projectile", {})["available"] is False


def test_bounce_restitution_compared_with_the_camera(tmp_path):
    x, _ = bounce_sound([0.8, 0.8, 0.8, 0.8, 0.8])
    wav = tmp_path / "b.wav"
    wavfile.write(wav, SR, (x / np.abs(x).max() * 0.9 * 32767).astype(np.int16))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=size=64x48:rate=30:duration={x.size / SR:.2f}",
                    "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", "-shortest",
                    str(tmp_path / "bounce.mov")], check=True, capture_output=True)
    clip = tmp_path / "bounce.mov"
    camera = {"bounces": {"e_mean": 0.79, "e_std": 0.01}}
    audio = listen(clip, tmp_path, x.size / SR, 1.0, "projectile", camera)
    r = audio["restitution"]
    assert abs(r["e_mean"] - 0.8) < 0.005
    (cmp,) = audio["comparisons"]
    assert cmp["quantity"] == "restitution" and cmp["agree"] and cmp["fused"] is not None
    step = audio_trail_step(audio)
    assert step["items"][0]["status"] == "accepted"
