"""Why the gallery pendulum's g comes out 1% high: each candidate cause measured, not argued.

  python scripts/pendulum_check.py

Reads the stored gallery result (gallery/gallery-pendulum/result.json, a record of the real run, unchanged) and the
original Wikimedia Commons clip (fetched once into var/pendulum_check/, CC BY-SA 4.0), and writes
data/pendulum/check.json. Local math only, no model calls. The causes checked, from the Nov 2 review:

1. the truth itself: local gravity at the filming site (Aachen), not 9.81;
2. the clock: the container's per-frame timestamps, and the analog stopwatch standing in the frame, read as an
   independent clock by tracking its sweep hand;
3. the length: what length would make the measurement agree, and the physical-pendulum correction for the bob's size;
4. the swing's amplitude: the size of the error a small-angle model would have made (the fitted law is the full
   nonlinear one, so this is a bound, not a correction);
5. the scale reference out of the swing plane: not applicable here (no reference object; the scale comes from the
   stated length), recorded as such.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.physics.gravity import AACHEN, SOURCE, local_g

URL = "https://commons.wikimedia.org/wiki/Special:FilePath/Schwingung_(Definition).webm"
CACHE = ROOT / "var" / "pendulum_check" / "source.webm"
RESULT = ROOT / "gallery" / "gallery-pendulum" / "result.json"
OUT = ROOT / "data" / "pendulum" / "check.json"
STATED_LENGTH_M = 0.5
# The stopwatch (a Junghans lab timer, 60 s per turn of the sweep hand on its seconds dial) in the 1920x1080 original:
# dial centre, and the stretch where the hand runs and the pendulum's own yellow string stays clear of the dial.
DIAL = (1437, 490)
HAND_WINDOW_S = (16.5, 28.5)
DIAL_DEG_PER_S = 6.0


def fetch() -> Path:
    if not CACHE.is_file():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(URL, headers={"User-Agent": "PhysicsLens research (github.com/Athmeeya2006/Physicslens)"})
        with urllib.request.urlopen(req, timeout=120) as r:
            CACHE.write_bytes(r.read())
    return CACHE


def frame_timing(path: Path) -> dict:
    """Per-frame timestamps from the container: a variable frame rate or a 29.97/30 mix-up would show here."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    t = np.sort(np.array([float(x) for x in out.split() if x.strip()]))
    d = np.diff(t)
    rate = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                           "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.strip()
    return {"declared_rate": rate, "frames": len(t), "interval_mean_s": float(d.mean()),
            "interval_min_s": float(d.min()), "interval_max_s": float(d.max())}


def hand_angles(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The sweep hand's axis angle per frame: the yellow pixels in a ring around the dial centre lie on one line through
    it, so the doubled-angle mean gives the hand's direction without needing to tell its tip from its tail."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    yy, xx = np.mgrid[0:1080, 0:1920]
    dx, dy = xx - DIAL[0], yy - DIAL[1]
    r = np.hypot(dx, dy)
    ang = np.arctan2(dx, -dy)
    ring = (r > 80) & (r < 330) & (xx > 1150)  # left of x = 1150 the pendulum's string can enter
    ts, an, i = [], [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = i / fps
        i += 1
        if not HAND_WINDOW_S[0] <= t <= HAND_WINDOW_S[1]:
            continue
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        m = ring & (hsv[..., 0] > 15) & (hsv[..., 0] < 38) & (hsv[..., 1] > 80) & (hsv[..., 2] > 50)
        if m.sum() < 50:
            continue
        ts.append(t)
        an.append(np.angle(np.exp(2j * ang[m]).mean()) / 2)
    return np.array(ts), np.degrees(np.unwrap(np.array(an) * 2) / 2)


def hand_rate(t: np.ndarray, a: np.ndarray, seed: int = 0) -> dict:
    """Degrees per video second with a 95% moving-block bootstrap (neighbouring frames' errors are correlated)."""
    slope = float(np.polyfit(t, a, 1)[0])
    rng, n, block = np.random.default_rng(seed), len(t), 25
    boots = []
    for _ in range(2000):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, n - block, n // block + 1)])[:n]
        boots.append(np.polyfit(t[idx], a[idx], 1)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"deg_per_video_s": slope, "ci95": [float(lo), float(hi)], "frames": n,
            "rms_residual_deg": float(np.std(a - np.polyval(np.polyfit(t, a, 1), t)))}


def main() -> None:
    res = json.loads(RESULT.read_text())
    gp = next(p for p in res["agreed_parameters"] if p["name"] == "g")
    g_meas, (lo, hi) = gp["value"], gp["ci95"]
    g_true = local_g(AACHEN)
    src = fetch()

    timing = frame_timing(src)
    t, a = hand_angles(src)
    hand = hand_rate(t, a)
    # g scales as 1/T^2: if one video second were k real seconds, the true g would be g_meas / k^2. The k that would
    # remove the whole overshoot, and the k the stopwatch reports.
    k_needed = float(np.sqrt(g_meas / g_true))
    # The hand turns DIAL_DEG_PER_S per real second, so a video second spans (measured rate / dial rate) real seconds.
    k_hand = hand["deg_per_video_s"] / DIAL_DEG_PER_S
    k_hand_ci = [hand["ci95"][0] / DIAL_DEG_PER_S, hand["ci95"][1] / DIAL_DEG_PER_S]

    # The pendulum's g is the stated length times the measured omega^2, so the length that would agree with local g:
    length_needed = STATED_LENGTH_M * g_true / g_meas
    px_per_m = res["fit_state"]["px_per_m"]
    bob_radius = res["object_size_px"] / px_per_m / 2
    physical = 0.4 * bob_radius**2 / STATED_LENGTH_M  # L_eff = L + (2/5) r^2 / L for a solid sphere
    theta0 = abs(res["fit_state"]["y0"][0])
    small_angle_bias = theta0**2 / 8  # T grows by theta0^2/16, so a small-angle fit's g would be off by about twice that

    out = {
        "generated": "scripts/pendulum_check.py",
        "truth": {"g": g_true, "site": AACHEN.name, "lat_deg": AACHEN.lat_deg, "height_m": AACHEN.height_m,
                  "site_note": AACHEN.note, "source": SOURCE},
        "measured": {"g": g_meas, "ci95": [lo, hi], "relative_error": g_meas / g_true - 1,
                     "truth_in_interval": bool(lo <= g_true <= hi)},
        "clock": {"container": timing, "stopwatch_hand": hand, "dial_deg_per_s": DIAL_DEG_PER_S,
                  "real_s_per_video_s": k_hand, "real_s_per_video_s_ci95": k_hand_ci,
                  "needed_to_explain": k_needed,
                  "excluded": bool(not k_hand_ci[0] <= k_needed <= k_hand_ci[1]),
                  "note": "the hand angle is read from yellow pixels and can be pulled by the pendulum's yellow string "
                          "near the dial; taken at face value the reading points the other way (a camera running fast "
                          "would make g still higher), and a 2% clock error is far beyond a mechanical stopwatch's or "
                          "a camera's, so the reading bounds the clock rather than measuring it"},
        "length": {"stated_m": STATED_LENGTH_M, "needed_m": length_needed,
                   "shorter_by_mm": (STATED_LENGTH_M - length_needed) * 1000,
                   "bob_radius_m": bob_radius, "physical_pendulum_correction_mm": physical * 1000,
                   "physical_pendulum_correction_relative": physical / STATED_LENGTH_M},
        "amplitude": {"theta0_rad": theta0, "small_angle_bias_relative": small_angle_bias,
                      "model": "full nonlinear pendulum (sin theta), so no amplitude bias in the fit"},
        "reference_plane": "not applicable: no reference object in the clip; the scale comes from the stated length",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
