"""Video check of the SO-100 case: does the logged joint path match what the top camera saw?

  python scripts/robot_video_check.py fetch             download the pinned top-camera video and episode index (free)
  python scripts/robot_video_check.py clips             cut, re-encode (AV1 to H.264, 15 fps) and save frame 0 (free)
  python scripts/robot_video_check.py track --max-usd X PAID: SAM 3.1 on each clip, box prompt on frame 0 (cached)
  python scripts/robot_video_check.py report            fit the camera, compare, write data/robot/video_check.json (free)

Data: lerobot/svla_so100_pickplace (Apache-2.0, same pinned revision as scripts/fetch_robot.py). The top camera is
640x480 at 30 fps, AV1, every episode in one file; SAM 3.1 rejects AV1, so each episode is cut out, every second frame
kept (15 fps, the arm moves well under a pixel per millisecond) and re-encoded as H.264.

Episodes: 33 and 34 (inside the robot case's training split, 0 to 34) fit the camera; 35 to 39 (the first five of
its held-out split, 35 to 49, taken in order, not chosen) are only scored. The joint path is the arm's measured joint
angles (observation.state) through MuJoCo Menagerie's trs_so_arm100 with the robot case's joint map; the tracked
point is the gripper (body Fixed_Jaw, the moving jaw included in the mask). The video also checks that joint map:
see MAIN_VARIANT.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.robot import arm_sim as A
from physicslens.robot import video_check as V

RAW = ROOT / "var" / "robot" / "raw"
WORK = ROOT / "var" / "robot" / "video"
OUT = ROOT / "data" / "robot" / "video_check.json"
HF = "https://huggingface.co/datasets/lerobot/svla_so100_pickplace/resolve/728583b5eaf9e739a7f119e2def466fa1d552402"
VIDEO_KEY = "observation.images.top"
XML = RAW / "menagerie" / "so_arm100.xml"
DATA_FILE = RAW / "lerobot" / "data" / "chunk-000" / "file-000.parquet"
EPISODES_FILE = RAW / "lerobot" / "meta" / "episodes" / "chunk-000" / "file-000.parquet"

CAMERA_EPS = [33, 34]
HELDOUT_EPS = [35, 36, 37, 38, 39]
JOINT_MAP_EPS = list(range(35))  # the robot case's training episodes, whose first frames fix the joint map
LOG_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MJ_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
BODY = "Fixed_Jaw"
STEP = 2  # keep every second frame: 15 fps
SIZE = (640, 480)
SHIFT_GRID = np.arange(-6.0, 6.01, 1.0)  # camera-to-encoder time offsets tried, in 30 fps frames
# The camera's focal length is bounded to a webcam's: 300 to 1000 px at 640 px wide is a horizontal field of view of
# 35 to 94 degrees. With the robot case's joint map as it stands, a free fit drifts to a camera kilometres below the
# table (a mirror image); with the corrected map (MAIN_VARIANT) the free fit lands inside the bounds by itself.
# FOCAL_PROFILE shows how much the focal length changes the answer.
FOCAL_RANGE = (300.0, 1000.0)
FOCAL_PROFILE = [300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 2000.0, 5000.0]
SAM_METADATA = {"mask_encoding": "one_bit", "include_confidence": "true", "propagation_direction": "forward"}
# The gripper's box on frame 0 of each clip (left, top, right, bottom, pixels): the black jaw-and-motor block, placed
# by hand on frame 0 (written by `clips` to var/robot/video/frames/) before any paid call. Every episode starts with
# the arm folded at rest, but the rest pose differs by up to about 15 px between episodes, so each has its own box.
GRIPPER_BOX = {33: (264, 88, 362, 168), 34: (270, 95, 370, 168), 35: (272, 92, 372, 168), 36: (260, 90, 358, 165),
               37: (268, 92, 368, 168), 38: (272, 92, 372, 168), 39: (258, 92, 358, 165)}


def fetch() -> None:
    meta = pd.read_parquet(EPISODES_FILE)
    files = sorted({(int(r[f"videos/{VIDEO_KEY}/chunk_index"]), int(r[f"videos/{VIDEO_KEY}/file_index"]))
                    for _, r in meta.iterrows()})
    for ch, fi in files:
        rel = f"videos/{VIDEO_KEY}/chunk-{ch:03d}/file-{fi:03d}.mp4"
        dest = RAW / "lerobot" / rel
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"fetching {rel}", flush=True)
            tmp = dest.with_suffix(".part")
            with urllib.request.urlopen(f"{HF}/{rel}", timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
        h = hashlib.sha256()
        with open(dest, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        h = h.hexdigest()
        print(f"{h}  lerobot/{rel}  ({dest.stat().st_size} bytes)")


def episode_rows() -> dict[int, dict]:
    meta = pd.read_parquet(EPISODES_FILE)
    out = {}
    for _, r in meta.iterrows():
        e = int(r["episode_index"])
        out[e] = {"length": int(r["length"]),
                  "from_s": float(r[f"videos/{VIDEO_KEY}/from_timestamp"]),
                  "to_s": float(r[f"videos/{VIDEO_KEY}/to_timestamp"]),
                  "file": RAW / "lerobot" / f"videos/{VIDEO_KEY}/chunk-{int(r[f'videos/{VIDEO_KEY}/chunk_index']):03d}"
                          f"/file-{int(r[f'videos/{VIDEO_KEY}/file_index']):03d}.mp4"}
    return out


def clip_path(e: int) -> Path:
    return WORK / "clips" / f"episode_{e:03d}.mp4"


def count_frames(p: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(p)], capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def clips() -> None:
    rows = episode_rows()
    for e in CAMERA_EPS + HELDOUT_EPS:
        r, out = rows[e], clip_path(e)
        want = math.ceil(r["length"] / STEP)
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            # Seek to the episode, keep frames 0, 2, 4, ... of it (exact frames, not a resampling) and encode H.264.
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{r['from_s']:.6f}", "-i", str(r["file"]),
                            "-vf", f"select=not(mod(n\\,{STEP})),setpts=N/{30 // STEP}/TB", "-r", str(30 // STEP),
                            "-frames:v", str(want), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                            "-pix_fmt", "yuv420p", "-threads", "1", "-an", str(out)], check=True)
        n = count_frames(out)
        jpg = WORK / "frames" / f"episode_{e:03d}_f0.jpg"
        jpg.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out), "-frames:v", "1", str(jpg)], check=True)
        print(f"episode {e}: {r['length']} frames logged, clip {n} frames (want {want}), "
              f"video {r['from_s']:.3f} to {r['to_s']:.3f} s ({(r['to_s'] - r['from_s']) * 30:.1f} frames)")


def ledger_spend(prefix: str = "robotvid-") -> dict:
    with sqlite3.connect(ROOT / "var" / "ledger.db") as conn:
        row = conn.execute("SELECT COALESCE(SUM(est_cost_usd), 0), COALESCE(SUM(video_frames), 0), COUNT(*) FROM spend "
                           "WHERE status IN ('committed', 'failed_billed') AND run_id LIKE ?", (prefix + "%",)).fetchone()
    return {"cost": float(row[0]), "frames": int(row[1]), "calls": int(row[2])}


def track(max_usd: float) -> None:
    os.environ.setdefault("APP_ENV", "dev")
    import meta_sam_parser as msp
    from physicslens.costguard import CostGuardClient
    from physicslens.costguard.cache import file_digest
    from physicslens.settings import load_settings

    settings = load_settings()
    if settings.mock_models:
        sys.exit("track needs MOCK_MODELS=0: it calls SAM 3.1 for real.")
    client = CostGuardClient(settings)
    start = ledger_spend()["cost"]
    for e in CAMERA_EPS + HELDOUT_EPS:
        txt = WORK / "sam" / f"episode_{e:03d}.txt"
        if txt.exists():
            continue
        if ledger_spend()["cost"] - start > max_usd:
            print("budget for this run reached; stopping", flush=True)
            break
        clip = clip_path(e)
        n = count_frames(clip)
        objects = [msp.PromptObject(id=1, prompts=[msp.FramePrompt(frame=0, box=msp.PromptBox(*GRIPPER_BOX[e]))])]
        prompt = msp.build_object_prompt(size=SIZE, objects=objects)
        r = client.sam_video(prompt=prompt, video_path=clip, video_hash=file_digest(clip), est_frames=n,
                             metadata=SAM_METADATA, run_id=f"robotvid-ep{e:03d}-{int(time.time())}", audience="dev")
        txt.parent.mkdir(parents=True, exist_ok=True)
        txt.write_text(r.text)
        (txt.with_suffix(".json")).write_text(json.dumps({"finish_reason": r.finish_reason, "cached": r.cached,
                                                           "usage": r.usage.as_dict(), "frames": n}))
        print(f"episode {e}: {n} frames sent, cached={r.cached}, frames billed {r.usage.video_frames}", flush=True)
    print(json.dumps(ledger_spend()))


# ---------- report ----------


def load_states() -> dict[int, np.ndarray]:
    d = pd.read_parquet(DATA_FILE)
    out = {}
    for e, g in d.groupby("episode_index"):
        g = g.sort_values("frame_index")
        out[int(e)] = np.stack(g["observation.state"].to_numpy()).astype(float)
    return out


def joint_map(states: dict[int, np.ndarray]) -> A.JointMap:
    """The robot case's map (scripts/robot_demo.py): sign +1 and the shared rest pose."""
    m = A.load(XML)
    rest = np.mean([states[e][0] for e in JOINT_MAP_EPS], axis=0)
    return A.JointMap(MJ_JOINTS, rest, m.key("rest").qpos.copy(), np.ones(6))


def main_joint_map(states: dict[int, np.ndarray]) -> A.JointMap:
    jm = joint_map(states)
    sign = np.ones(6)
    sign[VARIANTS[MAIN_VARIANT][0]] = -1.0
    return A.JointMap(jm.names, jm.deg_ref, jm.q_ref, sign)


@functools.cache
def sam_track(e: int):
    """SAM's track of the gripper, decoded once per process (decoding the masks is the slow part)."""
    from physicslens.perception.sam_track import tracks_from_text
    meta = json.loads((WORK / "sam" / f"episode_{e:03d}.json").read_text())
    tracks = tracks_from_text((WORK / "sam" / f"episode_{e:03d}.txt").read_text(),
                              completed=meta["finish_reason"] in (None, "completed"))
    return tracks.get("1") or next(iter(tracks.values())), meta


def episode_data(e: int, states: dict, jm: A.JointMap, m, shift: float, mirror: bool = False) -> dict:
    tr, meta = sam_track(e)
    q = V.shift_frames(states[e], shift)[::STEP]  # the joint reading at each kept video frame
    pos, rot = V.fk_body(m, jm, q, BODY)
    k = tr.frames.astype(int)
    k = k[k < len(q)]
    sel = np.isin(tr.frames.astype(int), k)
    area = tr.area_px[sel]
    b = tr.boxes[sel]
    edge = ((b[:, 0] <= 1) | (b[:, 1] <= 1) | (b[:, 2] >= SIZE[0] - 1) | (b[:, 3] >= SIZE[1] - 1)).astype(float)
    uv = tr.centers_px[sel].copy()
    if mirror:  # the image read left-right reversed (a mirrored camera image is the same as a mirrored arm)
        uv[:, 0] = SIZE[0] - 1 - uv[:, 0]
    return {"edge": edge, "pos": pos[k], "rot": rot[k], "uv": uv, "frames": k, "area": area,
            "conf": tr.confidence[sel], "q": q[k], "sent": meta["frames"], "q_all": q}


def fit_on(eps: list[int], states, jm, m, shift: float,
           focal: tuple[float, float] = FOCAL_RANGE, mirror: bool = False) -> tuple[V.CameraFit, dict]:
    data = {e: episode_data(e, states, jm, m, shift, mirror) for e in eps}
    cat = lambda key: np.concatenate([data[e][key] for e in eps])
    return V.fit_camera(cat("pos"), cat("rot"), cat("uv"), SIZE, focal_px=focal), data


def score(fit: V.CameraFit, d: dict) -> dict:
    pred, depth = fit.camera.project(V.tracked_points(d["pos"], d["rot"], fit.offset))
    err = V.pixel_errors(pred, d["uv"])
    return {"pred": pred, "depth": depth, "err_px": err, "err_mm": V.mm_errors(err, depth, fit.camera.f)}


def report() -> dict:
    for e in CAMERA_EPS + HELDOUT_EPS:
        print(f"decoding SAM track of episode {e}", flush=True)
        sam_track(e)
    states = load_states()
    jm0 = joint_map(states)
    m = A.load(XML)
    jm = main_joint_map(states)
    flip, _ = VARIANTS[MAIN_VARIANT]

    # Time offset between the camera and the encoder log: chosen on the camera episodes only.
    shifts = []
    for s in SHIFT_GRID:
        fit, _ = fit_on(CAMERA_EPS, states, jm, m, float(s))
        shifts.append((float(s), fit.median_px, fit.rms_px))
    best_shift = min(shifts, key=lambda t: t[1])[0]
    fit, train = fit_on(CAMERA_EPS, states, jm, m, best_shift)

    profile = []
    for f in FOCAL_PROFILE + [None]:
        fp, _ = fit_on(CAMERA_EPS, states, jm, m, best_shift, (f, f) if f else (100.0, 1e8))
        hd = [episode_data(e, states, jm, m, best_shift) for e in HELDOUT_EPS]
        sc = [score(fp, d) for d in hd]
        px = np.concatenate([x["err_px"] for x in sc])
        mm = np.concatenate([x["err_mm"] for x in sc])
        cam_h = float(np.linalg.norm(fp.camera.as_dict()["camera_position_m"]))
        profile.append({"focal_px": f if f else "free", "fitted_focal_px": fp.camera.f,
                        "camera_distance_from_base_m": cam_h, "camera_episodes_median_px": fp.median_px,
                        "heldout_median_px": float(np.median(px)), "heldout_median_mm": float(np.median(mm)),
                        "heldout_rms_mm": float(np.sqrt(np.mean(mm**2))),
                        "mm_per_px_median": float(np.median(np.concatenate([x["depth"] for x in sc])) * 1000 / fp.camera.f)})

    per_ep, all_rows = [], []
    for e in HELDOUT_EPS:
        d = episode_data(e, states, jm, m, best_shift)
        s = score(fit, d)
        px, mm = V.summary(s["err_px"]), V.summary(s["err_mm"])
        per_ep.append({"episode": e, "frames_sent": d["sent"], "frames_tracked": len(d["frames"]),
                       "px": px, "mm": mm, "mean_offset_px": (s["pred"] - d["uv"]).mean(0).tolist(),
                       "depth_m": [float(s["depth"].min()), float(s["depth"].max())]})
        speed = np.r_[0.0, np.linalg.norm(np.diff(d["pos"], axis=0), axis=1)] * 30 / STEP  # m/s of the gripper
        all_rows.append({"err_px": s["err_px"], "err_mm": s["err_mm"], "gripper_deg": d["q"][:, 5],
                         "area_rel": d["area"] / np.median(d["area"]), "conf": d["conf"], "speed": speed,
                         "height_m": d["pos"][:, 2], "pan_deg": d["q"][:, 0], "wrist_flex_deg": d["q"][:, 3],
                         "at_image_edge": d["edge"]})
    cat = {k: np.concatenate([r[k] for r in all_rows]) for k in all_rows[0]}
    ep_med_px = [r["px"]["median"] for r in per_ep]
    ep_med_mm = [r["mm"]["median"] for r in per_ep]
    ep_rms_px = [r["px"]["rms"] for r in per_ep]
    ep_rms_mm = [r["mm"]["rms"] for r in per_ep]
    ep_p90_mm = [r["mm"]["p90"] for r in per_ep]

    e_px = cat["err_px"]
    factors = {
        "gripper_opening_deg": (cat["gripper_deg"], list(np.percentile(cat["gripper_deg"], [0, 25, 50, 75, 100]))),
        "mask_area_vs_episode_median": (cat["area_rel"], [0.0, 0.6, 0.85, 1.15, 1.5, float(max(cat["area_rel"].max(), 1.5))]),
        "gripper_speed_m_per_s": (cat["speed"], list(np.percentile(cat["speed"], [0, 50, 75, 90, 100]))),
        "gripper_height_m": (cat["height_m"], list(np.percentile(cat["height_m"], [0, 25, 50, 75, 100]))),
        "sam_confidence": (cat["conf"], list(np.percentile(cat["conf"], [0, 10, 50, 100]))),
        "mask_box_at_image_edge": (cat["at_image_edge"], [0.0, 0.5, 1.0]),
        "wrist_flex_deg": (cat["wrist_flex_deg"], list(np.percentile(cat["wrist_flex_deg"], [0, 25, 50, 75, 100]))),
    }
    where = {k: {"spearman_with_px_error": V.rank_correlation(v, e_px), "bins": V.binned(e_px, v, edges)}
             for k, (v, edges) in factors.items()}

    # Every joint-map variant, fitted on the camera episodes with the focal length free: where does the camera end up?
    alternatives = {name: variant_fit(name, states, jm0, m, best_shift, (100.0, 1e8)) for name in VARIANTS}

    spend = ledger_spend()
    train_err = np.concatenate([score(fit, train[e])["err_px"] for e in CAMERA_EPS])
    out = {
        "generated": "scripts/robot_video_check.py",
        "data": {"dataset": "lerobot/svla_so100_pickplace", "camera": "observation.images.top, 640x480",
                 "video": "AV1 at 30 fps, cut per episode, every second frame kept (15 fps), re-encoded H.264",
                 "camera_fit_episodes": CAMERA_EPS, "heldout_episodes": HELDOUT_EPS,
                 "joint_angles": "observation.state (the arm's measured joints)",
                 "tracked_point": f"gripper: MuJoCo body {BODY} plus a fitted offset; SAM mask centroid in the image"},
        "tracking": {"model": "sam-3.1", "prompt": "one box on frame 0 (build_object_prompt), propagated forward",
                     "box_frame0_px": {str(e): list(b) for e, b in GRIPPER_BOX.items()},
                     "frames_sent": int(sum(json.loads((WORK / "sam" / f"episode_{e:03d}.json").read_text())["frames"]
                                            for e in CAMERA_EPS + HELDOUT_EPS)),
                     "ledger": {"video_frames": spend["frames"], "calls": spend["calls"]}},
        "projection": {"kind": "pinhole: square pixels, no skew, no distortion, principal point at the image centre",
                       "fitted": "camera rotation (3), translation (3), focal length (1), tracked point's offset in the "
                                 "gripper body frame (3): 10 numbers, soft-L1 loss with a 5 px scale",
                       "fitted_on": CAMERA_EPS, "frames": fit.n, **fit.camera.as_dict(),
                       "offset_in_body_m": fit.offset.tolist(),
                       "camera_episodes_px": V.summary(train_err),
                       "focal_bounds_px": list(FOCAL_RANGE),
                       "focal_note": "bounded to a webcam's field of view (35 to 94 degrees horizontal); see focal_profile",
                       "focal_profile": profile,
                       "camera_distance_from_base_m": float(np.linalg.norm(fit.camera.as_dict()["camera_position_m"])),
                       "time_shift_frames_30fps": best_shift,
                       "time_shift_note": "joint reading taken this many 30 fps frames after each video frame; chosen "
                                          "on the camera episodes by median pixel error",
                       "time_shift_scan": [{"shift": s, "median_px": a, "rms_px": b} for s, a, b in shifts]},
        "heldout": {
            "per_episode": per_ep,
            "median_px": V.t_interval(ep_med_px), "rms_px": V.t_interval(ep_rms_px),
            "median_mm": V.t_interval(ep_med_mm), "rms_mm": V.t_interval(ep_rms_mm), "p90_mm": V.t_interval(ep_p90_mm),
            "interval_note": "mean over the 5 held-out episodes of each episode's statistic, Student-t 95% interval "
                             "(4 degrees of freedom); episodes, not frames, are the independent units",
            "pooled_px": V.summary(cat["err_px"]), "pooled_mm": V.summary(cat["err_mm"]),
            "mm_note": "pixel error times the point's depth over the focal length: the distance across the line of "
                       "sight; the metric scale comes from the simulator's link lengths through the fitted camera"},
        "where_it_disagrees": where,
        "joint_map": {"used": MAIN_VARIANT, "flipped_from_robot_case": [LOG_NAMES[j] for j in flip],
                      "why": "chosen on the camera episodes only: with the robot case's map (sign +1 on every joint) "
                             "the camera that fits best sits below the table looking up, which is a mirror image; "
                             "flipping the pan and wrist roll signs (or reading the image left-right reversed, which "
                             "fits identically) puts it above the table looking down, with a webcam focal length, "
                             "and no worse error. The video cannot tell those two apart",
                      "variants_free_focal": alternatives},
    }
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("projection", "heldout")}, indent=1, default=float)[:6000])
    return out


MAIN_VARIANT = "pan_and_roll_flipped"
VARIANTS = {"as_assumed": ([], False), "image_mirrored": ([], True), "pan_sign_flipped": ([0], False),
            "pan_and_roll_flipped": ([0, 4], False), "wrist_flex_sign_flipped": ([3], False),
            "wrist_roll_sign_flipped": ([4], False), "lift_elbow_flex_flipped": ([1, 2, 3], False),
            "all_arm_signs_flipped": ([0, 1, 2, 3, 4], False)}


def variant_fit(name: str, states, jm, m, shift: float, focal=FOCAL_RANGE) -> dict:
    flip, mirror = VARIANTS[name]
    sign = np.ones(6)
    sign[flip] = -1.0
    jm2 = A.JointMap(jm.names, jm.deg_ref, jm.q_ref, sign)
    f2, _ = fit_on(CAMERA_EPS, states, jm2, m, shift, focal, mirror)
    cam = f2.camera.as_dict()
    held = np.concatenate([score(f2, episode_data(e, states, jm2, m, shift, mirror))["err_px"] for e in HELDOUT_EPS])
    return {"flipped_joints": [LOG_NAMES[j] for j in flip], "image_mirrored": mirror,
            "camera_episodes_median_px": f2.median_px, "heldout_median_px": float(np.median(held)),
            "heldout_rms_px": float(np.sqrt(np.mean(held**2))), "focal_px": f2.camera.f,
            "camera_height_m": cam["camera_position_m"][2],
            "camera_looks_down": bool(cam["optical_axis_world"][2] < 0)}


def variants() -> None:
    """Exploration: each joint-map variant fitted on the camera episodes, with where the camera ends up."""
    states = load_states()
    jm = joint_map(states)
    m = A.load(XML)
    for name in VARIANTS:
        for f in (FOCAL_RANGE, (100.0, 1e8)):
            print(name, f, json.dumps(variant_fit(name, states, jm, m, -2.0, f), default=float), flush=True)


def overlays(fit: V.CameraFit | None = None, shift: float = 0.0) -> None:
    """For checking by eye (var/robot/video/check/): SAM's box and centroid in green, the projected joint path
    in red when a camera fit is given, on six frames of each clip."""
    import cv2

    states = load_states() if fit is not None else None
    jm = main_joint_map(states) if fit is not None else None
    m = A.load(XML) if fit is not None else None
    out_dir = WORK / "check"
    out_dir.mkdir(parents=True, exist_ok=True)
    for e in CAMERA_EPS + HELDOUT_EPS:
        if not (WORK / "sam" / f"episode_{e:03d}.txt").exists():
            continue
        tr, _ = sam_track(e)
        pred = None
        if fit is not None:
            d = episode_data(e, states, jm, m, shift)
            pred = dict(zip(d["frames"].tolist(), score(fit, d)["pred"].tolist(), strict=True))
        cap = cv2.VideoCapture(str(clip_path(e)))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        want = set(np.linspace(0, n - 1, 6).astype(int).tolist())
        tiles, i = [], 0
        by_frame = {int(f): k for k, f in enumerate(tr.frames)}
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i in want:
                if i in by_frame:
                    k = by_frame[i]
                    x1, y1, x2, y2 = (int(v) for v in tr.boxes[k])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.circle(img, (int(tr.centers_px[k][0]), int(tr.centers_px[k][1])), 4, (0, 255, 0), -1)
                if pred is not None and i in pred:
                    cv2.circle(img, (int(pred[i][0]), int(pred[i][1])), 4, (0, 0, 255), -1)
                cv2.putText(img, f"ep {e} f {i}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                tiles.append(cv2.resize(img, (320, 240)))
            i += 1
        rows = [np.hstack(tiles[j:j + 3]) for j in range(0, len(tiles) - len(tiles) % 3, 3)]
        cv2.imwrite(str(out_dir / f"episode_{e:03d}.jpg"), np.vstack(rows))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "fetch":
        fetch()
    elif cmd == "clips":
        clips()
    elif cmd == "track":
        track(float(sys.argv[sys.argv.index("--max-usd") + 1]))
    elif cmd == "variants":
        variants()
    elif cmd == "overlays":
        overlays()
    else:
        r = report()
        p = r["projection"]
        cam = V.Camera(np.array(p["rvec"]), np.array(p["tvec_m"]), p["focal_px"], tuple(p["principal_point_px"]))
        overlays(V.CameraFit(cam, np.array(p["offset_in_body_m"]), 0.0, 0.0, 0), p["time_shift_frames_30fps"])
