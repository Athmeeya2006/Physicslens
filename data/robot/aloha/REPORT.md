# ALOHA arms: a simulator corrected from public recordings

Status: **proven**. Every number is in `result.json` and `transfer.json`, written by `python scripts/aloha_demo.py`
(after `python scripts/aloha_demo.py fetch`) and `python scripts/aloha_demo.py transfer` (after `fetch-transfer`). Local
math only: no hardware, no model calls.

## Data, model, method

- **Data:** LeRobot `lerobot/aloha_static_coffee` (MIT, revision `b144896feb`): 50 teleoperated episodes,
  55000 frames at 50 per second; per frame the leader arms' joint angles (command) and the follower arms' (measured), radians.
- **Simulator:** MuJoCo Menagerie `aloha` (commit `fb36886b30`, BSD-3-Clause, Copyright (c) 2023 Trossen Robotics; `LICENSE-menagerie-aloha`).
  Its actuator gains, torque limits, joint damping, armature and friction were already fitted by its authors to sinusoidal
  test motions on a real ALOHA.
- **Split by time:** the first half fits every candidate law, the next fifth chooses, the first 70% refits the winner, the
  last 30% (15 episodes) is used once, for the comparison.
- **Candidates and selection:** the SO-100 case's servo laws and rule (`data/robot/REPORT.md`), on all twelve arm joints
  (the grippers' command and state are in different units and are left as published).
- **MJCF rule:** each joint's laws are written as actuator `kp`, `kv`, `delay` and joint `frictionloss` (its damping moved
  into `kv`). A joint whose MJCF form tracks worse than the published actuator on the training episodes keeps the
  published one.

## Identified laws (95% intervals: bootstrap over training episodes) and held-out error per joint (deg RMS)

| Joint | Law | Parameters | In the MJCF | Published to corrected |
|---|---|---|---|---|
| left/waist | first-order lag with delay, deadband, angle-dependent offset | tau 39.2 ms [33.8, 44.5], d 48 ms [44.1, 52.9], db 0.156 deg [0.141, 0.165], b 0.562 deg [-3.26, 3.32] | corrected | 0.20 to 0.13 |
| left/shoulder | first-order lag with delay, deadband, angle-dependent offset | tau 34.5 ms [32.3, 37.2], d 44 ms [41.4, 46.4], db 0.319 deg [0.301, 0.337], b -0.498 deg [-0.771, -0.306] | published (MJCF form tracked worse on training) | 0.22 to 0.22 |
| left/elbow | first-order lag with delay, deadband, angle-dependent offset | tau 43.6 ms [40.6, 45.6], d 34.9 ms [33, 37.6], db 0.341 deg [0.331, 0.35], b 2.8 deg [2.46, 3.35] | published (MJCF form tracked worse on training) | 0.20 to 0.20 |
| left/forearm_roll | first-order lag with delay, deadband, angle-dependent offset | tau 60.7 ms [59, 62.4], d 21.3 ms [19.7, 22.8], db 0.128 deg [0.12, 0.137], b -0.2 deg [-0.356, -0.126] | published (MJCF form tracked worse on training) | 0.11 to 0.10 |
| left/wrist_angle | second-order servo with delay, deadband, angle-dependent offset | wn 23.4 rad/s [22.3, 23.7], zeta 0.932 [0.889, 0.942], d 0 ms [0, 0.74], db 0.116 deg [0.109, 0.121], b 0.266 deg [0.167, 0.617] | corrected | 1.24 to 0.88 |
| left/wrist_rotate | first-order lag with delay, deadband, angle-dependent offset | tau 34.8 ms [33.7, 36.5], d 28.4 ms [27, 29.6], db 0.0623 deg [0.0594, 0.065], b 0.0766 deg [0.0495, 0.167] | corrected | 0.70 to 0.19 |
| right/waist | second-order servo with delay, deadband, angle-dependent offset | wn 52.7 rad/s [22.8, 51.8], zeta 1.13 [0.673, 1.19], d 40 ms [19.6, 40], db 0.432 deg [0.422, 0.489], b -1.4 deg [-1.61, -0.77] | corrected | 0.46 to 0.31 |
| right/shoulder | second-order servo with delay, deadband, angle-dependent offset | wn 20.4 rad/s [20.2, 23.2], zeta 0.741 [0.73, 0.765], d 0 ms [0, 7.23], db 0.41 deg [0.383, 0.421], b -0.418 deg [-0.618, -0.295] | published (MJCF form tracked worse on training) | 0.25 to 0.25 |
| right/elbow | first-order lag with delay, deadband, angle-dependent offset | tau 39.9 ms [38.2, 41.7], d 36.7 ms [34.8, 38.4], db 0.309 deg [0.302, 0.314], b 2.02 deg [1.66, 2.44] | published (MJCF form tracked worse on training) | 0.17 to 0.17 |
| right/forearm_roll | first-order lag with delay, deadband | tau 50.9 ms [50.3, 51.8], d 27.8 ms [27.1, 28.3], db 0.0912 deg [0.0824, 0.096], b -0.0173 deg [-0.0194, -0.0129] | corrected | 0.13 to 0.11 |
| right/wrist_angle | first-order lag with delay, deadband | tau 55.5 ms [55, 55.9], d 23.9 ms [23.3, 24.5], db 0.207 deg [0.201, 0.214], b 0.392 deg [0.388, 0.394] | corrected | 1.41 to 0.18 |
| right/wrist_rotate | first-order lag with delay, deadband, angle-dependent offset | tau 33.1 ms [32.2, 33.9], d 25.5 ms [24.5, 26.4], db 0.0522 deg [0.0505, 0.0544], b 0.128 deg [0.064, 0.199] | corrected | 0.49 to 0.14 |

## Held-out comparison (15 episodes, logged commands replayed open loop in MuJoCo, RMS over the twelve arm joints)

| Model | Error |
|---|---|
| Menagerie aloha, as published | 0.629 deg [0.604, 0.654] |
| + one fitted command delay (best: 0.00 s) | 0.629 deg |
| + fitted delay and per-joint offsets | 0.898 deg |
| **corrected** | **0.318 deg** [0.291, 0.347] |
| the identified laws on their own | 0.150 deg |

Paired improvement over the published model: 0.311 deg (95% 0.271 to 0.345), 1.98 times closer.

## The same correction on the robot's other tasks (nothing refitted; every episode)

| Dataset | Episodes | Published (deg) | Corrected on coffee (deg) | Episodes closer |
|---|---|---|---|---|
| lerobot/aloha_static_battery (apache-2.0) | 49 | 0.71 | 0.36 | 49 of 49 |
| lerobot/aloha_static_candy (mit) | 50 | 0.63 | 0.49 | 50 of 50 |
| lerobot/aloha_static_tape (apache-2.0) | 50 | 0.77 | 0.49 | 50 of 50 |

## Using the corrected model

1. Put `aloha_identified.xml` in Menagerie's `aloha/` folder (it uses Menagerie's meshes from `assets/`).
2. Drive it with your logged or planned joint commands plus each joint's command correction, as the replay does:
   the offset `b` (and, for laws with an angle-dependent term, `sc cos(u) + ss sin(u)`, with u in radians) minus the
   joint's `mjcf.command_recentre_deg`, all from `result.json`, for the joints marked `used_in_mjcf`.

```python
import json, math, mujoco
r = json.load(open("result.json"))
m = mujoco.MjModel.from_xml_path("aloha_identified.xml")
def corrected_command(joint, u_deg):  # u_deg: the command for this joint, in degrees
    j = r["joints"][joint]
    if not j.get("used_in_mjcf", True):
        return u_deg
    p = {k: v["value"] for k, v in j["params"].items()}
    off = p["b"] + p.get("sc", 0.0) * math.cos(math.radians(u_deg)) + p.get("ss", 0.0) * math.sin(math.radians(u_deg))
    return u_deg + off - j["mjcf"]["command_recentre_deg"]
```

To calibrate another ALOHA from its own recordings: `physicslens robot calibrate <owner>/<dataset>`.
