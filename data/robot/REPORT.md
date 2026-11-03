# SO-100 arm: a simulator corrected from public recordings

Status: **proven**. Every number is in `result.json` (written by `python scripts/robot_demo.py` from the pinned files `python scripts/fetch_robot.py` downloads; see `SOURCES.md`). Local math only: no hardware, no model calls.

## Data, model, method

- **Data:** LeRobot `lerobot/svla_so100_pickplace` (Apache-2.0): 50 teleoperated pick-and-place episodes, 19,631 frames at 30 per second; per frame the command (`action`, the leader arm) and the measured joints (`observation.state`).
- **Simulator:** MuJoCo Menagerie `trs_so_arm100` (Apache-2.0). Defaults: position actuators `kp=50`, `dampratio=1`, `forcerange=+-3.5`; joints `armature=0.1`, `frictionloss=0.1`; no actuator delay.
- **Split by time:** episodes 0 to 24 fit every candidate law, 25 to 34 choose, 0 to 34 refit the winner, 35 to 49 (15 episodes) are used once, for the comparison.
- **Candidates per joint** (e = u(t - d) + b - theta): first-order lag `theta' = e / tau` and second-order servo `theta'' = wn^2 e - 2 zeta wn theta'`, each with optional speed limit, dry friction, deadband and an angle-dependent offset (`sc cos(theta) + ss sin(theta)`). All simulated open loop from the command alone. A richer law is kept only if it lowers held-out error (95% interval of the paired gain above zero) and lowers BIC.

## Identified laws (95% intervals: bootstrap over training episodes)

| Joint | Law | Parameters |
|---|---|---|
| shoulder_pan | lag, delay, deadband | tau 60.4 ms [58.9, 61.5], d 47.1 ms [46.1, 48.5], db 0.187 deg [0.172, 0.197] |
| shoulder_lift | lag, delay, deadband, angle-dependent offset | tau 61.9 ms [60.2, 63.3], d 50.0 ms [48.5, 51.9], db 0.40 deg [0.33, 0.48], b 0.56 deg [0.48, 0.63], sc 0.98 [0.88, 1.11], ss 0.60 [0.52, 0.68] deg |
| elbow_flex | second order, delay, deadband | wn 29.1 rad/s [26.3, 35.1], zeta 0.98 [0.95, 1.08], d 21.2 ms [17.3, 26.7], db 1.21 deg [1.18, 1.25], b 1.54 deg [1.53, 1.56] |
| wrist_flex | lag, delay, deadband | tau 60.4 ms [59.0, 61.9], d 40.2 ms [39.3, 41.5], db 0.40 deg [0.38, 0.41], b 0.28 deg [0.27, 0.29] |
| wrist_roll | lag, delay, deadband | tau 54.8 ms [51.4, 56.7], d 47.8 ms [46.4, 50.4], db 0.34 deg [0.33, 0.36] |
| gripper | not identified | closes on the cube; its best law sits at its parameter bounds |

Rejected: the speed limit (motion here never exceeds about 170 deg/s, so none shows) and dry friction (the deadband explains the stalls better); second order beats the lag only on the elbow. Per-episode fits agree on delays, time constants and deadbands; the elbow's wn (per-episode quartiles 23 to 52 rad/s) and the shoulder lift's offset terms vary more than their pooled intervals.

## Written into the MJCF (`so_arm100_identified.xml`, diff in `so_arm100_identified.diff`)

Per arm actuator: `kp = wn^2 I`, `kv = 2 zeta wn I` (I from the model's mass matrix, armature included; a lag is written as an overdamped pair with poles at 1/tau and 8/tau), native actuator `delay`, and joint `frictionloss = kp * db`. On the command instead (MJCF cannot hold them): the offset b and the shoulder lift's angle-dependent offset, minus the static sag MuJoCo's own gravity adds (measured on training episodes). Gripper unchanged.

## Held-out comparison: 15 episodes replayed open loop in MuJoCo (RMS deg, 95% interval)

| Model | Arm (5 joints) |
|---|---|
| (a) Menagerie defaults | 0.754 [0.716, 0.792] |
| (c) fair baseline: defaults + one delay for all joints (20 ms, chosen on training) | 0.655 [0.617, 0.695] |
| (c2) stronger baseline: (c) + constant offset per joint | 0.552 [0.536, 0.569] |
| (b) identified MJCF + command offsets | **0.383 [0.367, 0.401]** |
| identified MJCF, no command offsets | 0.726 [0.695, 0.759] |
| identified laws themselves (no MuJoCo) | 0.272 [0.256, 0.291] |

Paired improvement of (b): 0.371 deg [0.331, 0.412] over the defaults (1.97 times closer), 0.272 [0.231, 0.314] over the fair baseline (1.71 times), 0.169 [0.156, 0.183] over the stronger baseline (1.44 times). Rule: "proven" only if it beats both defaults and the fair baseline beyond their intervals; it does.

Plainly: the defaults already get timing roughly right; most of the gain is the deadband (largest on the elbow) and the offsets. MuJoCo friction only approximates a deadband, which is why (b) trails the laws themselves (mostly on the elbow). The log-to-model joint map (sign +1, shared rest pose) is an assumption; turning gravity off moves the default from 0.754 to 0.887, so it matters at the 0.1 deg level.

## Limits

Only the motions in these 50 episodes (one task, one arm unit); payloads, wear, temperature and other units are not covered. The delay includes bus and logging latency (what a replay of logged commands needs). Gripper not identified.

## Video check (SAM 3.1 on the top camera; `scripts/robot_video_check.py`, numbers in `video_check.json`)

The gripper was tracked with SAM 3.1 in the top camera video (one box prompt per clip, 1,381 frames at 15 per second)
and compared with where the logged joint angles put it, through a pinhole camera fitted on two training episodes
(episodes 33 and 34: rotation, position, focal length and the tracked point on the gripper). On the 5 held-out
episodes 35 to 39 the tracked gripper and the logged joint path agree to a median 4.1 mm (95% over episodes 2.1 to
6.1 mm; 7.6 px), RMS 5.7 mm. They disagree most when the arm is high and moving (a timing jitter, joint calibration or
cable flex, which the data cannot separate), when the gripper is open (the moving jaw is inside the mask), when the
cube merges into the mask during a grasp, and where the gripper leaves the frame.

One finding about the joint map: the video only fits a physical camera if the shoulder pan and wrist roll angles are
read with the opposite sign from the map the replay assumes (sign +1 on every joint); with the map as assumed, the
best camera is a mirror image below the table. The video alone cannot tell "pan and roll reversed" from "the image is
mirrored". Pan turns about the vertical axis, so this hardly moves where gravity acts in the MuJoCo replay, but the
assumption is stated here rather than hidden.

## Suggested upstream change (MuJoCo Menagerie `trs_so_arm100`; not submitted)

1. Actuator `delay` of about 40 to 50 ms per arm joint (about 20 ms on the elbow), with `nsample` sized for it.
2. Joint `frictionloss` raised so the static band matches the measured deadband (0.2 to 0.4 deg; 1.2 deg on the elbow).
3. A README note that LeRobot recordings need per-joint command offsets of up to 1.5 deg between leader and follower.
