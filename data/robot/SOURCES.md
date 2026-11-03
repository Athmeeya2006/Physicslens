# Sources: the SO-100 robot arm case

Everything here is fetched by `python scripts/fetch_robot.py` into `var/robot/raw/` (gitignored) and checked against
`data/robot/sha256.txt`. Nothing raw is tracked; only the results the scripts write are.

## Recordings: LeRobot `lerobot/svla_so100_pickplace`

- URL: https://huggingface.co/datasets/lerobot/svla_so100_pickplace
- Licence: Apache-2.0 (dataset card metadata `license: apache-2.0`); licence text: https://www.apache.org/licenses/LICENSE-2.0
- Version: Hub revision `728583b5eaf9e739a7f119e2def466fa1d552402` (last modified 2026-07-16), LeRobot dataset format
  v3.0, `robot_type: so100`.
- Contents verified: 50 episodes, 19,631 frames at 30 frames per second (timestamps evenly spaced at 1/30 s in every
  episode; the runner refuses the data otherwise), one task ("Pick up the cube and place it in the box."), episodes of
  326 to 575 frames. Per frame: `action` (6 floats, the command: the leader arm's joint reading during teleoperation)
  and `observation.state` (6 floats, the follower arm's measured joints), named `main_shoulder_pan`,
  `main_shoulder_lift`, `main_elbow_flex`, `main_wrist_flex`, `main_wrist_roll`, `main_gripper`, in degrees of
  LeRobot's SO-100 calibration.
- Files used (about 0.6 MB): `data/chunk-000/file-000.parquet` (sha256 `e74b2786...767e1`, equal to the Hub's LFS id),
  `meta/info.json`, `meta/stats.json`, `meta/tasks.parquet`, `meta/episodes/chunk-000/file-000.parquet`, `README.md`.
  Full hashes in `sha256.txt`.
- Not used: the two camera videos (`videos/observation.images.top/...` 320 MB and `videos/observation.images.wrist/...`
  149 MB, AV1), not downloaded.
- Exact download: `python scripts/fetch_robot.py` (plain HTTPS from
  `https://huggingface.co/datasets/lerobot/svla_so100_pickplace/resolve/<revision>/<path>`).
- Citation: Cadene et al., "LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch",
  https://github.com/huggingface/lerobot, 2024. The dataset was published by the LeRobot team (Hugging Face) with the
  SmolVLA work: Shukor et al., "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics",
  arXiv:2506.01844, 2025.

## Simulator model: MuJoCo Menagerie `trs_so_arm100`

- URL: https://github.com/google-deepmind/mujoco_menagerie/tree/b0655057a9df2abe39ea7fc989914ba6f248301c/trs_so_arm100
- Licence: Apache-2.0 (`trs_so_arm100/LICENSE`; https://www.apache.org/licenses/LICENSE-2.0). Derived by the Menagerie
  authors from The Robot Studio's SO-ARM100 URDF (https://github.com/TheRobotStudio/SO-ARM100).
- Version: Menagerie commit `b0655057a9df2abe39ea7fc989914ba6f248301c` (main on 2026-10-31); the folder's last change
  is commit `71f066ad0be9cd271f7ed58c030243ef157af9f4` (2026-07-04). The README calls it "Standard Open Arm-100 5DOF,
  Version 1.3"; needs MuJoCo 3.1.6 or later (run here with MuJoCo 3.17.0).
- Files used: `so_arm100.xml`, `scene.xml`, `LICENSE`, `README.md` and the 18 meshes in `assets/` (about 3 MB).
- Defaults read from the file (class `so_arm100`): every joint `frictionloss="0.1"`, `armature="0.1"`, no joint
  damping; every actuator `<position kp="50" dampratio="1" forcerange="-3.5 3.5"/>` (kv from dampratio is 4.47 to
  5.13 N m s/rad), no actuator delay; timestep 0.002 s.
- Citation: Zakka, K. et al., "MuJoCo Menagerie: A collection of high-quality simulation models for MuJoCo", 2022,
  http://github.com/google-deepmind/mujoco_menagerie.

## What was used, and how

- Episodes 0 to 24 fit the candidate laws, 25 to 34 choose between them, 0 to 34 refit the winner and give the
  bootstrap intervals, and 35 to 49 are the held-out test (split by time, never shuffled).
- `result.json`, `so_arm100_identified.xml` and `so_arm100_identified.diff` are written by
  `python scripts/robot_demo.py` from these files alone. The identified MJCF is a modified copy of the Apache-2.0
  Menagerie file (the change is stated in its header comment and in the diff).

## Top camera video (for the video check)

- `videos/observation.images.top/chunk-000/file-000.mp4` from the same dataset and pinned revision, 320,383,994 bytes,
  sha256 `7c7f2674c7af4ac9fde4c0301ad25325ef8c655f0bd455b4b3b8a9ece1965d73` (AV1, 30 fps; cut per episode and
  re-encoded to H.264 at 15 fps for SAM 3.1). Fetched by `python scripts/robot_video_check.py fetch` into
  `var/robot/video/` (gitignored).
