# Sources: machine faults from physics on public data

All raw data is downloaded by `scripts/fetch_health_public.py` into `var/health_public/` (not tracked) and checked
against a sha256 for every file. Fetched 2026-11-02. Nothing from these datasets is redistributed here except
derived numbers (`result.json`, `REPORT.md`) and the checksum lists.

## Used

### 1. Bearings: University of Ottawa UORED-VAFCLS

- Dataset: "University of Ottawa Rolling-element Dataset: Vibration and Acoustic Faults under Constant Load and Speed
  conditions (UORED-VAFCLS)", Mendeley Data, version 5, published 2023-10-13, DOI 10.17632/y2px5tg92h.5,
  https://data.mendeley.com/datasets/y2px5tg92h/5
- Licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ (as stated in the Mendeley record's `data_licence`).
- Used: the 60 raw MATLAB files (`H_*_0`, `I_*_1/2`, `O_*_1/2`, `B_*_1/2`, `C_*_1/2`), about 310 MB; column 1
  (accelerometer on the drive-end bearing, 42 kHz, 10 s) and the speed value in row 1 of column 3. The spectrogram
  zips (1.8 GB) were not downloaded. Labels come from the file names (H healthy, I inner race, O outer race, B ball,
  C cage; stage 0 healthy, 1 developing, 2 faulty). Faults developed in accelerated run-to-failure tests, not seeded.
- Geometry (from the paper): NSK 6203ZZ and FAFNIR 203KD, 8 balls, ball diameter 6.77 mm, pitch diameter 28.50 mm;
  contact angle taken as 0 (deep-groove bearing, not stated in the paper). Nominal 1,750 rpm, 400 N (ball faults
  recorded without load).
- Download: `python scripts/fetch_health_public.py bearings` (Mendeley public API, each file checked against the
  sha256 Mendeley publishes). Checksums: `uored_files.sha256`.
- Citation: M. Sehri, P. Dumond, M. Bouchard, "University of Ottawa constant load and speed rolling-element bearing
  vibration and acoustic fault signature datasets", Data in Brief 49 (2023) 109327,
  https://doi.org/10.1016/j.dib.2023.109327

### 2. Fan and pump sound: MIMII

- Dataset: "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection",
  Zenodo record 3384388, version "public 1.0", 2019-09-20, DOI 10.5281/zenodo.3384388,
  https://zenodo.org/records/3384388
- Licence: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/ ("made available by Hitachi, Ltd. under a
  Creative Commons Attribution-ShareAlike 4.0 International license", record description). Derived results here are
  shared under the same terms where they are adaptations of the data.
- Used: `6_dB_fan.zip` and `6_dB_pump.zip`, model IDs 00, 02, 04, 06. Per ID a fixed subset chosen by a seeded
  shuffle of the sorted file names (seed 20261102, see `mimii_selection` in the fetch script): 40 abnormal clips,
  40 normal test clips, 60 normal reference clips; 1,120 clips, about 2.7 GB. Only microphone channel 1 is analysed.
  The clips are read straight out of the zips with HTTP range requests (CRC checked), so the 10 GB archives are never
  downloaded whole.
- Download: `python scripts/fetch_health_public.py mimii`. Checksums: `mimii_files.sha256` (sha256 of each extracted
  `.wav`); Zenodo md5 of the zips: 6_dB_fan.zip 0890f7d3c2fd8448634e69ff1d66dd47, 6_dB_pump.zip
  a09ba6060c10fc09cd4c8770213b0b9f.
- Citation: H. Purohit, R. Tanabe, K. Ichige, T. Endo, Y. Nikaido, K. Suefusa, Y. Kawaguchi, "MIMII Dataset: Sound
  Dataset for Malfunctioning Industrial Machine Investigation and Inspection", Proc. DCASE 2019 Workshop;
  arXiv:1909.09347. Baseline AUCs quoted from its Table 3 (6 dB column).

### 3. Spin-down: in-wheel motor coast-down

- Dataset: "Data and scripts for: Identifiability of a Regenerative Braking Loss Model for an In-Wheel Motor Drive
  from Direct-Current-Side Data", Zenodo record 23231539, version 2 (October 2026), DOI 10.5281/zenodo.23231539,
  https://zenodo.org/records/23231539
- Licence: data CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), scripts MIT (record `LICENSE.txt`).
- Used: `bench_spindown_lifted.csv` only (five open-circuit coast-downs of the lifted wheel from about 900 rpm,
  10 Hz). The published comparison values (mean T0/J_w = 13.380 +- 0.380 rad/s^2, k/J_w = 0.0790 +- 0.0028 1/s) are
  from the record's `Results.md`.
- Download: `python scripts/fetch_health_public.py spindown`. sha256
  426c1aee42e64e9af3794629a504397a161840f968bc52b6f5577444851e209e (Zenodo md5 0fbefe8eca39d1049d8b7a52da456b63).
- Citation: Le Hoang Long, Khong Vu Quang, Pham Huu Nam, Pham Van Thoan, Pham Minh Hieu, Nguyen Thanh Tung, data and
  scripts for "Identifiability of a Regenerative Braking Loss Model for an In-Wheel Motor Drive from Direct-Current-Side
  Data", World Electric Vehicle Journal (manuscript), Zenodo 2026.

### Published baselines cited, data not used

- DCASE 2020 Challenge Task 2 baseline AUCs (fan and pump, per machine ID), from the task page
  https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds . Quoted as numbers only.

## Considered and rejected

- Case Western Reserve University Bearing Data Center (https://engineering.case.edu/bearingdatacenter): no licence or
  terms of use on the site (checked 2026-11-02: overview, project history, apparatus and download pages; only
  "(c) 2024 Case Western Reserve University"). Unclear licence, so not used. Third-party re-uploads elsewhere carry
  licences their uploaders cannot grant.
- Paderborn University bearing dataset (KAt-DataCenter, https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter):
  CC BY-NC 4.0, "noncommercial academic use"; commercial use needs the authors' permission. Not in the allowed set
  (non-commercial clause), and a CC BY alternative with real faults exists.
- DCASE 2020 Task 2 development dataset (Zenodo 3678171): CC BY-NC-SA 4.0. Not used (non-commercial clause); its
  published baseline AUCs are quoted for context only.
- Air-driven flat rotors (Zenodo 5141571, CC BY 4.0): steady driven rotation, no coast-downs, so it cannot test the
  spin-down law.
