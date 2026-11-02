# Sources: lab measurement on labelled real footage

Clips are not committed (they stay in `var/lab_footage/`); `scripts/lab_footage.py fetch` downloads them and
`results.json` holds only the numbers the analysis produced.

- **Delfys75**: A. Castaneda Garcia et al., "Learning Physics From Video: Unsupervised Physical Parameter Estimation
  for Continuous Dynamical Systems", CVPR 2025, pp. 27924 to 27933. Data: Kaggle dataset
  `jaswar/physical-parameter-prediction` (https://www.kaggle.com/datasets/jaswar/physical-parameter-prediction),
  MIT licence (as stated on the dataset page). Download: `kaggle datasets download -d jaswar/physical-parameter-prediction`.
  Archive SHA-256: 1d2320351105ceb0b90c8a926d815acb0fe4eb629ed859318b90e269853c7137 (downloaded 2026-11-02).
  Used: the pendulum takes (string lengths 45, 90 and 150 cm, measured with a tape to 0.1 cm, five takes each).
  The lab's location is not stated; local gravity is taken for Delft (the authors' affiliation), which differs from
  anywhere in the Netherlands by under 0.02%.
- **Wikimedia Commons**: "Fadenpendelgeschwindigkeit (2 grad).webm" by Rhetos / Mathe-AC Lernwerkstatt Aachen,
  https://commons.wikimedia.org/wiki/File:Fadenpendelgeschwindigkeit_(2_grad).webm, CC BY-SA 4.0
  (https://creativecommons.org/licenses/by-sa/4.0/). A 60 cm pendulum with a ruler in view. Converted to H.264
  (SAM 3.1 does not accept WebM); local gravity for Aachen.
- Rejected: Physics 101 (no licence stated), PhyGenBench (no licence, no videos), IRIS (CC BY-NC 4.0, kept out
  so every result here can be shared without a non-commercial restriction).
