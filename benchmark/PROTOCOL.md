# Benchmark protocol

Real clips with independently measured answers, written down **before** any analysis is run, so PhysicsLens can't be tuned to them. Every clip is reported in `RESULTS.md`, failures included.

## Filming rules (all clips)

- Phone on a support (a stack of books is fine), side-on to the motion, not moving. Good light, no zoom.
- At least 10 s of motion (bounces and drops: the whole fall and at least three bounces).
- Normal speed unless a row says slow motion; if slow motion, write down the factor.
- A reference of known size in the plane of motion when the row needs lengths (an A4 sheet taped flat behind the motion).

## What to measure, per system

| System | Ground truth, measured before analysis | Compared with |
|---|---|---|
| Pendulum | string length to the bob's centre (tape measure); period by stopwatch over 20 swings | g (from the given length), period |
| Spring and mass | stiffness from hanging known masses, k = m g / extension; mass of the bob (kitchen scale) | natural frequency, against sqrt(k / m) |
| Drop and bounce | drop height (tape behind the ball); rebound heights from the video frame where the ball stops, e = sqrt(h_after / h_before) | restitution per bounce; g |
| Car bounce test | (no lab truth) repeat three releases per corner | agreement across releases (the study's chi-squared test) |
| Fan or wheel spin-down | rotation rate at two times from slow motion, counted by hand | spin-down law and its rate |
| Sliding block | incline angle at which it starts to slide, mu_static = tan(angle) | kinetic friction from the slide (should be below mu_static) |

## Files

- `truths.json`: one entry per clip, written before the analysis: `{"clip": "pendulum_1.mp4", "family": "pendulum", "given": {"pendulum_length_m": 0.52}, "truth": {"g": 9.81, "period_s": 1.447}, "notes": "..."}`. Keep the clips next to it (they are not committed unless their licence allows).
- `RESULTS.md` and `results.json`: written by `scripts/benchmark.py` (truth, measured value with interval, error, whether the 95% interval contains the truth).

## Target size

15 to 20 clips across at least four systems. Report every one. `benchmark/FILMING.md` is the checklist for the next
four (a door closer twice, a bouncing ball, a spinning toy, a 3D throw).

## Acceptance criteria (what "accepted" means in every number)

The rules in force, with the constants that set them (`backend/physicslens/physics/verify.py`, `rigor.py`, `fit.py`,
`uncertainty.py`, `prescreen.py`):

1. **Held-out forecast.** Each law is fitted on the first 75% of the motion (`TRAIN_FRACTION`) and run on to the end;
   its RMSE over the last 25%, divided by the motion's standard deviation, must be under 0.15 (`MAX_HELDOUT_NRMSE`),
   or under 0.3 (`HARD_MAX_HELDOUT_NRMSE`) when that RMSE is within 3 times the tracking noise
   (`MAX_HELDOUT_NOISE_RATIO`: a correct law cannot beat the noise floor).
2. **Rolling origins.** A law that passes is also restarted from 55, 65, 75 and 85% of the clip (`ROLLING_ORIGINS`),
   parameters fixed and only the starting state refitted on the data before each point, and must forecast the next
   15% (`ROLLING_HORIZON`); the median of those errors must pass the same rule. Live mode skips this for speed.
3. **BIC with the effective sample size.** n_eff = n (1 - rho) / (1 + rho) from the lag-1 autocorrelation rho of the
   best law's residuals, at least 10 (`MIN_N_EFF`), the same n_eff for every law. A winner needs a BIC lead of 10
   (`DECISIVE_DELTA_BIC`); otherwise the laws tie and an experiment is requested. A law is fitted from its special
   cases' solutions, so it can never end worse than a law it contains.
4. **Referee.** After acceptance, up to three alternatives proposed by a second model are fitted and judged by rules 1
   to 3; the verdict changes only if one passes them and leads by BIC.
5. **Fast refusal.** Before any simulation, motion that no law's acceleration explains (R² under 0.2, velocity
   flatness over 0.5, a quadratic trend explaining under 95%: `prescreen.py`) is refused as noise-like.
6. **Uncertainty.** Residual block bootstrap refits: 40 when time allows, never fewer than 8
   (`MIN_BOOTSTRAP_REPS`) within 60 s (`MAX_BOOTSTRAP_SECONDS`); the result records how many ran. Pairs of parameters
   whose refits correlate beyond 0.9 are reported with filming advice (`IDENTIFIABILITY_R`).

## Seeds and determinism

- Bootstrap resampling: `numpy.random.default_rng(0)` (`physics/uncertainty.py`), so a stored track gives the same
  intervals every time.
- Solar-system discovery noise: `seed=0` (`sky/discover.py`); the negative control uses seeds 0 to 19.
- Controllers: the 50 bootstrap variants are drawn with a fixed seed stored with the run.
- Synthetic tests set their own seeds (`np.random.default_rng(k)` in each test).
- Model calls are not deterministic (SAM and the reasoning model vary slightly between identical calls), which is
  why every analysis is cached by its inputs and the gallery stores the result, not a recipe to rerun.

## Reproducing the README's numbers

Every number the README quotes from a result is listed in `scripts/readme_numbers.py` with the file it comes from,
and `python scripts/readme_numbers.py` (or `make numbers`, also run in CI) recomputes each one and compares it at
the precision the README writes it. The sources:

| Numbers | Recompute with | Reads |
|---|---|---|
| Gallery results (pendulum, paper bag, spring, Apophis, Halley, Io) | `scripts/readme_numbers.py` | `gallery/*/result.json` |
| Jupiter's four moons, Kepler's third law | `python scripts/jupiter_demo.py report` (writes `var/jupiter/report.json`; the committed copy is `data/jupiter/report.json`) | the four stored analyses |
| The engine against JPL, direct-track GM | `python scripts/sky_demo.py validate` | `data/sky/` (cached JPL states and masses) |
| Le Verrier's Neptune | `python scripts/sky_demo.py leverrier` | `data/sky/leverrier.json` |
| Halley and Apophis forecasts, apparitions, the 2029 aftermath | `python scripts/halley_demo.py prediction`, `python scripts/orbit_demo.py prediction` | `data/halley/`, `data/orbit/` |
| Classical baselines | `python scripts/baselines.py` | `benchmark/results/` |
| The reasoning model alone | `python scripts/llm_alone_table.py` (the answers were collected once by `scripts/llm_alone.py`, paid, with approval) | `benchmark/llm_alone/` |
| Physics audit study | `python scripts/audit_study.py analysis` (free, from the stored runs) | `data/audit_study/` |
| Lens calibration gain, 3D tilt, synthetic validation | `make test` (each is a test with its numbers in the docstring) | `tests/` |
| Test count | `pytest --collect-only -q` | `tests/` |

Analyses themselves (SAM 3.1 and the reasoning model) are paid and run only with approval; everything in this table
is local and free.
