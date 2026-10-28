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

15 to 20 clips across at least four systems. Report every one.
