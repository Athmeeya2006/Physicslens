# Asteroid follow-up: sources

| What | Source | Terms | Fetched |
|---|---|---|---|
| Observations of 2024 YR4 with their uncertainties (ADES fields) | Minor Planet Center public API (`data.minorplanetcenter.net/api/get-obs`), cached under `var/astro/ast/` | MPC data, free to use with acknowledgement of the MPC and the observers | 2026-11-01 |
| Observatory codes and parallax constants | MPC, `minorplanetcenter.net/iau/lists/ObsCodes.html` (excerpt in `tests/fixtures/`) | as above | 2026-11-01 |
| Starting guess for the orbit | JPL Horizons vectors (only the start: the fit uses the observations alone, and converges to the same orbit from 30,000 km away) | JPL Solar System Dynamics | 2026-11-01 |
| `results.json`, `example_ades.psv` | `scripts/asteroid_demo.py` | derived | generated |

## Format

ADES PSV, version 2022, per "A Concise Description of the Astrometry Data Exchange Standard" (version of
16 March 2026, IAU-ADES/ADES-Master on GitHub), section 5: the `# version=` record first, context records, a keyword
record whose tokens all start with a lower-case letter, the identification group first. The MPC code, names and
telescope in the example are placeholders; nothing is submitted.
