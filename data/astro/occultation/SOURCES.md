# Occultations: sources

| File | What | Source | Terms | Fetched |
|---|---|---|---|---|
| `umbriel_2020-09-21_chords.csv` | 19 positive chords (sites, ingress and egress times with errors) and 5 negative observers (sites, recording windows) of the 2020 September 21 occultation by Umbriel | Assafin et al. 2023, "Kilometer-precise (UII) Umbriel physical properties from the multichord stellar occultation on 2020 September 21", arXiv:2310.11196, Tables 1 to 3 | published measurements, cited | 2026-11-01 |
| Umbriel's geocentric positions | JPL Horizons (target 702), cached under `var/astro/ast/` | JPL Solar System Dynamics, free to use | 2026-11-01 |
| `results.json` | `scripts/occultation_demo.py` | derived | generated |

## Values compared against

- Circular fit: R = 582.4 ± 0.8 km, limb parameter 4.2 km; elliptical fit: a' = 582.6 ± 1.1 km, equivalent radius
  581.7 ± 2.0 km with the position angle fixed (Assafin et al. 2023, Table 4).
- The ephemeris offsets differ from the paper's (it used DE435/URA111; Horizons today uses newer ephemerides), which
  moves the centre but not the size.
