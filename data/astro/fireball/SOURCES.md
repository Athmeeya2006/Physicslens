# Fireball data: sources

| File | What | Source | Licence and terms | Fetched |
|---|---|---|---|---|
| `dfn_DN150417.csv` | heights and speeds of Desert Fireball Network event DN150417 (the example of the alpha-beta notebook) | `desertfireballnetwork/alpha_beta_modules` on GitHub | MIT (Copyright 2018 hadry) | 2026-11-01 |
| `winchcombe_wmpl_report_all.txt`, `winchcombe_wmpl_report_initial.txt` | the published trajectory solutions of the Winchcombe fireball (every measured point, state vector, radiant, orbit) | Zenodo record 10.5281/zenodo.6685719, "Fireball data for the Winchcombe meteorite" (McMullan et al.), file `trajectory.tar` | CC BY 4.0 | 2026-11-01 |
| `winchcombe_darkflight_config.cfg` | the dark-flight start state the authors used (end point, speed, angles and their uncertainties, density, shape range, ground level) | same record | CC BY 4.0 | 2026-11-01 |
| `winds_nottingham_03354_2021-02-28_12Z.csv`, `winds_nottingham_03354_2021-03-01_00Z.csv` | radiosonde winds (height, east, north) from Nottingham Watnall (WMO 03354), the soundings either side of the fall | University of Wyoming upper-air archive (TEXT:LIST), parsed by `astro/fireball.py` | public observations (WMO / Met Office); the archive asks for acknowledgement | 2026-11-01 |
| `winchcombe_finds.csv` | the ten recovered stones: place, mass, coordinates to three decimals (as published, to protect the homeowners) | Russell et al. 2024, "Recovery and curation of the Winchcombe (CM2) meteorite", Meteoritics & Planetary Science 59, 973 (doi:10.1111/maps.13956), Table 1; finders' names left out | published measurements, cited | 2026-11-01 |
| `results.json`, `winchcombe_strewn_field.geojson`, `winchcombe_strewn_field.kml` | fits, the orbit, the simulated strewn field and the comparison with every find | `scripts/fireball_demo.py` | derived | generated |

## Values compared against (each checked at its source)

- Orbit (a, e, i, node, argument of perihelion, q, geocentric speed) and its 1-sigma uncertainties: McMullan et al.
  2023, "The Winchcombe Fireball: that Lucky Survivor" (arXiv:2303.12126), orbit table, and the record's
  `_initial` report.
- Dark flight of three stones (impact speed, minutes of fall, the shape factors that match): same paper, section
  "Strewn field" (main mass 46 to 49 m/s after about 4.5 min at shape 1.4 to 1.5; 160 g 35 to 40 m/s after about
  6 min at 1.5 to 1.7; 16.5 g 17 to 19 m/s after about 16 min at 2.7).
- Last fragmentation (a flare) at 34.9 km: same paper, fragmentation table.
- Entry mass 13 ± 3 kg: King et al. 2022 as quoted by Russell et al. 2024 (the paper's own model: 12.5 kg).
- Alpha-beta equation and the two 50 g lines (13.20 and 4.4): Sansom et al. 2019, ApJ 885, 115 (eqs. 1, 10, 11),
  checked in the arXiv source.
- Drag coefficient against Mach number (sphere and cube, Carter et al. 2011): as written in the Desert Fireball
  Network's `DFN_darkflight` code (`atm_functions.dragcoefff`, MIT).

## Notes and corrections

- The alpha-beta notebook prints alpha = 294.45, beta = 1.140 for DN150417. Running the same notebook code with
  current SciPy gives alpha = 249.6, beta = 1.806, a lower value of the same objective (0.817 against 0.885), and
  PhysicsLens finds the same point. The printed values came from an older run. Both points fall on the same side
  of the 50 g lines ("unlikely").
