# Sources: four more orbits

- `mercury.csv`, `mars.csv`, `ceres.csv`, `encke.csv`: heliocentric state vectors (ecliptic J2000, AU and AU/day, TDB) from the
  JPL Horizons API, fetched 2026-11-02 by `scripts/orbit_more.py fetch` (COMMAND 199, 499, `1;` and `DES=2P;CAP;NOFRAG`;
  CENTER 500@10). Public data from NASA/JPL Solar System Dynamics.
- The Sun's GM used for the comparison: `data/orbit/sun_gm.json` (JPL Horizons, Sun physical data).
- `results.json`: written by `scripts/orbit_more.py check` from the pipeline's own results (the videos and runs are under
  `var/orbit_more/`, not tracked). The analysis saw only the rendered pixels; JPL's positions drew the videos and then
  checked the laws.
