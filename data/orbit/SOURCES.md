# Orbit data sources

All files here were fetched by `scripts/orbit_demo.py fetch` from free public NASA/JPL services and are cached so the demo runs offline.

| File | Source | Content |
|---|---|---|
| `apophis.csv`, `earth.csv` | JPL Horizons API (https://ssd.jpl.nasa.gov/api/horizons.api) | Heliocentric state vectors (ecliptic J2000, AU and AU/day, TDB), daily, 2018 to 2029, for 99942 Apophis and Earth (399) |
| `apophis_flyby.csv`, `earth_flyby.csv` | JPL Horizons API | The same, every 10 minutes from 2029-04-11 to 2029-04-16 |
| `cad_2029.json` | JPL SBDB Close-Approach Data API (https://ssd-api.jpl.nasa.gov/cad.api) | Apophis's official 2029 Earth close approach |
| `sun_gm.json` | JPL Horizons API, Sun (10) physical data | GM of the Sun |

Each CSV header records the API version and the fetch date. Credit: NASA/JPL Solar System Dynamics (Horizons, CNEOS).
