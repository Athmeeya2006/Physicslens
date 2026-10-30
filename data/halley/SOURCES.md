# Halley data sources

All files here were fetched by `scripts/halley_demo.py fetch` from free public NASA/JPL services and are cached so the demo runs offline.

| File | Source | Content |
|---|---|---|
| `halley.csv`, `earth.csv` | JPL Horizons API (https://ssd.jpl.nasa.gov/api/horizons.api) | Heliocentric state vectors (ecliptic J2000, AU and AU/day, TDB), daily, 1982 to 1990, for comet 1P/Halley (small-body record 90000030) and Earth (399) |
| `halley_2061.csv` | JPL Horizons API | The same for Halley, daily, 2059 to 2063: JPL's own prediction of the next perihelion, integrated with the planets |
| `sun_gm.json` | JPL Horizons API, Sun (10) physical data | GM of the Sun |

Record 90000030 is JPL's orbit solution for Halley fitted to observations through the 1986 apparition. Each CSV header records the API version and the fetch date. Credit: NASA/JPL Solar System Dynamics (Horizons).
