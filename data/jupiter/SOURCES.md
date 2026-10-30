# Jupiter data sources

Fetched by `scripts/jupiter_demo.py fetch` from the free public JPL Horizons API (https://ssd.jpl.nasa.gov/api/horizons.api) and cached so the demo runs offline.

| File | Content |
|---|---|
| `io.csv`, `europa.csv`, `ganymede.csv`, `callisto.csv` | Jupiter-centred state vectors (CENTER 500@599, ecliptic J2000, km and km/s, TDB) over two orbits from 2026-01-01, about 360 rows each |
| `jupiter.json` | Jupiter's GM and equatorial radius (Horizons physical data for 599) |

Each CSV header records the API version and fetch date. Credit: NASA/JPL Solar System Dynamics (Horizons).
