# Satellites: sources

| What | Source | Terms | Fetched |
|---|---|---|---|
| The ISS trajectory (CCSDS OEM, EME2000, 4-minute states over 15 days) | NASA JSC (TOPO office), `nasa-public-data.s3.amazonaws.com/iss-coords/current/ISS_OEM/ISS.OEM_J2K_EPH.txt`, cached under `var/astro/sat/`; an excerpt in `tests/fixtures/iss_oem_head.txt` | U.S. government work, public domain | 2026-11-01 |
| Catalogue orbits (GP elements as OMM JSON, groups "stations" and "visual") | CelesTrak, `celestrak.org/NORAD/elements/gp.php` (fetched once and cached, as CelesTrak asks) | free to use with acknowledgement of CelesTrak | 2026-11-01 |
| `results.json` | `scripts/satellite_demo.py` | derived | generated |

## Formats and constants

- IOD observation format: G. Lewis, "Interactive Orbit Determination (IOD) Version 0" (1998, clarified 2002), as
  reproduced at satobs.org/position/IODformat.html: columns 1-15 object, 17-20 station, 22 condition, 24-40 UTC,
  42-43 time uncertainty (M x 10^(X-8)), 45 angle format, 46 epoch, 48-61 position, 63-64 position uncertainty.
- Zonal harmonics J2, J3, J4 (EGM2008, unnormalised); the acceleration is tested against the gradient of the
  potential.
- No real manoeuvre is reproduced: NASA's current ephemeris lists none, and historical catalogue orbits need a
  Space-Track account (the human's to create).
