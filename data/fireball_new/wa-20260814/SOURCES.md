# New fireball (Washington, 2026 August 14): sources

| Input | URL | Licence and terms | Retrieved | SHA-256 of the cached file |
|---|---|---|---|---|
| nasa_meo | https://fireballs.ndc.nasa.gov/skyfalls/events/20260814-074833 | NASA Meteoroid Environment Office public event page (US Government work, no copyright in the US) | 2026-11-02 | `790e673a3f591c45ca0f24c30898983d3ec0f09781755048b598c99e78fb409e` |
| cneos | https://ssd-api.jpl.nasa.gov/fireball.api?date-min=2026-08-14&date-max=2026-08-15&req-loc=true&vel-comp=true&req-alt=true | NASA/JPL CNEOS fireball API (US Government data, public) | 2026-11-02 | `8addfbdc512fcdd7bb59efd7902860f19a73ca5ab7e934c13bfabc1b6f135424` |
| sounding | https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-y2d/USM00072786-data-beg2026.txt.zip | NOAA NCEI IGRA2 (US Government data, public domain); Spokane WA, WMO 72786 | 2026-11-02 | `4b77a2ab2e0a000a3c78ada334bbbb8eb87164a74ae5d19aee4cf8dbe8cea380` |
| elevation | https://api.open-meteo.com/v1/elevation?latitude=47.80,47.85,47.90,47.95,48.00,47.85,47.85,47.95,47.95&longitude=-119.41,-119.41,-119.41,-119.41,-119.41,-119.30,-119.52,-119.30,-119.52 | Open-Meteo elevation API (Copernicus DEM GLO-90; data CC BY 4.0, attribution) | 2026-11-02 | `ec3dd2ff76a27e7d05e13a7003602892da147c13a9ad0a690c295c9d56fb4411` |

Command: `PYTHONPATH=backend .venv/bin/python scripts/fireball_new.py wa-20260814 --register` (downloads cached in `var/fireball_new/`).

Searched and not usable for this event: the Global Meteor Network trajectory summaries (CC BY 4.0) have no solution for it and none of their 2026 events ends below 32 km; the University of Wyoming archive returned no Spokane (72786) sounding for the date, so the same launch is taken from NOAA's IGRA2. No find had been reported in public sources as of the retrieval date.
