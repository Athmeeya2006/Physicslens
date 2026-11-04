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
  Space-Track account (not yet created).

## Trajectory history and burn records (for `history.json`)

| What | Source | Terms | Fetched |
|---|---|---|---|
| Archived copies of NASA's ISS ephemeris (the same OEM file as above), 2021-02 to 2026-10: 293 captures with distinct content, listed by the Internet Archive's CDX index (`web.archive.org/cdx/search/cdx?url=nasa-public-data.s3.amazonaws.com/iss-coords/current/ISS_OEM/ISS.OEM_J2K_EPH.txt&collapse=digest`) and read from `web.archive.org/web/<timestamp>id_/<url>`, one request every 4 s, cached under `var/astro/sat/wayback/` | NASA JSC (TOPO office), archived by the Internet Archive | U.S. government work, public domain; Internet Archive access for research | 2026-11-02 |
| The current file again, for the forward look | `nasa-public-data.s3.amazonaws.com/iss-coords/current/ISS_OEM/ISS.OEM_J2K_EPH.txt` (the URL the trajectory data page points to; still the official location) | public domain | 2026-11-02 |
| Planned burns | The "TRAJECTORY EVENT SUMMARY" table in the header comments of those files (event, ignition GMT as day-of-year, delta-v in m/s and ft/s, apogee and perigee). Read only for scoring; the detector never sees it. The current file (created 2026-10-30) has no table. | public domain | 2026-11-02 |
| Measured burns | NASA Glenn PIMS "ISS Handbook" reports, one PDF per burn, `gipoc.grc.nasa.gov/pims/pimsdocs/public/ISS Handbook/hb_vib_vehicle_<vehicle>_Reboost_<date>.pdf` (start time from the on-board acceleration step, delta-v measured or as predicted by flight controllers). The directory cannot be listed (HTTP 403), so the reports were found by web search: 19 burns from 2022-01 to 2026-03; PIMS does not document every burn. Each URL is in `announced_burns.json`. | U.S. government work | 2026-11-02 |
| Announced burns | NASA's space station blog and ISS Daily Summary Reports (URLs per entry in `announced_burns.json`); times converted from the stated US time zone to UTC | U.S. government work | 2026-11-02 |
| `announced_burns.json` | collected by hand from the sources above, after the detection rule was fixed in code | derived | 2026-11-02 |
| `history.json` | `scripts/satellite_history.py` | derived | generated |

Notes:
- Space-Track's historical element sets were not used: they need an account and their terms restrict redistribution.
  The Internet Archive's copies of CelesTrak's `stations.txt` are sparse (a few hundred captures over 27 years) and
  were not needed, because the archived NASA ephemerides are both denser and more precise.
- The archive is dense from mid 2022 to May 2024 (a file every 1 to 3 days) and sparse before and after (often weeks
  or months apart); gaps over 7 days are not scored.
- The NG-19 PIMS report gives "233/14:35" next to the date 2023-08-11, which is day 223; its figure axis reads 223,
  so day 223 is used.
- Five records were looked up only because the detector flagged their gaps; they are marked
  `looked_up_after_detection`, and the strict count in `history.json` (`strict_before_lookups`) treats those gaps as
  false alarms:
  - 2022-10-17 reboost: NASA ISS Daily Summary Report, `www.nasa.gov/blogs/stationreport/2022/10/page/2` (date only);
  - 2022-12-21 debris avoidance, 1.0 m/s: `www.nasa.gov/blogs/stationreport/2022/12/21/iss-daily-summary-report-12-21-2022/`;
  - 2023-03-06 debris avoidance, 0.7 m/s: Roscosmos via Interfax, `www.interfax.com/newsroom/top-stories/88483/`
    (NASA's blog post on it was quoted by news outlets but not retrieved);
  - 2023-03-14 debris avoidance, 0.3 m/s: Roscosmos via Interfax, `interfax.com/newsroom/top-stories/88682/`;
  - 2023-08-24 debris avoidance (lowering, about 500 m): NASA,
    `www.nasa.gov/blogs/spacestation/2023/08/24/station-preps-for-new-cargo-crew-and-avoids-space-debris/`, and
    Roscosmos via Interfax, `www.interfax.com/newsroom/top-stories/93923/`.
  The one remaining flagged gap (2023-08-04 to 2023-08-07, +0.24 m/s) has no record found and stays a false alarm.
- Every other burn record and its URL is listed in `announced_burns.json`: 19 NASA PIMS ISS Handbook reports
  (`gipoc.grc.nasa.gov/pims/pimsdocs/public/ISS%20Handbook/hb_vib_vehicle_<name>.pdf` for Progress 79P 2022-01-12,
  NG-17 Cygnus 2022-06-25, Progress 81P 2022-10-01 and 2023-01-18, Progress 83P 2023-07-04, NG-19 Cygnus 2023-08-11,
  Progress 85P 2023-10-19 and 2023-12-23, Progress 87P 2024-03-14 and 2024-04-26, Cygnus NG-20 two burns 2024-05-24
  and 2024-06-08, Progress 91P 2025-03-17, 2025-07-16 and 2025-08-14, Dragon SpX-33 2025-09-03 and 2025-11-07,
  Progress 93P 2026-03-13) and three NASA blog posts:
  `www.nasa.gov/blogs/spacestation/2022/04/23/axiom-mission-1-undock-postponed-to-sunday-space-station-reboosts`,
  `www.nasa.gov/blogs/spacestation/2022/06/20/cygnus-reboost-aborted-next-steps-being-planned/` (the aborted 5 s
  attempt, kept so it is not counted as a miss) and `www.nasa.gov/blogs/stationreport/2024/01/29`.
- What is missing: the archive has no NASA file between most of the June 2024 to October 2026 burns, so most PIMS
  burns after May 2024 fall in unscored gaps; a complete, dated list of every ISS burn is not public in one place.
