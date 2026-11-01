# The Sun's rotation: sources

| What | Source | Terms | Fetched |
|---|---|---|---|
| SDO/HMI continuum images (JPEG 2000) and their FITS headers, one every 6 hours over five 14-day periods | Helioviewer API (`getClosestImage`, `getJP2Image`, `getJP2Header`, source 18), cached under `var/astro/sun/frames/` (not committed) | NASA SDO data, courtesy of NASA/SDO and the HMI science team; free to use with acknowledgement | 2026-11-01 |
| NOAA SWPC Solar Region Summaries (daily) | NOAA NCEI mirror: `ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/solar_region_summaries/`, cached under `var/astro/sun/srs/`; one saved as `tests/fixtures/noaa_srs_20240508.txt` | U.S. government work, public domain | 2026-11-01 |
| `results.json` | `scripts/sun_demo.py` | derived | generated |

## Values compared against

- Published rotation profiles as implemented in `sunpy.sun.models.differential_rotation` (sunpy 8.0.1): Howard, Harvey &
  Forgach 1990 (Sol. Phys. 130, 295) and Snodgrass 1983 (ApJ 270, 288) in microradians per second, and Allen's
  Astrophysical Quantities (14.44 − 3.0 sin² φ degrees per day, a sunspot profile).
- Region returns: the region NOAA numbers on its return (a new number) within 5 degrees of latitude and 15 degrees of
  the predicted Carrington longitude; its east-limb crossing worked back from where it was first listed. NOAA's own
  forecast: section II ("Regions Due to Return") of the summaries, taken as the middle of its three-day window.
