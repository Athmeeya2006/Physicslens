# Refreshed transit ephemerides: sources

Planets whose published ephemeris predicts 2027-01-01 to worse than 30 minutes (1 sigma), refreshed with transit
mid-times measured here from TESS light curves, checked on held-out transits, and (when the check passed)
pre-registered as blind predictions in `predictions/transit-timing/`.

| File | What | Source | Licence and terms | Retrieved |
|---|---|---|---|---|
| `results.json` | every planet examined: selection numbers, published ephemeris and its references, TESS sectors used (MAST product ids), every mid-time, the refreshed ephemeris, the held-out check, the shift at 2027-01-01, the windows, and why a planet was dropped or not registered | `scripts/ephemeris_refresh.py` | derived | generated 2026-11-02 |
| `refreshed_ephemerides.csv` | one row per refreshed planet, with ExoClock's own field names (`t0_bjd_tdb`, `t0_unc`, `period_days`, `period_unc`, `duration_hours`, `depth_mmag`; uncertainties in days); the reference epoch is the one where t0 and P are least correlated | derived | | generated 2026-11-02 |
| `tess_midtimes.csv` | every individual TESS mid-time (BJD_TDB, 1-sigma in days as fitted, unscaled), with its epoch and sector | derived from TESS SPOC 2-min light curves | | generated 2026-11-02 |
| `windows_2026-11_2027-05.csv` | each predicted transit from 2026-11-15 to 2027-05-15: mid-time, 68% and 95% intervals, the published ephemeris' prediction and 95% interval, the shift | derived | | generated 2026-11-02 |

Inputs (cached under `var/ephemeris/`, not committed):

- NASA Exoplanet Archive, table `pscomppars`, TAP query
  `https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select pl_name,hostname,tic_id,pl_trandep,pl_ratror,pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,pl_orbper,pl_orbpererr1,pl_orbpererr2,pl_trandur,sy_vmag,sy_tmag,pl_tranmid_reflink,pl_orbper_reflink,ra,dec from pscomppars where tran_flag=1&format=csv`,
  retrieved 2026-11-02. The ranking itself comes from `data/astro/timing/pscomppars_transiting.csv` (the same
  table, retrieved 2026-11-01; every selected planet's t0 and P are identical in both). Licence: free to use with the
  acknowledgement "This research has made use of the NASA Exoplanet Archive, which is operated by the California
  Institute of Technology, under contract with the National Aeronautics and Space Administration under the
  Exoplanet Exploration Program." Each planet's t0 and P reference (the paper the archive took them from) is in
  `results.json` and in each prediction's inputs.
- TESS SPOC 2-minute light curves (PDCSAP flux, quality flag 0) from MAST, through lightkurve 2.6
  (`lk.search_lightcurve("TIC <id>", author="SPOC", exptime=120)`), retrieved 2026-11-02. Each sector's MAST
  `obs_id`, `obsid`, file name and download URL are listed per planet in `results.json` and in each prediction.
  TESS data are public (NASA); acknowledge the TESS mission (funded by NASA's Science Mission Directorate) and MAST.

## Exact commands

```
.venv/bin/python scripts/ephemeris_refresh.py          # select, measure, refresh, check, register, write this folder
.venv/bin/python scripts/score_predictions.py --dry-run # which registered predictions are due (none before their date)
PYTHONPATH=backend MOCK_MODELS=1 APP_ENV=dev .venv/bin/pytest -q tests/test_ephemeris_refresh.py tests/test_prereg.py
```

## How the registrations were made

Registered on 2026-11-02 between 10:57 and 11:19 UTC by `scripts/ephemeris_refresh.py`. The first run registered
six (Qatar-8 b, WASP-117 b, HAT-P-11 b, HAT-P-3 b, KELT-1 b, HAT-P-5 b), then stopped on a dropped MAST connection
at the next candidate. The rerun (now with download retries) did not register those six again: it recomputed them
from the same cached light curves, checked that every window equals the registered value
(`recomputed_matches_registered` in `results.json`), and registered TOI-150.01 and HAT-P-7 b. Earlier development
registrations made before the final rules existed were deleted before any of these were made; none was ever pushed.

## Method, in short

- Mid-times: `astro/transits.py` (the code used on Kepler-88), per sector: a box least squares search near the
  catalogue period fixes the sector's phase, each transit is detrended locally and fitted with one transit shape
  (batman) for the sector, 2-minute exposures (0.00139 d). Partial transits and failed fits are rejected with the
  reason, never used.
- Bad measurements are removed before any fit, with the reason in `results.json`, by two rules that never look
  across observing seasons (so they cannot help a forecast pass): a transit whose own light-curve fit has a reduced
  chi-square over 3 and over 3 times the planet's median; a mid-time more than 5 robust sigma off the rest of its
  own season (the period from a Theil-Sen fit over all transits, the season's offset its own median).
- A transit_times call fits one transit shape per sector and needs three usable transits in it, so sectors with
  fewer (long periods, gaps) give no times; that is why several planets were dropped.
- Only sectors observed after the published reference mid-time are used, so the new times are not the ones the
  published t0 was measured from. Some published fits (for example ExoClock's Kokori et al. 2022 and 2023) already
  used early TESS sectors; for those the prior and the new times are not fully independent, and the prior's
  weight is small next to the new times either way.
- Refresh: weighted least squares of t(N) = t0 + N P on the TESS times (errors scaled by their own scatter, never
  down) plus the published t0 and P as independent Gaussian prior rows, used only when they agree with the TESS
  times within 3 sigma (otherwise dropped, and the reason recorded). Epochs are counted from the published reference
  transit and assigned season by season, each against the ephemeris refitted on the earlier seasons.
- Held-out check: fitted on every observing season but the last (plus the prior), it must forecast the last
  season's transits with a chi-square per transit of at most 4 (`timing.HELDOUT_CHI2`), each residual over the
  forecast's own uncertainty and the measurement's. Only planets that pass are registered, and only when the TESS
  mid-times scatter at most 3 times their own errors (`ephemeris.MAX_ERROR_SCALE`). That last rule was added after
  KELT-8 b passed the held-out check only because its errors were inflated 7.8 times (its mid-times alternate by
  about 15 minutes between odd and even transits, a systematic not understood here); it is applied to every
  planet alike, and it can only withhold a registration, never make one.
- The archive's composite row can take t0 and P from different papers (HAT-P-3 b: t0 from Kokori et al. 2022, P
  from Stassun et al. 2017); the large uncertainty at 2027 often comes from the older period. Both references are
  recorded per planet.
- Intervals: 68% is 1 sigma and 95% is 1.96 sigma of the fit covariance propagated to each epoch. They assume the
  period is constant; a planet with timing variations or a changing period will miss, and that is what scoring
  will show.

## What ExoClock and ETD accept (checked 2026-11-02; nothing was submitted to either)

- ExoClock (https://www.exoclock.space): observers register an observatory and upload light curves of their own
  ground-based observations ("Upload Observation", https://www.exoclock.space/contribute; HOPS recommended, other
  software allowed); the team reviews them. There is no public form for submitting an ephemeris or bare
  mid-times; space-telescope and literature mid-times are compiled by the team (its database pages "Space
  Observations" and "Literature Mid-times"). Its planets database (https://www.exoclock.space/database/planets,
  JSON at https://www.exoclock.space/database/planets_json) describes each ephemeris with `t0_bjd_tdb`, `t0_unc`,
  `period_days`, `period_unc`, `duration_hours`, `depth_mmag`, and states that all T0 values are BJD_TDB. The CSV
  here uses those names so a reviewer could compare directly.
- Exoplanet Transit Database, now part of VarAstro (https://var.astro.cz/ETD; FAQ
  https://var.astro.cz/var-astro/doc/VarAstro_faq_en_01.pdf): uploads are photometry, "a text format, where the
  first column is the Julian date (full or reduced) and the second column is the star's brightness value
  (magnitude or flux)", or the AAVSO format, with bulk upload for logged-in users; the transit is then fitted
  online and its mid-time, duration and depth stored. It does not take bare mid-times or ephemerides as input.
- So neither takes a refreshed ephemeris directly. What either could use: `refreshed_ephemerides.csv` as a
  reference to compare with its own ephemerides, and `tess_midtimes.csv` (BJD_TDB mid-times with errors) as space
  mid-times of the kind ExoClock compiles. The TESS photometry itself is public at MAST.
