# Transit timing data: sources

| File | What | Source | Licence and terms | Fetched |
|---|---|---|---|---|
| `wasp12b_yee2020.csv` | 158 WASP-12b mid-times (139 transits, 19 occultations), BJD_TDB, 1-sigma, epoch, original source | Table `timings_fullsource.tex` in the arXiv source of Yee et al. 2020, "The Orbit of WASP-12b Is Decaying", ApJL 888, L5 (doi:10.3847/2041-8213/ab5c16; arXiv:1911.09131) | published measurements, cited; the original observers are named per row | 2026-11-01 |
| `kepler88b_transits.csv` | Kepler-88 b (KOI-142.01) mid-times measured by PhysicsLens (`astro/transits.py`) from Kepler PDCSAP long-cadence light curves, quarters 0 to 17, with a transit model (batman) and per-transit uncertainty | NASA Kepler mission via MAST (lightkurve); light curves cached under `var/astro/kepler88/` (not committed) | NASA data, public domain; acknowledge the Kepler mission and MAST | 2026-11-01 |
| `kepler88b_rejected.json` | every epoch not used, with the reason (partial, no clear transit, off the timing curve) | derived | | 2026-11-01 |
| `kepler88b_shape.json` | the stacked transit shape (radius ratio, a/R*, inclination, limb darkening) | derived | | 2026-11-01 |
| `pscomppars_transiting.csv` | 4,815 transiting planets: reference mid-time, period and their uncertainties, duration, V magnitude | NASA Exoplanet Archive, table `pscomppars`, TAP query `select ... where tran_flag=1` | This research has made use of the NASA Exoplanet Archive, operated by Caltech under contract with NASA. Free to use with that acknowledgement | 2026-11-01 |
| `results.json`, `wasp12b_windows.csv`, `wasp12b_windows.ics` | verdicts, fitted laws, the hidden-planet search and transit windows | `scripts/timing_demo.py` | derived | generated |

## Values compared against (each checked at its source)

- WASP-12b: dP/dt = -29 ± 2 ms per year (Yee et al. 2020, abstract and arXiv page, checked 2026-11-01).
- Kepler-88 c (KOI-142c): P = 22.3383 (+0.0036 -0.0031) days, M = 215.9 (+7.6 -7.5) Earth masses; TTV period about
  630 days, amplitude above 10 hours; star 1.022 solar masses (Nesvorný et al. 2013, ApJ 777, 3,
  doi:10.1088/0004-637X/777/1/3, Tables 3 and 4 and section 3 of the arXiv source, checked 2026-11-01).
- The TTV super-period formula P^j = 1/|j/P' - (j-1)/P| is eq. (psuper) of Lithwick, Xie & Wu 2012
  (arXiv:1207.4192), checked in the paper's source.

## Notes and corrections

- The archive's period for KOI-142 b (10.91647 d) is a linear fit through ±13 hours of timing variations; the mean
  transit period over the Kepler mission comes out at 10.950 d here. The 10.954204 d in Nesvorný et al. is the
  osculating period of their dynamical model, not a transit-to-transit mean: using it to predict transits drifts
  by about an hour per orbit. Both are correct for what they are; the windows here track the measured times.
