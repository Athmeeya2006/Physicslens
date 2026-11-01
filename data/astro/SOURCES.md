# Astronomical data: sources, licences and corrections

Every file under `data/astro/` comes from a public source listed here, with the date it was fetched. Nothing is
fabricated. Values recalled from memory in the planning notes were checked against their sources before use;
corrections are listed at the end.

## Catalogue and services

| What | Where | Licence | Used for |
|---|---|---|---|
| Gaia DR3 (`gaiadr3.gaia_source`), queried by cone with astroquery, cached as `cache/gaia_<hash>.json` | ESA Gaia archive, https://gea.esac.esa.int/archive/ | CC BY-NC 3.0 IGO (https://www.cosmos.esa.int/web/gaia-users/license, checked 2026-11-01). Acknowledge: "This work has made use of data from the European Space Agency (ESA) mission Gaia, processed by the Gaia Data Processing and Analysis Consortium (DPAC)." | the reference stars a plate solution is refined against |
| astrometry.net (local `solve-field`, or the web service with the person's own key) | https://astrometry.net, API https://astrometry.net/doc/net/api.html (checked 2026-11-01) | the software is BSD-3 / GPL parts; images sent to the web service are processed by a third party: never send anything private | blind plate solving |

## Accounts a task needs (created by the person, never by PhysicsLens)

- astrometry.net web service: a free account for an API key (only for the web fallback; the local solver needs none).
- Later cases (see each case's SOURCES.md): Copernicus CDS (ERA5 winds), JSOC (SDO/HMI export), MPC observatory
  code (submitting astrometry), possibly Space-Track.

## Corrections to values recalled in the plan

- The Gaia data licence is CC BY-NC 3.0 IGO, not CC BY-SA (ESA licence page, 2026-11-01).
