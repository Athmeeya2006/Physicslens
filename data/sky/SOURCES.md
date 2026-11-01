# Sky data sources

Everything here was fetched by `scripts/sky_fetch.py` on 2026-10-31 from the free NASA/JPL Horizons API (https://ssd.jpl.nasa.gov/api/horizons.api, version 1.2) and is cached so the sky engine and its tests run offline.

| File | Horizons query | Content |
|---|---|---|
| `states_j2000.csv` | VECTORS, CENTER=500@10 (the Sun), REF_PLANE=ECLIPTIC, OUT_UNITS=AU-D, TIME_TYPE=TDB, at 2000-01-01 12:00 | Heliocentric state of every catalogue body at J2000. Planets beyond Earth and Pluto are their system barycentres (ids 4 to 9), so their moons' masses sit where the system's mass is; Mercury, Venus, Earth and the Moon are the bodies themselves (199, 299, 399, 301); Ceres is small body 1, Apophis 99942, Halley record 90000030 |
| `sun_ssb_j2000.csv` | the Sun (10), CENTER=500@0 (solar-system barycentre), same frame and epoch | For the barycentric frame |
| `jovian_j2000.csv` | Io, Europa, Ganymede, Callisto (501 to 504), CENTER=500@599 (Jupiter's centre), same frame and epoch | The Galilean moons around Jupiter |
| `validation.csv` | Earth (399), Jupiter barycentre (5), Saturn barycentre (6), CENTER=500@10, TLIST of J2000 + 365.25 k days, k = 0 to 10 | JPL's yearly positions for the ten-year check |
| `gm.json` | OBJ_DATA of each body | GM in km^3/s^2. A planet system's GM is the sum of its planet and major moons, each read from Horizons' physical data, parts listed: Mars 499 (Horizons lists no GM for Phobos and Deimos, recorded as missing), Jupiter 599 + 501 to 504, Saturn 699 + 601 to 606 and 608, Uranus 799 + 701 to 705, Neptune 899 + 801, Pluto 999 + 901 |

Jupiter's oblateness for the moon system, J2 = 0.01469643 at a reference radius of 71,492 km, is from the Juno gravity field (Iess et al. 2018, Nature 555, 220), written in `backend/physicslens/sky/bodies.py`.

Credit: NASA/JPL Solar System Dynamics (Horizons).
