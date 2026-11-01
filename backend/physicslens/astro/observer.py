"""Where the observer stands and what they therefore see: topocentric RA/Dec, altitude and azimuth, light time.

astropy does the hard parts. A position in the solar system becomes what a telescope on Earth points at by:
light-travel time (the body is seen where it was when the light left it), the observer's own position on the
spinning Earth (parallax: one Earth radius seen from the distance, 8.8 arcsec at 1 AU and about 15 arcmin for a
near-Earth asteroid at 0.01 AU) and aberration from the observer's velocity (up to about 20.5 arcsec), which astropy
applies in the GCRS and AltAz frames.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import (
    GCRS,
    ICRS,
    AltAz,
    CartesianRepresentation,
    EarthLocation,
    SkyCoord,
    get_body_barycentric,
)
from astropy.time import Time

C_AU_PER_DAY = 173.1446326742403


@dataclass(frozen=True)
class Site:
    lat_deg: float
    lon_deg: float
    elevation_m: float = 0.0
    name: str = "observer"

    def __post_init__(self) -> None:
        if not (-90 <= self.lat_deg <= 90 and -180 <= self.lon_deg <= 360 and -500 <= self.elevation_m <= 9000):
            raise ValueError("latitude -90 to 90, longitude -180 to 360 degrees, elevation -500 to 9000 m")

    @property
    def location(self) -> EarthLocation:
        return EarthLocation(lat=self.lat_deg * u.deg, lon=self.lon_deg * u.deg, height=self.elevation_m * u.m)


def altaz(coord: SkyCoord, t: Time, site: Site) -> SkyCoord:
    """Altitude and azimuth (no refraction) of a sky position for this site and time."""
    return coord.transform_to(AltAz(obstime=t, location=site.location))


def observer_barycentric_au(t: Time, site: Site) -> np.ndarray:
    """The observer's position relative to the solar-system barycentre (AU, ICRS axes): Earth's plus the site's."""
    earth = get_body_barycentric("earth", t).xyz.to(u.au).value
    geo = site.location.get_gcrs(t).cartesian.xyz.to(u.au).value
    return earth + geo


def apparent_from_ephemeris(position_au: Callable[[Time], np.ndarray], t: Time, site: Site,
                            iterations: int = 3) -> tuple[SkyCoord, float]:
    """Where a body appears from the site at time t, given its barycentric ICRS position (AU) as a function of TDB
    time: iterate the light-travel time, then let astropy add the observer's aberration (GCRS frame at the site).
    Returns the topocentric apparent coordinate and the light time in seconds."""
    obs = observer_barycentric_au(t, site)
    tau = 0.0
    for _ in range(iterations):
        body = position_au(t.tdb - tau * u.day)
        tau = float(np.linalg.norm(body - obs)) / C_AU_PER_DAY
    rel = body - obs
    astrometric = SkyCoord(CartesianRepresentation(rel * u.au), frame=ICRS())
    gcrs = GCRS(obstime=t, obsgeoloc=site.location.get_gcrs(t).cartesian.without_differentials(),
                obsgeovel=site.location.get_gcrs(t).velocity.to_cartesian())
    # astrometric (light time applied, barycentric axes) -> apparent from the moving observer (aberration)
    apparent = SkyCoord(astrometric.ra, astrometric.dec, frame=ICRS()).transform_to(gcrs)
    return apparent, tau * 86400.0
