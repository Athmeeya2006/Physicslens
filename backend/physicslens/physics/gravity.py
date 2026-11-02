"""Local gravity at a place on Earth, the truth a measured g is compared with.

"Standard gravity" (9.80665) is a convention, not the value anywhere in particular: g grows by 0.5% from the equator to
the poles and falls 0.03% per 100 m of height. A pendulum measured to 0.4% needs the local value, so a clip's truth is
computed from where it was filmed rather than written as 9.81.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Somigliana's closed form of the international gravity formula on the WGS 84 ellipsoid (NIMA TR8350.2, 3rd ed., 2000,
# eq. 4-1, with its constants g_e, k and e^2). Normal gravity on the ellipsoid, without local anomalies (a few parts in
# 10^5 for a lowland city, far below a pendulum's error).
G_EQUATOR = 9.7803253359
SOMIGLIANA_K = 0.00193185265241
E2 = 0.00669437999013
# Free-air gradient: -0.3086 mGal per metre (the standard linear correction; NIMA TR8350.2 eq. 4-3 to first order).
FREE_AIR_PER_M = 3.086e-6

SOURCE = "WGS 84 normal gravity (Somigliana's formula, NIMA TR8350.2) with the free-air correction for height"


@dataclass(frozen=True)
class Site:
    name: str
    lat_deg: float
    height_m: float
    note: str


# The pendulum and paper-bag clips are by Mathe-AC Lernwerkstatt, a teaching workshop in Aachen (the uploader's own
# description on Wikimedia Commons). The exact room is not given, so the city centre is used; across the city's height
# range (about 125 to 410 m) local g changes by under 0.01%, twenty times less than the pendulum's interval.
AACHEN = Site("Aachen, Germany", 50.776, 173.0, "city centre; the clip's room is not stated, the city spans 125 to 410 m")


def normal_gravity(lat_deg: float, height_m: float = 0.0) -> float:
    """g in m/s^2 at a geodetic latitude and height above the ellipsoid."""
    s2 = math.sin(math.radians(lat_deg)) ** 2
    g0 = G_EQUATOR * (1 + SOMIGLIANA_K * s2) / math.sqrt(1 - E2 * s2)
    return g0 - FREE_AIR_PER_M * height_m


def local_g(site: Site) -> float:
    return normal_gravity(site.lat_deg, site.height_m)
