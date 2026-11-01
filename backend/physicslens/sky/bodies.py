"""The catalogue: which bodies the sky engine knows, their JPL Horizons ids and how their masses are made up."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogueBody:
    name: str
    horizons: str  # the Horizons COMMAND for its position: a planet system's barycentre carries its moons' mass
    gm_parts: tuple[str, ...]  # Horizons ids whose GMs add up to this body's (planet plus major moons)
    massive: bool  # False: a test particle (asteroid, comet) that feels every massive body and pulls on none
    colour: str  # for rendering only
    radius_km: float = 0.0  # for drawing and for collision warnings only


CATALOGUE: tuple[CatalogueBody, ...] = (
    CatalogueBody("Sun", "10", ("10",), True, "#ffcc33", 695_700),
    CatalogueBody("Mercury", "199", ("199",), True, "#a8a8a8", 2_440),
    CatalogueBody("Venus", "299", ("299",), True, "#e6c27a", 6_052),
    CatalogueBody("Earth", "399", ("399",), True, "#4f8fe6", 6_371),
    CatalogueBody("Moon", "301", ("301",), True, "#cfcfcf", 1_737),
    CatalogueBody("Mars", "4", ("499", "401", "402"), True, "#d2603a", 3_390),
    CatalogueBody("Jupiter", "5", ("599", "501", "502", "503", "504"), True, "#d9a066", 69_911),
    CatalogueBody("Saturn", "6", ("699", "601", "602", "603", "604", "605", "606", "608"), True, "#e3cf8a", 58_232),
    CatalogueBody("Uranus", "7", ("799", "701", "702", "703", "704", "705"), True, "#8fd6e3", 25_362),
    CatalogueBody("Neptune", "8", ("899", "801"), True, "#4a6fe0", 24_622),
    CatalogueBody("Pluto", "9", ("999", "901"), True, "#c9b79c", 1_188),
    CatalogueBody("Ceres", "1;", ("1;",), True, "#9c9c9c", 470),
    CatalogueBody("Apophis", "99942;", (), False, "#ff6b6b"),
    CatalogueBody("Halley", "90000030;", (), False, "#9be3c9"),
)
BY_NAME = {b.name: b for b in CATALOGUE}

# The Jovian system, simulated around Jupiter's centre (radii in Jupiter radii and hours in the demos).
GALILEAN = {"Io": "501", "Europa": "502", "Ganymede": "503", "Callisto": "504"}
GALILEAN_COLOURS = {"Io": "#e8c94a", "Europa": "#e9e2d2", "Ganymede": "#a89a86", "Callisto": "#7a6e60"}
# Jupiter's oblateness for the moons: J2 = 0.014696, reference radius 71,492 km (Iess et al. 2018, Nature 555,
# Juno gravity field, unnormalized J2 = 14696.43e-6 at R = 71492 km).
JUPITER_J2 = 14696.43e-6
JUPITER_J2_RADIUS_KM = 71_492.0

VALIDATION_BODIES = {"Earth": "399", "Jupiter": "5", "Saturn": "6"}
