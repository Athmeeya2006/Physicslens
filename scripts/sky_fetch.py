"""Fetch the solar-system catalogue's initial states and masses from NASA/JPL Horizons into data/sky/ (free API).

  python scripts/sky_fetch.py

Everything the sky engine (backend/physicslens/sky/) needs is cached here so the app and the tests run offline:
  states_j2000.csv     heliocentric state of every catalogue body at J2000 (2000-01-01 12:00 TDB), ecliptic J2000
  sun_ssb_j2000.csv    the Sun's state relative to the solar-system barycentre at the same epoch
  jovian_j2000.csv     the four Galilean moons relative to Jupiter (planet centre), same epoch and frame
  validation.csv       yearly heliocentric positions 2000 to 2010 of Earth, Jupiter and Saturn, for the 10-year check
  gm.json              GM of every massive body: a planet system's mass is its planet plus its major moons, each
                       from Horizons' physical data, with the parts listed
"""

from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.sky.bodies import CATALOGUE, GALILEAN, VALIDATION_BODIES

DATA = ROOT / "data" / "sky"
HORIZONS = "https://ssd.jpl.nasa.gov/api/horizons.api"
EPOCH = "2000-01-01 12:00"
EPOCH_STOP = "2000-01-01 12:01"
J2000 = 2451545.0


def _get(params: dict[str, str]) -> dict:
    with urllib.request.urlopen(f"{HORIZONS}?{urllib.parse.urlencode(params)}", timeout=180) as r:
        return json.loads(r.read())


def vectors(command: str, center: str, start: str = "", stop: str = "", step: str = "",
            tlist: list[float] | None = None) -> tuple[np.ndarray, str]:
    span = ({"TLIST": "'" + " ".join(f"{t:.4f}" for t in tlist) + "'", "TLIST_TYPE": "'JD'"} if tlist else
            {"START_TIME": f"'{start}'", "STOP_TIME": f"'{stop}'", "STEP_SIZE": f"'{step}'"})
    d = _get({"format": "json", "COMMAND": f"'{command}'", "EPHEM_TYPE": "VECTORS", "CENTER": f"'{center}'", **span,
              "VEC_TABLE": "'2'", "REF_PLANE": "'ECLIPTIC'", "OUT_UNITS": "'AU-D'", "CSV_FORMAT": "'YES'",
              "TIME_TYPE": "'TDB'"})
    text = d["result"]
    if "$$SOE" not in text:
        sys.exit(f"Horizons gave no vectors for {command}: {text[:300]}")
    rows = []
    for line in text.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines():
        f = [v.strip() for v in line.split(",")]
        rows.append([float(f[0]), *(float(v) for v in f[2:8])])
    return np.array(rows), f"{d['signature']['source']} {d['signature']['version']}"


GM_PATTERN = re.compile(r"GM\s*(?:\(planet\))?[ ,(]*km\^3/s\^2\)?\s*=\s*([\d.]+)")
GM_SMALL = re.compile(r"\bGM=\s*([\d.]+)")


def gm(command: str, required: bool = True) -> float | None:
    txt = _get({"format": "json", "COMMAND": f"'{command}'", "OBJ_DATA": "'YES'", "MAKE_EPHEM": "'NO'"})["result"]
    m = GM_PATTERN.search(txt) or GM_SMALL.search(txt)
    if not m:
        if required:
            sys.exit(f"Horizons gave no GM for {command}")
        return None  # a small moon Horizons lists no GM for (Phobos): left out, and recorded as missing
    return float(m.group(1))


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    today = f"{datetime.now(UTC):%Y-%m-%d}"
    rows, version = [], ""
    for b in CATALOGUE:
        if b.name == "Sun":
            continue
        table, version = vectors(b.horizons, "500@10", EPOCH, EPOCH_STOP, "1 m")
        rows.append([b.name, *table[0]])
        print(f"{b.name}: state", flush=True)
    header = (f"# JPL Horizons API ({version}), fetched {today}, heliocentric (CENTER=500@10), ecliptic J2000, "
              "AU and AU/day, TDB\n# name,jd_tdb,x,y,z,vx,vy,vz")
    with (DATA / "states_j2000.csv").open("w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join([r[0], *(f"{v:.17g}" for v in r[1:])]) + "\n")
    sun, version = vectors("10", "500@0", EPOCH, EPOCH_STOP, "1 m")
    np.savetxt(DATA / "sun_ssb_j2000.csv", sun[:1], delimiter=",", comments="", fmt="%.17g",
               header=f"# JPL Horizons API ({version}), fetched {today}, the Sun relative to the solar-system "
                      "barycentre (CENTER=500@0), ecliptic J2000, AU and AU/day, TDB\n# jd_tdb,x,y,z,vx,vy,vz")
    moons = []
    for name, cmd in GALILEAN.items():
        table, version = vectors(cmd, "500@599", EPOCH, EPOCH_STOP, "1 m")
        moons.append([name, *table[0]])
    with (DATA / "jovian_j2000.csv").open("w") as f:
        f.write(f"# JPL Horizons API ({version}), fetched {today}, relative to Jupiter's centre (CENTER=500@599), "
                "ecliptic J2000, AU and AU/day, TDB\n# name,jd_tdb,x,y,z,vx,vy,vz\n")
        for r in moons:
            f.write(",".join([r[0], *(f"{v:.17g}" for v in r[1:])]) + "\n")
    with (DATA / "validation.csv").open("w") as f:
        f.write(f"# JPL Horizons API, fetched {today}, heliocentric positions every 365.25 days from J2000, "
                "ecliptic J2000, AU, TDB\n# name,jd_tdb,x,y,z\n")
        for name, cmd in VALIDATION_BODIES.items():
            table, _ = vectors(cmd, "500@10", tlist=[J2000 + 365.25 * k for k in range(11)])
            for r in table:
                f.write(",".join([name, *(f"{v:.17g}" for v in r[:4])]) + "\n")
            print(f"{name}: validation", flush=True)
    masses = {}
    for b in CATALOGUE:
        if not b.massive:
            continue
        parts = {c: gm(c, required=(i == 0)) for i, c in enumerate(b.gm_parts)}
        masses[b.name] = {"gm_km3_s2": sum(v for v in parts.values() if v is not None), "parts": parts,
                          "missing": [c for c, v in parts.items() if v is None]}
        print(f"{b.name}: GM {masses[b.name]['gm_km3_s2']:.6f} from {list(parts)}", flush=True)
    for name, cmd in GALILEAN.items():
        masses[name] = {"gm_km3_s2": gm(cmd), "parts": {cmd: None}}
        masses[name]["parts"][cmd] = masses[name]["gm_km3_s2"]
    masses["Jupiter (planet)"] = {"gm_km3_s2": gm("599"), "parts": {"599": gm("599")}}
    (DATA / "gm.json").write_text(json.dumps({
        "source": f"JPL Horizons API, physical data of each body, fetched {today}",
        "note": "A planet's system (barycentre) GM is the planet's plus its major moons'; parts list the Horizons ids",
        "bodies": masses}, indent=1) + "\n")


if __name__ == "__main__":
    main()
