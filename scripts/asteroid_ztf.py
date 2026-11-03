"""Astrometry from real telescope frames: Apophis measured on public ZTF images, against JPL.

  python scripts/asteroid_ztf.py        # network: IRSA (ZTF), JPL Horizons, the Gaia archive; cached under var/astro/ztf/

The asteroid follow-up page starts from published positions; this closes the loop from pixels. For each ZTF science
frame (Palomar 48-inch, MPC code I41) that contains 99942 Apophis during its 2021 apparition:
1. a cutout around JPL's predicted position at mid-exposure (IRSA image service);
2. the frame's WCS refined against Gaia DR3 stars detected in the cutout (astro/platesolve.refine_wcs), with its RMS;
3. the asteroid's centroid (a 2D Gaussian, astro/centroid.gaussian_centroid) on the source nearest the prediction;
4. measured RA/Dec minus JPL's astrometric position from I41 at that instant, in arcsec.
The prompt's target: residuals comparable to the WCS RMS. ZTF data: public data releases (Bellm et al. 2019, Masci et
al. 2019), via IRSA; Gaia DR3 (ESA). Writes data/astro/asteroids/ztf.json.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from physicslens.astro import centroid as C
from physicslens.astro import platesolve as P

CACHE = ROOT / "var" / "astro" / "ztf"
OUT = ROOT / "data" / "astro" / "asteroids" / "ztf.json"
DESIG = "99942"
START, STOP = "2021-02-10", "2021-03-20"  # Apophis near magnitude 15.5 to 16.4, well inside ZTF's reach
SIZE_ARCSEC = 360
MAX_FRAMES = 40


def get(url: str, name: str, binary: bool = False):
    p = CACHE / name
    if not p.is_file():
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=180) as r:
            p.write_bytes(r.read())
    return p.read_bytes() if binary else p.read_text()


def horizons(times_jd: list[float] | None = None, start: str = START, stop: str = STOP, step: str = "1 h") -> list[tuple[float, float, float]]:
    """JPL's astrometric RA, Dec (degrees) of Apophis from Palomar (I41): on a grid, or at given UTC Julian days."""
    q = {"format": "json", "COMMAND": f"'{DESIG}'", "OBJ_DATA": "NO", "MAKE_EPHEM": "YES", "EPHEM_TYPE": "OBSERVER",
         "CENTER": "'I41@399'", "QUANTITIES": "'1'", "ANG_FORMAT": "DEG", "EXTRA_PREC": "YES", "TIME_DIGITS": "FRACSEC",
         "CAL_FORMAT": "JD"}
    if times_jd:
        q["TLIST"] = " ".join(f"'{t:.9f}'" for t in times_jd)
        q["TLIST_TYPE"] = "JD"
        q["TIME_TYPE"] = "UT"
    else:
        q.update(START_TIME=f"'{start}'", STOP_TIME=f"'{stop}'", STEP_SIZE=f"'{step}'")
    url = "https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(q)
    key = f"hz_{abs(hash(url)) % 10**12}.json"
    txt = json.loads(get(url, key))["result"]
    body = txt[txt.index("$$SOE") + 5:txt.index("$$EOE")]
    out = []
    for line in body.strip().splitlines():
        parts = line.split()
        nums = [p for p in parts if p.replace(".", "", 1).replace("-", "", 1).replace("+", "", 1).isdigit()]
        out.append((float(nums[0]), float(nums[1]), float(nums[2])))
    return out


def frames() -> list[dict]:
    """ZTF science frames containing Apophis: one search per hour of the ephemeris, within half an hour of it."""
    rows, seen = [], set()
    for jd, ra, dec in horizons():
        url = ("https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci?" + urllib.parse.urlencode(
            {"POS": f"{ra:.5f},{dec:.5f}", "WHERE": f"obsjd BETWEEN {jd - 1 / 48:.6f} AND {jd + 1 / 48:.6f}", "ct": "csv"}))
        for r in csv.DictReader(io.StringIO(get(url, f"irsa_{jd:.4f}.csv"))):
            k = (r["filefracday"], r["field"], r["ccdid"], r["qid"])
            if k not in seen and r["imgtypecode"] == "o":
                seen.add(k)
                rows.append(r)
    return sorted(rows, key=lambda r: float(r["obsjd"]))


def cutout(r: dict, ra: float, dec: float) -> Path:
    f = r["filefracday"]
    name = f"ztf_{f}_{int(r['field']):06d}_{r['filtercode']}_c{int(r['ccdid']):02d}_o_q{r['qid']}_sciimg.fits"
    url = (f"https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/{f[:4]}/{f[4:8]}/{f[8:]}/{name}?"
           + urllib.parse.urlencode({"center": f"{ra:.5f},{dec:.5f}", "size": f"{SIZE_ARCSEC}arcsec", "gzip": "false"}))
    get(url, name, binary=True)
    return CACHE / name


def measure(path: Path, ra_p: float, dec_p: float) -> dict:
    with fits.open(path) as h:
        hdu = next(x for x in h if x.data is not None and x.data.ndim == 2)
        img, hdr = np.asarray(hdu.data, float), hdu.header
    img = np.nan_to_num(img, nan=float(np.nanmedian(img)))
    wcs0 = WCS(hdr)
    xy = np.array(C.detect(img, fwhm=2.5, threshold_sigma=5.0, max_sources=400))
    if len(xy) < 8:
        raise ValueError(f"only {len(xy)} sources detected")
    # Centroid every detection, so the WCS rests on the same centroids as the asteroid.
    noise = C.background(img)[1]
    cen = []
    for x, y in xy:
        try:
            s = C.gaussian_centroid(img, x, y, box=11, noise=noise)
            cen.append((s.x, s.y))
        except Exception:  # an edge or a blend: that source is left out of the fit
            continue
    cen = np.array(cen)
    c0 = wcs0.pixel_to_world(img.shape[1] / 2, img.shape[0] / 2)
    stars = P.gaia_stars(float(c0.ra.deg), float(c0.dec.deg), SIZE_ARCSEC / 3600 * 0.75, g_max=19.0)
    # The asteroid is not a star: keep it out of the matching, then find it.
    px, py = wcs0.world_to_pixel(SkyCoord(ra_p, dec_p, unit="deg"))
    far = np.hypot(cen[:, 0] - px, cen[:, 1] - py) > 6
    sol = P.refine_wcs(cen[far], stars, wcs0, radius_px=6.0)
    px, py = sol.wcs.world_to_pixel(SkyCoord(ra_p, dec_p, unit="deg"))
    d = np.hypot(cen[:, 0] - px, cen[:, 1] - py)
    i = int(np.argmin(d))
    if d[i] > 8:  # 8 px is 8 arcsec: a JPL position is far better than that, so a miss means no detection
        raise ValueError(f"no source within 8 px of the prediction (nearest {d[i]:.1f} px)")
    s = C.gaussian_centroid(img, cen[i, 0], cen[i, 1], box=11, noise=noise)
    m = sol.wcs.pixel_to_world(s.x, s.y)
    return {"ra": float(m.ra.deg), "dec": float(m.dec.deg), "wcs_rms_arcsec": sol.rms_arcsec, "stars": sol.n_matched,
            "snr": float(s.snr), "centroid_sigma_px": float(np.hypot(s.sigma_x, s.sigma_y) / np.sqrt(2))}


def main() -> None:
    rows = frames()[:MAX_FRAMES]
    print(len(rows), "ZTF frames contain Apophis", flush=True)
    mids = [float(r["obsjd"]) + float(r["exptime"]) / 2 / 86400 for r in rows]
    jpl = horizons(times_jd=mids)
    results, skipped = [], []
    for r, mid, (_, ra_p, dec_p) in zip(rows, mids, jpl):
        try:
            m = measure(cutout(r, ra_p, dec_p), ra_p, dec_p)
        except Exception as e:  # a frame that cannot be measured is reported, never silently dropped
            skipped.append({"frame": r["filefracday"], "reason": str(e)[:160]})
            continue
        dra = ((m["ra"] - ra_p + 180) % 360 - 180) * np.cos(np.radians(dec_p)) * 3600
        ddec = (m["dec"] - dec_p) * 3600
        results.append({"frame": r["filefracday"], "filter": r["filtercode"], "utc": Time(mid, format="jd").isot,
                        "dra_arcsec": float(dra), "ddec_arcsec": float(ddec), **m})
        print(results[-1]["utc"], f"{dra:+.2f} {ddec:+.2f} arcsec, WCS rms {m['wcs_rms_arcsec']:.2f} ({m['stars']} stars)", flush=True)
    if not results:
        sys.exit(f"no frame measured: {skipped[:3]}")
    a = np.array([[x["dra_arcsec"], x["ddec_arcsec"]] for x in results])
    miss = np.hypot(a[:, 0], a[:, 1])
    out = {"generated": "scripts/asteroid_ztf.py", "object": "99942 Apophis", "telescope": "ZTF, Palomar 48-inch (MPC code I41)",
           "window": [START, STOP], "frames_found": len(rows), "frames_measured": len(results), "skipped": skipped,
           "residual_rms_arcsec": float(np.sqrt(np.mean(a ** 2))), "median_miss_arcsec": float(np.median(miss)),
           "mean_offset_arcsec": {"ra": float(a[:, 0].mean()), "dec": float(a[:, 1].mean())},
           "median_wcs_rms_arcsec": float(np.median([x["wcs_rms_arcsec"] for x in results])),
           "reference": "JPL Horizons astrometric RA/Dec from I41 at mid-exposure (UT)",
           "data": "ZTF public data via IRSA (Bellm et al. 2019; Masci et al. 2019); Gaia DR3 (ESA)", "frames": results}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("frames", "skipped")}, indent=1), f"skipped {len(skipped)}")


if __name__ == "__main__":
    main()
