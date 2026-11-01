"""Plate solving: which part of the sky a frame shows, as an astropy WCS, with the fit's RMS in arcseconds.

Three ways, in order:
  solve_field()   astrometry.net's local solve-field (blind; needs the program and its index files, see README)
  solve_web()     the astrometry.net web service with the person's own key (ASTROMETRY_NET_API_KEY in .env)
  refine_wcs()    from a rough pointing (a FITS header, or a previous frame) to an exact solution, by matching the
                  detected stars with a catalogue and fitting the WCS: also how every blind solution is polished
The reference catalogue is Gaia DR3 (gaia_stars), with queries cached under data/astro/cache/.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "astro" / "cache"
NOVA = "https://nova.astrometry.net"  # API: https://astrometry.net/doc/net/api.html (checked 2026-11-01)
INSTALL_HINT = ("solve-field is not installed. On Debian or Ubuntu: sudo apt install astrometry.net, then index files "
                "matching your field size (for 0.1 to 2 degree fields, the 4200-series, e.g. sudo apt install "
                "astrometry-data-tycho2 or download from http://data.astrometry.net/4200/). See the README.")


class PlateSolveError(RuntimeError):
    pass


@dataclass
class Solution:
    wcs: WCS
    rms_arcsec: float
    matched: int
    method: str  # solve-field | web | refine

    def summary(self) -> dict[str, float | int | str]:
        ra, dec = self.wcs.wcs.crval
        scale = float(np.mean(np.abs(np.linalg.svd(self.wcs.pixel_scale_matrix, compute_uv=False)))) * 3600
        return {"method": self.method, "ra_deg": float(ra), "dec_deg": float(dec), "scale_arcsec_per_px": scale,
                "rms_arcsec": float(self.rms_arcsec), "matched_stars": int(self.matched)}


def tan_wcs(ra_deg: float, dec_deg: float, scale_arcsec: float, rotation_deg: float, shape: tuple[int, int]) -> WCS:
    """A gnomonic (TAN) WCS centred on the image: the standard model of a camera on the sky."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra_deg, dec_deg]
    w.wcs.crpix = [shape[1] / 2 + 0.5, shape[0] / 2 + 0.5]
    s, r = scale_arcsec / 3600, math.radians(rotation_deg)
    w.wcs.cd = np.array([[-s * math.cos(r), s * math.sin(r)], [s * math.sin(r), s * math.cos(r)]])
    return w


def match_rms(wcs: WCS, xy: np.ndarray, sky: SkyCoord) -> float:
    pred = wcs.pixel_to_world(xy[:, 0], xy[:, 1])
    return float(np.sqrt(np.mean(pred.separation(sky).arcsec ** 2)))


def refine_wcs(xy: np.ndarray, catalog: SkyCoord, wcs0: WCS, radius_px: float = 15.0, iterations: int = 6,
               min_matches: int = 6) -> Solution:
    """Match detected sources (pixel x, y) to catalogue stars through the current WCS, refit the WCS on the pairs,
    and repeat with a shrinking match radius."""
    from scipy.spatial import cKDTree
    xy = np.asarray(xy, dtype=float)
    wcs = wcs0.deepcopy()
    # A header pointing is often tens of pixels off: first find the common shift by letting every nearby pair of
    # (source, catalogue star) vote on it, then match tightly around that shift.
    cx, cy = wcs.world_to_pixel(catalog)
    ok = np.isfinite(cx) & np.isfinite(cy)
    near = cKDTree(np.column_stack([cx[ok], cy[ok]])).query_ball_point(xy, r=4 * radius_px)
    votes = np.array([np.array([cx[ok][j], cy[ok][j]]) - xy[i] for i, js in enumerate(near) for j in js])
    if len(votes):
        hist, ex, ey = np.histogram2d(votes[:, 0], votes[:, 1], bins=int(8 * radius_px / 2),
                                      range=[[-4 * radius_px, 4 * radius_px]] * 2)
        ix, iy = np.unravel_index(np.argmax(hist), hist.shape)
        shift = np.array([(ex[ix] + ex[ix + 1]) / 2, (ey[iy] + ey[iy + 1]) / 2])
        wcs.wcs.crpix = wcs.wcs.crpix - shift  # the catalogue lands shift pixels away from the sources: move it back
    pairs = (np.empty(0, int), np.empty(0, int))
    for k in range(iterations):
        cx, cy = wcs.world_to_pixel(catalog)
        ok = np.isfinite(cx) & np.isfinite(cy)
        tree = cKDTree(np.column_stack([cx[ok], cy[ok]]))
        dist, idx = tree.query(xy, distance_upper_bound=radius_px)
        good = np.isfinite(dist)
        cat_idx = np.nonzero(ok)[0][idx[good]]
        src_idx = np.nonzero(good)[0]
        # One catalogue star per source: keep the closest when two sources claim the same star.
        order = np.argsort(dist[good])
        _, first = np.unique(cat_idx[order], return_index=True)
        keep = order[first]
        pairs = (src_idx[keep], cat_idx[keep])
        if len(pairs[0]) < min_matches:
            raise PlateSolveError(f"only {len(pairs[0])} stars matched the catalogue (need {min_matches}): the "
                                  "starting pointing or scale is too far off for refinement; solve blind first")
        wcs = fit_wcs_from_points((xy[pairs[0], 0], xy[pairs[0], 1]), catalog[pairs[1]], proj_point=SkyCoord(
            *wcs.wcs.crval, unit="deg"), projection="TAN")
        radius_px = max(2.0, radius_px / 2) if k < iterations - 1 else radius_px
    return Solution(wcs, match_rms(wcs, xy[pairs[0]], catalog[pairs[1]]), len(pairs[0]), "refine")


def _number(name: str, value: float | None, lo: float, hi: float) -> list[str]:
    if value is None:
        return []
    v = float(value)
    if not (math.isfinite(v) and lo <= v <= hi):
        raise PlateSolveError(f"{name} must be between {lo} and {hi}")
    return [f"{v:.8g}"]


def solve_field(image_path: Path | str, scale_low_arcsec: float | None = None, scale_high_arcsec: float | None = None,
                ra_deg: float | None = None, dec_deg: float | None = None, radius_deg: float | None = None,
                timeout_s: float = 120.0) -> Solution:
    """Blind solve with the local solve-field. Arguments are numbers checked here and passed as a list (no shell)."""
    exe = shutil.which("solve-field")
    if exe is None:
        raise PlateSolveError(INSTALL_HINT)
    src = Path(image_path)
    if not src.is_file():
        raise PlateSolveError(f"{src}: no such file")
    with tempfile.TemporaryDirectory(prefix="physicslens-solve-") as tmp:
        work = Path(tmp) / f"frame{src.suffix.lower()}"
        shutil.copy(src, work)  # a path we chose: nothing from the person's file name reaches the command line
        args = [exe, "--overwrite", "--no-plots", "--no-remove-lines", "--dir", tmp, "--cpulimit", f"{int(timeout_s)}"]
        if scale_low_arcsec is not None and scale_high_arcsec is not None:
            args += ["--scale-units", "arcsecperpix", "--scale-low", *_number("scale_low", scale_low_arcsec, 0.01, 3600),
                     "--scale-high", *_number("scale_high", scale_high_arcsec, 0.01, 3600)]
        if ra_deg is not None and dec_deg is not None:
            args += ["--ra", *_number("ra", ra_deg, 0, 360), "--dec", *_number("dec", dec_deg, -90, 90),
                     "--radius", *_number("radius", radius_deg if radius_deg is not None else 5.0, 0.01, 180)]
        args.append(str(work))
        try:
            subprocess.run(args, check=False, capture_output=True, timeout=timeout_s + 30)
        except subprocess.TimeoutExpired as e:
            raise PlateSolveError(f"solve-field took longer than {timeout_s:.0f} s") from e
        wcs_file, corr_file = work.with_suffix(".wcs"), work.with_suffix(".corr")
        if not wcs_file.is_file():
            raise PlateSolveError("solve-field found no solution (wrong scale hint, too few stars, or no index files "
                                  "covering this field)")
        wcs = WCS(fits.getheader(wcs_file))
        rms, n = math.nan, 0
        if corr_file.is_file():
            corr = fits.getdata(corr_file)
            # Matched pairs: field star positions against the index stars' sky positions.
            pred = wcs.pixel_to_world(corr["field_x"] - 1, corr["field_y"] - 1)
            sky = SkyCoord(corr["index_ra"], corr["index_dec"], unit="deg")
            rms, n = float(np.sqrt(np.mean(pred.separation(sky).arcsec ** 2))), len(corr)
    return Solution(wcs, rms, n, "solve-field")


def solve_web(image_path: Path | str, api_key: str | None = None, timeout_s: float = 300.0,
              scale_low_arcsec: float | None = None, scale_high_arcsec: float | None = None) -> Solution:
    """The astrometry.net web service with the person's own API key (free account; never one of ours).
    Uploads the image: never use it for anything private."""
    import httpx
    key = api_key or os.environ.get("ASTROMETRY_NET_API_KEY", "")
    if not key:
        raise PlateSolveError("no ASTROMETRY_NET_API_KEY: create a free account at nova.astrometry.net and put the "
                              "key from your profile in .env, or install solve-field")
    with httpx.Client(timeout=60) as http:
        login = http.post(f"{NOVA}/api/login", data={"request-json": json.dumps({"apikey": key})}).json()
        if login.get("status") != "success":
            raise PlateSolveError(f"astrometry.net login failed: {login.get('errormessage', login)}")
        req: dict[str, object] = {"session": login["session"], "publicly_visible": "n", "allow_commercial_use": "n"}
        if scale_low_arcsec and scale_high_arcsec:
            req.update(scale_units="arcsecperpix", scale_lower=float(scale_low_arcsec), scale_upper=float(scale_high_arcsec))
        with open(image_path, "rb") as fh:
            up = http.post(f"{NOVA}/api/upload", data={"request-json": json.dumps(req)},
                           files={"file": (Path(image_path).name, fh, "application/octet-stream")}).json()
        subid = up.get("subid")
        if subid is None:
            raise PlateSolveError(f"astrometry.net upload failed: {up}")
        deadline = time.monotonic() + timeout_s
        job = None
        while time.monotonic() < deadline:
            jobs = [j for j in http.get(f"{NOVA}/api/submissions/{subid}").json().get("jobs", []) if j]
            if jobs:
                job = jobs[0]
                status = http.get(f"{NOVA}/api/jobs/{job}").json().get("status")
                if status == "success":
                    break
                if status == "failure":
                    raise PlateSolveError("astrometry.net could not solve this image")
            time.sleep(5)
        else:
            raise PlateSolveError(f"astrometry.net did not finish within {timeout_s:.0f} s")
        wcs_bytes = http.get(f"{NOVA}/wcs_file/{job}", headers={"Referer": f"{NOVA}/api/login"}).content
    import io
    header = fits.Header.fromfile(io.BytesIO(wcs_bytes))
    return Solution(WCS(header), math.nan, 0, "web")  # RMS comes from refine_wcs against Gaia


def gaia_stars(ra_deg: float, dec_deg: float, radius_deg: float, g_max: float = 16.0) -> SkyCoord:
    """Gaia DR3 stars in a cone, brighter than G = g_max, cached (one query per field, ever)."""
    key = hashlib.sha256(f"{ra_deg:.5f},{dec_deg:.5f},{radius_deg:.4f},{g_max:.2f}".encode()).hexdigest()[:16]
    path = CACHE / f"gaia_{key}.json"
    if path.is_file():
        rows = json.loads(path.read_text())
    else:
        from astroquery.gaia import Gaia
        query = (f"SELECT ra, dec, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE phot_g_mean_mag < {float(g_max)} AND "
                 f"1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {float(ra_deg)}, {float(dec_deg)}, {float(radius_deg)}))")
        table = Gaia.launch_job_async(query).get_results()
        rows = {"ra": [float(v) for v in table["ra"]], "dec": [float(v) for v in table["dec"]],
                "g": [float(v) for v in table["phot_g_mean_mag"]], "query": query,
                "source": "ESA Gaia DR3 (gaiadr3.gaia_source) via the Gaia archive, CC BY-NC 3.0 IGO (cosmos.esa.int/web/gaia-users/license, checked 2026-11-01)"}
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows))
    return SkyCoord(rows["ra"], rows["dec"], unit="deg")
