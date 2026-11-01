"""Point sources and streaks: detection, PSF centroids with formal uncertainties, and a trailed-source model.

A mask centroid (what SAM gives) averages the pixels inside an outline, so its error is a sizeable fraction of a pixel
even on a clean star; fitting the star's light profile uses every pixel's brightness and reaches a few hundredths of a
pixel at good signal to noise. mask_centroid() is kept here only to measure that difference on the same frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class Source:
    x: float  # pixel column (0-based, pixel centres at integers)
    y: float
    sigma_x: float  # formal 1-sigma uncertainty of the centroid, pixels
    sigma_y: float
    flux: float
    fwhm: float
    snr: float


@dataclass(frozen=True)
class Trail:
    x0: float  # midpoint
    y0: float
    length: float  # pixels, end to end
    angle: float  # radians, from +x towards +y
    flux: float
    width: float  # Gaussian sigma across the trail, pixels
    sigma_x0: float
    sigma_y0: float
    profile: np.ndarray  # brightness along the trail (counts per pixel of length), start to end

    @property
    def endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        dx, dy = 0.5 * self.length * math.cos(self.angle), 0.5 * self.length * math.sin(self.angle)
        return (self.x0 - dx, self.y0 - dy), (self.x0 + dx, self.y0 + dy)


def background(image: np.ndarray) -> tuple[float, float]:
    """Sky level and noise from sigma-clipped statistics."""
    from astropy.stats import sigma_clipped_stats
    _, median, std = sigma_clipped_stats(image, sigma=3.0)
    return float(median), float(std)


def detect(image: np.ndarray, fwhm: float = 3.0, threshold_sigma: float = 5.0, max_sources: int = 2000) -> list[tuple[float, float]]:
    """Rough positions of point-like sources (photutils DAOStarFinder), brightest first."""
    from photutils.detection import DAOStarFinder
    sky, noise = background(image)
    found = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * max(noise, 1e-9))(image - sky)
    if found is None:
        return []
    order = np.argsort(-np.asarray(found["flux"]))[:max_sources]
    return [(float(found["xcentroid"][i]), float(found["ycentroid"][i])) for i in order]


def _gauss2d(p: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    amp, x0, y0, s, bg = p
    return amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * s * s)) + bg


def gaussian_centroid(image: np.ndarray, x: float, y: float, box: int = 11, noise: float | None = None) -> Source:
    """Fit a circular 2D Gaussian plus a flat sky in a box around (x, y); uncertainties from the fit's covariance
    scaled by the pixel noise."""
    h = box // 2
    xi, yi = round(x), round(y)
    H, W = image.shape
    x1, x2, y1, y2 = max(0, xi - h), min(W, xi + h + 1), max(0, yi - h), min(H, yi + h + 1)
    cut = image[y1:y2, x1:x2].astype(float)
    if cut.size < 9:
        raise ValueError("the source is too close to the edge to fit")
    yy, xx = np.mgrid[y1:y2, x1:x2]
    bg0 = float(np.median(np.concatenate([cut[0], cut[-1], cut[:, 0], cut[:, -1]])))
    p0 = np.array([max(cut.max() - bg0, 1e-6), x, y, 1.5, bg0])
    lo = [0.0, x1 - 0.5, y1 - 0.5, 0.3, -np.inf]
    hi = [np.inf, x2 - 0.5, y2 - 0.5, box, np.inf]
    fit = least_squares(lambda p: (_gauss2d(p, xx, yy) - cut).ravel(), p0, bounds=(lo, hi), x_scale="jac")
    amp, x0, y0, s, _bg = fit.x
    resid = fit.fun
    sigma_pix = noise if noise is not None else float(np.std(resid) * math.sqrt(cut.size / max(cut.size - 5, 1)))
    J = fit.jac
    try:
        cov = np.linalg.inv(J.T @ J) * sigma_pix**2
        sx, sy = math.sqrt(max(cov[1, 1], 0.0)), math.sqrt(max(cov[2, 2], 0.0))
    except np.linalg.LinAlgError:
        sx = sy = math.inf
    flux = 2 * math.pi * amp * s * s
    snr = flux / (sigma_pix * math.sqrt(4 * math.pi * s * s) + 1e-12)
    return Source(float(x0), float(y0), sx, sy, float(flux), float(2.3548 * s), float(snr))


def mask_centroid(image: np.ndarray, x: float, y: float, box: int = 11, level: float = 0.5) -> tuple[float, float]:
    """The centre of a thresholded outline, as a segmentation mask gives it: every pixel above `level` of the peak
    (over sky) counts the same. For comparison only."""
    h = box // 2
    xi, yi = round(x), round(y)
    cut = image[yi - h:yi + h + 1, xi - h:xi + h + 1].astype(float)
    sky = float(np.median(np.concatenate([cut[0], cut[-1], cut[:, 0], cut[:, -1]])))
    mask = (cut - sky) >= level * (cut.max() - sky)
    ys, xs = np.nonzero(mask)
    return float(xs.mean() + xi - h), float(ys.mean() + yi - h)


def _trail_model(p: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """A line segment of uniform brightness blurred by a Gaussian PSF (analytic: erf along the trail)."""
    from scipy.special import erf
    x0, y0, length, angle, flux, width, bg = p
    c, s = math.cos(angle), math.sin(angle)
    u = (xx - x0) * c + (yy - y0) * s  # along the trail
    v = -(xx - x0) * s + (yy - y0) * c  # across it
    half = length / 2
    along = 0.5 * (erf((u + half) / (math.sqrt(2) * width)) - erf((u - half) / (math.sqrt(2) * width)))
    across = np.exp(-v * v / (2 * width * width)) / (math.sqrt(2 * math.pi) * width)
    return flux / max(length, 1e-6) * along * across + bg


def fit_trail(image: np.ndarray, x: float, y: float, length_guess: float, angle_guess: float, box: int | None = None) -> Trail:
    """A streak (a satellite, a fast asteroid, a meteor): midpoint, length, angle, flux and width, with the
    midpoint's formal uncertainty, plus the measured brightness along the trail (flares and tumbling show there)."""
    half_box = int(box // 2 if box else max(8, length_guess * 0.75 + 6))
    xi, yi = round(x), round(y)
    H, W = image.shape
    x1, x2, y1, y2 = max(0, xi - half_box), min(W, xi + half_box + 1), max(0, yi - half_box), min(H, yi + half_box + 1)
    cut = image[y1:y2, x1:x2].astype(float)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    bg0 = float(np.median(cut))
    p0 = np.array([x, y, length_guess, angle_guess, max(cut.sum() - bg0 * cut.size, 1.0), 1.5, bg0])
    lo = [x1, y1, 1.0, angle_guess - math.pi / 2, 0.0, 0.3, -np.inf]
    hi = [x2, y2, 2.5 * length_guess + 5, angle_guess + math.pi / 2, np.inf, 10.0, np.inf]
    fit = least_squares(lambda p: (_trail_model(p, xx, yy) - cut).ravel(), p0, bounds=(lo, hi), x_scale="jac")
    x0, y0, length, angle, flux, width, bg = fit.x
    sigma_pix = float(np.std(fit.fun))
    try:
        cov = np.linalg.inv(fit.jac.T @ fit.jac) * sigma_pix**2
        sx, sy = math.sqrt(max(cov[0, 0], 0.0)), math.sqrt(max(cov[1, 1], 0.0))
    except np.linalg.LinAlgError:
        sx = sy = math.inf
    # Brightness along the trail: sum across it in one-pixel steps from one end to the other.
    n = max(2, round(length))
    c, s = math.cos(angle), math.sin(angle)
    steps = np.linspace(-length / 2, length / 2, n)
    across = np.arange(-3 * width, 3 * width + 1, 1.0)
    profile = []
    for u in steps:
        px = x0 + u * c - across * s
        py = y0 + u * s + across * c
        ix, iy = np.clip(np.round(px).astype(int), 0, W - 1), np.clip(np.round(py).astype(int), 0, H - 1)
        profile.append(float(np.sum(image[iy, ix] - bg)))
    return Trail(float(x0), float(y0), float(length), float(angle), float(flux), float(width), sx, sy, np.asarray(profile))


def synthetic_star(shape: tuple[int, int], x: float, y: float, flux: float, fwhm: float, sky: float = 100.0) -> np.ndarray:
    """A noiseless star, each pixel the integral of the Gaussian over the pixel (so sub-pixel positions are exact)."""
    from scipy.special import erf
    s = fwhm / 2.3548
    H, W = shape
    xs = np.arange(W)
    ys = np.arange(H)
    fx = 0.5 * (erf((xs + 0.5 - x) / (math.sqrt(2) * s)) - erf((xs - 0.5 - x) / (math.sqrt(2) * s)))
    fy = 0.5 * (erf((ys + 0.5 - y) / (math.sqrt(2) * s)) - erf((ys - 0.5 - y) / (math.sqrt(2) * s)))
    return sky + flux * np.outer(fy, fx)


def accuracy_vs_snr(snrs: tuple[float, ...] = (5, 10, 20, 50, 100), trials: int = 60, fwhm: float = 3.0,
                    seed: int = 0) -> list[dict[str, float]]:
    """Centroid error (RMS over trials, pixels) of the PSF fit and of a mask centroid, against signal to noise, on
    stars at random sub-pixel positions. Pixel noise is fixed; the star's flux sets the SNR."""
    rng = np.random.default_rng(seed)
    noise = 5.0
    s = fwhm / 2.3548
    out = []
    for snr in snrs:
        flux = snr * noise * math.sqrt(4 * math.pi * s * s)
        err_psf, err_mask = [], []
        for _ in range(trials):
            x, y = 15 + rng.uniform(-0.5, 0.5), 15 + rng.uniform(-0.5, 0.5)
            img = synthetic_star((31, 31), x, y, flux, fwhm) + rng.normal(0, noise, (31, 31))
            src = gaussian_centroid(img, 15, 15, noise=noise)
            mx, my = mask_centroid(img, 15, 15)
            err_psf.append(math.hypot(src.x - x, src.y - y))
            err_mask.append(math.hypot(mx - x, my - y))
        out.append({"snr": float(snr), "psf_rms_px": float(np.sqrt(np.mean(np.square(err_psf)))),
                    "mask_rms_px": float(np.sqrt(np.mean(np.square(err_mask))))})
    return out
