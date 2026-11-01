"""Brightness over time: aperture photometry with uncertainties, comparison stars, and event timing.

event_timing() fits the mid-time of a dip (a transit, an eclipse, an occultation) with a trapezoid: flat, a linear
ingress, a flat bottom, a linear egress. Its uncertainty is the fit's, widened by the residuals' own scatter and the
clock's uncertainty added in quadrature, so a typed start time (low trust) shows in the answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class Event:
    mid_time: float  # same units as the input times (e.g. BJD or seconds)
    sigma: float  # 1-sigma, fit and clock combined
    sigma_fit: float
    sigma_clock: float
    depth: float  # fraction of the out-of-event flux
    duration: float  # first to last contact
    ingress: float
    rms: float

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


def aperture_series(images: list[np.ndarray], positions: list[tuple[float, float]], radius: float = 4.0,
                    annulus: tuple[float, float] = (7.0, 11.0), gain: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Flux (counts) and its 1-sigma uncertainty for each frame at each position, sky from an annulus (photutils).
    Returns arrays of shape (frames, positions)."""
    from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture, aperture_photometry
    flux = np.zeros((len(images), len(positions)))
    err = np.zeros_like(flux)
    for k, img in enumerate(images):
        ap = CircularAperture(positions, r=radius)
        an = CircularAnnulus(positions, r_in=annulus[0], r_out=annulus[1])
        sky = ApertureStats(img, an)
        raw = aperture_photometry(img, ap)["aperture_sum"]
        npix = ap.area
        f = np.asarray(raw) - np.asarray(sky.median) * npix
        var = np.clip(f, 0, None) / gain + npix * np.asarray(sky.std) ** 2 * (1 + npix / an.area)
        flux[k], err[k] = f, np.sqrt(var)
    return flux, err


def relative(flux: np.ndarray, err: np.ndarray, target: int, comparisons: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Target divided by the summed comparison stars (removes clouds and airmass common to all), normalised to its
    median, with propagated uncertainty."""
    comp = flux[:, comparisons].sum(axis=1)
    comp_err = np.sqrt((err[:, comparisons] ** 2).sum(axis=1))
    ratio = flux[:, target] / comp
    r_err = np.abs(ratio) * np.sqrt((err[:, target] / flux[:, target]) ** 2 + (comp_err / comp) ** 2)
    norm = np.median(ratio)
    return ratio / norm, r_err / norm


def trapezoid(t: np.ndarray, mid: float, depth: float, duration: float, ingress: float, base: float = 1.0) -> np.ndarray:
    x = np.abs(t - mid)
    half = duration / 2
    ing = np.clip(ingress, 1e-9, half)
    out = np.full_like(t, base, dtype=float)
    bottom = x <= half - ing
    slope = (x > half - ing) & (x < half)
    out[bottom] = base - depth
    out[slope] = base - depth * (half - x[slope]) / ing
    return out


def event_timing(t: np.ndarray, flux: np.ndarray, flux_err: np.ndarray | None = None, clock_sigma: float = 0.0,
                 guess_mid: float | None = None) -> Event:
    """Mid-time of a dip with its uncertainty (clock_sigma in the units of t, e.g. days)."""
    t = np.asarray(t, dtype=float)
    f = np.asarray(flux, dtype=float)
    w = 1.0 / np.asarray(flux_err, dtype=float) if flux_err is not None else np.ones_like(f)
    base0 = float(np.median(f))
    mid0 = guess_mid if guess_mid is not None else float(t[np.argmin(f)])
    depth0 = max(base0 - float(np.min(f)), 1e-6)
    below = t[f < base0 - depth0 / 2]
    dur0 = float(np.ptp(below)) * 1.3 if len(below) > 1 else float(np.ptp(t)) / 5
    span = float(np.ptp(t))
    p0 = np.array([mid0, depth0, dur0, dur0 / 4, base0])
    lo = [t.min(), 0.0, 3 * np.median(np.diff(t)), 1e-9, base0 - 1]
    hi = [t.max(), 1.0, span, span / 2, base0 + 1]
    fit = least_squares(lambda p: (trapezoid(t, *p) - f) * w, p0, bounds=(lo, hi), x_scale="jac")
    mid, depth, dur, ing, base = fit.x
    model = trapezoid(t, *fit.x)
    rms = float(np.std(f - model))
    J = fit.jac
    try:
        cov = np.linalg.inv(J.T @ J)
        chi2 = float(np.sum(fit.fun ** 2) / max(len(t) - 5, 1))
        sigma_fit = float(np.sqrt(max(cov[0, 0] * max(chi2, 1.0), 0.0)))
    except np.linalg.LinAlgError:
        sigma_fit = float("inf")
    sigma = float(np.hypot(sigma_fit, clock_sigma))
    return Event(float(mid), sigma, sigma_fit, float(clock_sigma), float(depth / base), float(dur), float(ing), rms)
