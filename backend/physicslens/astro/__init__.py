"""Real astronomical data: telescope frames, sky videos and light curves, measured without SAM.

Point sources need a different measurement chain from a ball in a phone video: PSF centroids (a twentieth of a pixel
where a mask centroid manages about two), a plate solution that ties pixels to the star catalogue instead of an
assumed scale, and times with stated uncertainties. The laws are still proposed by the reasoning model and decided by
the same held-out verification (analysis.py).
"""
