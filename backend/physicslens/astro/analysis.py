"""Astro modes of the pipeline: calibrated sky positions or a time series in, a verified law out, no SAM.

  mode "astro_frames"  a moving object's sky positions (from centroids and a plate solution) over time: its motion on
                       the sky, in arcseconds on the tangent plane about its mean position, time in days
  mode "series"        one quantity over time (brightness, a timing residual): family "line", q = the value

From the numbers on, everything is the video pipeline's: the reasoning model (Muse Spark Open) proposes laws and may
write its own, the same fitting, held-out forecast, rolling origins and BIC decide, bootstrap refits give the
intervals. Every result carries provenance (data source, licence, time source and uncertainty, WCS RMS) and an
outcome card (export/outcome.py).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..agent.reasoning import propose_hypotheses_sampled
from ..agent.schemas import Scene
from ..costguard import CostGuardClient
from ..export.outcome import build_outcome
from ..physics.coords import Coordinates
from ..physics.custom import MAX_PARAMS_ORBIT, FreeParam, build
from ..physics.models import MODELS
from ..physics.uncertainty import bootstrap_full
from ..physics.verify import judge_candidates
from ..sigproc.summary import summarize
from ..sigproc.trajectory import Trajectory

MODES = ("video", "astro_frames", "series")
MIN_SAMPLES = 12


class AstroAnalysisError(ValueError):
    pass


def tangent_plane_arcsec(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Gnomonic projection about the mean position: (xi east, eta north) in arcseconds, and the projection centre."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    ra0 = float(np.angle(np.mean(np.exp(1j * ra))))
    dec0 = float(np.mean(dec))
    cosc = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    xi = np.cos(dec) * np.sin(ra - ra0) / cosc
    eta = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cosc
    return np.degrees(np.column_stack([xi, eta])) * 3600, (float(np.degrees(ra0) % 360), float(np.degrees(dec0)))


def even(t: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The fitter assumes even sampling: resample onto an even grid with the same number of points."""
    order = np.argsort(t)
    t, q = np.asarray(t, float)[order], np.asarray(q, float)[order]
    grid = np.linspace(t[0], t[-1], len(t))
    return grid, np.column_stack([np.interp(grid, t, q[:, i]) for i in range(q.shape[1])])


def _scene(name: str, family: str, note: str) -> Scene:
    """The scene, stated rather than read from an image: in astro modes there is no picture for a model to look at."""
    return Scene.model_construct(moving_object=name, moving_object_box=None, motion_family="other", second_object=None,
                                 second_object_box=None, reference_kind="none", reference_box=None, observations=note,
                                 lookalikes=False)


def analyze_coordinates(client: CostGuardClient, t: np.ndarray, q: np.ndarray, family: str, *, run_id: str,
                        name: str, frame_note: str, units: dict[str, str], provenance: dict[str, Any],
                        audience: str = "dev", truth: dict[str, Any] | None = None,
                        action: str | None = None, progress: Callable[[str], None] = lambda _m: None) -> dict[str, Any]:
    """The shared core: hypotheses from the reasoning model, then the pipeline's fitting and verification."""
    from ..pipeline import _candidates, fit_and_verify, fit_summary  # the video pipeline's own steps, reused
    if len(t) < MIN_SAMPLES:
        raise AstroAnalysisError(f"{len(t)} measurements: at least {MIN_SAMPLES} are needed to fit and verify a law")
    t, q = even(t, q)
    t = t - t[0]
    dt = float(t[1] - t[0])
    traj = Trajectory(t, q if q.shape[1] == 2 else np.column_stack([q[:, 0], np.zeros(len(q))]), np.arange(len(t)), 1 / dt)
    summary = {**summarize(traj, uncalibrated=False), "frame_note": frame_note, "units": units}
    trail: list[dict[str, Any]] = [{"step": "data", "provider": None, "model": "provenance",
                                    "text": f"{provenance.get('data_source')}; times from {provenance.get('time_source')} "
                                            f"(±{provenance.get('time_sigma_s', 0):.3g} s)."
                                            + (f" Plate solution RMS {provenance['wcs_rms_arcsec']:.3g} arcsec."
                                               if provenance.get("wcs_rms_arcsec") is not None else "")}]
    progress("Proposing hypotheses")
    sampled = propose_hypotheses_sampled(client, summary, _scene(name, family, frame_note), run_id, audience)
    usable = []
    for f in sampled.merged.free_models:  # a law written for other variables (say theta) can't be fitted to these
        try:
            build(family, f.title, f.acceleration, [FreeParam(**p.model_dump()) for p in f.params], None,
                  max_params=MAX_PARAMS_ORBIT)
            usable.append(f)
        except ValueError as e:
            trail.append({"step": "hypotheses", "provider": None, "model": "law parser",
                          "text": f"Left out the agent's law {f.title!r}: {e}"})
    # The family is the data's shape (positions on the sky, or one value in time), not a guess.
    hyp = sampled.merged.model_copy(update={"family": family, "free_models": usable})
    calls = [{"provider": c.provider, "model": c.model} for c in sampled.calls]
    ids, rationales = _candidates(hyp, None, MAX_PARAMS_ORBIT)
    ids = [m for m in ids if MODELS[m].family == family]
    trail.append({"step": "hypotheses", "provider": calls[0]["provider"] if calls else None,
                  "model": calls[0]["model"] if calls else None,
                  "text": f"Proposed {len(ids)} laws: " + ", ".join(MODELS[m].title for m in ids) + "."})
    progress("Fitting and verifying")
    coords = Coordinates(family, t, q if family == "plane" else q[:, :1], ("x", "y") if family == "plane" else ("s",), {})
    coords, verdict = fit_and_verify(traj, family, ids, coords)
    judged = judge_candidates(verdict) if verdict.ranked else {}
    result: dict[str, Any] = {
        "mode": "astro_frames" if family == "plane" else "series", "status": verdict.status, "reason": verdict.reason,
        "family": family, "units": units, "provenance": provenance, "llm_calls": calls, "trail": trail,
        "ranking": [{**fit_summary(f, rationales.get(f.model_id)), "status": judged.get(f.model_id, ("", ""))[0],
                     "why": judged.get(f.model_id, ("", ""))[1]} for f in verdict.ranked],
        "n_eff": verdict.n_eff, "summary": summary,
    }
    trail.append({"step": "verdict", "provider": None, "model": "held-out verification", "text": verdict.reason})
    lead = verdict.winner
    if lead is not None and coords is not None:
        spec = MODELS[lead.model_id]
        boot = bootstrap_full(spec, coords, lead, reps=40)
        result["equation"] = {"model_id": spec.id, "title": spec.title, "latex": spec.latex}
        result["parameters"] = [{"name": p.name, "latex": p.latex, "unit": p.unit, "value": boot.estimates[p.name].value,
                                 "std": boot.estimates[p.name].std,
                                 "ci95": [boot.estimates[p.name].ci_low, boot.estimates[p.name].ci_high]}
                                for p in spec.params]
        result["bootstrap_reps"] = boot.reps
        result["fit_state"] = {"model_id": spec.id, "params": lead.params, "y0": [float(v) for v in lead.y0], "ctx": {}}
    result["outcome"] = build_outcome(result, truth=truth, action=action)
    return result


def analyze_sky_track(client: CostGuardClient, times_utc, ra_deg: np.ndarray, dec_deg: np.ndarray, *, run_id: str,
                      name: str, provenance: dict[str, Any], **kw: Any) -> dict[str, Any]:
    """A moving object's sky positions (degrees) at times (astropy Time): its law on the sky."""
    t_days = (times_utc.tdb.jd - times_utc.tdb.jd[0]).astype(float)
    xy, centre = tangent_plane_arcsec(np.asarray(ra_deg, float), np.asarray(dec_deg, float))
    note = (f"Sky positions of {name}, projected on the tangent plane about RA {centre[0]:.5f} deg, Dec {centre[1]:.5f} "
            "deg: x east and y north in arcseconds; time in days (TDB). Wherever the numbers say m, read arcsec; "
            "wherever they say s, read day.")
    return analyze_coordinates(client, t_days, xy, "plane", run_id=run_id, name=name, frame_note=note,
                               units={"length": "arcsec", "time": "day"},
                               provenance={**provenance, "projection_centre_deg": list(centre)}, **kw)


def analyze_series(client: CostGuardClient, t: np.ndarray, values: np.ndarray, *, run_id: str, name: str,
                   value_unit: str, time_unit: str, provenance: dict[str, Any], **kw: Any) -> dict[str, Any]:
    """One quantity over time (brightness, a timing residual): its law in time."""
    note = (f"A time series: {name} in {value_unit} against time in {time_unit}. Wherever the numbers say m, read "
            f"{value_unit}; wherever they say s, read {time_unit}.")
    return analyze_coordinates(client, np.asarray(t, float), np.asarray(values, float)[:, None], "line", run_id=run_id,
                               name=name, frame_note=note, units={"length": value_unit, "time": time_unit},
                               provenance=provenance, **kw)
