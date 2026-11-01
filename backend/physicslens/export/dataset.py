"""An open dataset of real-world physical parameters, one row per verified measurement.

Only accepted results enter, only parameters that are measured (not assumed, not given by the user, not in arbitrary
units), and only when the law forecast its held-out motion within the acceptance threshold. Values depend on the
conditions they were measured in, so every row carries the clip, its licence and the run it came from.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Any

from ..physics.verify import MAX_HELDOUT_NRMSE

# Fitted parameter (by model) to a named physical property. Anything not listed is left out rather than guessed.
PROPERTIES: dict[tuple[str, str], tuple[str, str]] = {
    ("damped_harmonic", "zeta"): ("damping_ratio", "damping ratio of the oscillation"),
    ("damped_harmonic", "omega"): ("natural_frequency", "undamped angular frequency"),
    ("harmonic", "omega"): ("natural_frequency", "angular frequency"),
    ("oscillator_dry", "omega"): ("natural_frequency", "undamped angular frequency"),
    ("oscillator_mixed", "omega"): ("natural_frequency", "undamped angular frequency"),
    ("oscillator_mixed", "zeta"): ("damping_ratio", "viscous damping ratio, alongside dry friction"),
    ("pendulum", "g"): ("gravity", "gravitational acceleration from the swing period and the given length"),
    ("pendulum_linear_damping", "gamma"): ("joint_damping", "viscous damping rate of the swing"),
    ("pendulum_quadratic_damping", "beta"): ("drag_coefficient", "air-drag coefficient of the swing"),
    ("linear_drag", "k"): ("drag_rate", "linear drag rate (inverse of the velocity time constant)"),
    ("quadratic_drag", "c"): ("drag_coefficient", "quadratic drag coefficient per unit mass"),
    ("spin_dry", "c"): ("friction_dry", "dry friction deceleration of the spin"),
    ("spin_viscous", "k"): ("friction_viscous", "viscous friction rate of the spin"),
    ("spin_air", "q"): ("drag_coefficient", "air drag coefficient of the spin"),
}
FIELDS = ("id", "object", "property", "meaning", "value", "std", "units", "family", "model", "heldout_error",
          "scale_basis", "conditions", "run_id", "clip", "license")


def entries(run_id: str, result: dict[str, Any], meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Dataset rows for one stored result (empty if it doesn't pass the quality gate)."""
    if result.get("status") != "accepted" or not result.get("fit_state"):
        return []
    model = result["fit_state"]["model_id"]
    ranking = result.get("ranking") or []
    held = next((r.get("heldout_nrmse") for r in ranking if r.get("model_id") == model), None)
    if held is None or held > MAX_HELDOUT_NRMSE:
        return []
    cal = result.get("calibration") or {}
    rows = []
    for p in result.get("parameters", []):
        prop = PROPERTIES.get((model, p.get("name", ""))) or _free_law_property(model, p)
        if prop is None or p.get("uncalibrated") or p.get("derived") or p.get("std") is None:
            continue
        rows.append({
            # GM belongs to the body being orbited, not to the dot that was tracked; the source column names it.
            "id": f"{run_id}-{prop[0]}",
            "object": "central body" if prop[0] == "gravitational_parameter" else (result.get("scene") or {}).get("moving_object"),
            "property": prop[0], "meaning": prop[1], "value": p["value"], "std": p["std"], "units": p.get("unit", ""),
            "family": result.get("family"), "model": model, "heldout_error": held,
            "scale_basis": cal.get("source"), "conditions": meta.get("description", ""), "run_id": run_id,
            "clip": meta.get("title", ""), "license": meta.get("credit", ""),
        })
    return rows


def _free_law_property(model: str, p: dict[str, Any]) -> tuple[str, str] | None:
    """A law the agent wrote has no fixed parameter names; its GM is recognised by units of length^3 per time^2."""
    if not model.startswith("custom_"):
        return None
    unit = (p.get("unit") or "").replace(" ", "")
    if re.fullmatch(r"[A-Za-z_]+\^3/[A-Za-z]+\^2", unit):
        return ("gravitational_parameter", "gravitational parameter GM of the body at the centre of the orbit")
    return None


def to_csv(rows: list[dict[str, Any]]) -> str:
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return out.getvalue()


FIELD_TYPES = {"value": "sc:Float", "std": "sc:Float", "heldout_error": "sc:Float"}
FIELD_NOTES = {
    "id": "run id and property, unique", "object": "what was tracked", "property": "the property, from a fixed list",
    "meaning": "the property in words", "value": "fitted value", "std": "one standard deviation (bootstrap)",
    "units": "units of value and std", "family": "motion family", "model": "the law's id",
    "heldout_error": "forecast RMSE over the last quarter divided by the motion's standard deviation",
    "scale_basis": "where lengths came from", "conditions": "what was filmed and how", "run_id": "the gallery result",
    "clip": "the clip's title", "license": "the clip's source and licence",
}


def croissant(json_bytes: bytes, csv_bytes: bytes, site: str) -> dict[str, Any]:
    """Croissant 1.0 metadata (schema.org Dataset plus the ML Commons vocabulary) for the two published files."""
    import hashlib
    base = f"{site}/gallery"
    return {
        "@context": {
            "@language": "en", "@vocab": "https://schema.org/", "sc": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/", "rai": "http://mlcommons.org/croissant/RAI/",
            "dct": "http://purl.org/dc/terms/", "citeAs": "cr:citeAs", "column": "cr:column",
            "conformsTo": "dct:conformsTo", "data": {"@id": "cr:data", "@type": "@json"},
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"}, "examples": {"@id": "cr:examples", "@type": "@json"},
            "extract": "cr:extract", "field": "cr:field", "fileObject": "cr:fileObject", "fileProperty": "cr:fileProperty",
            "format": "cr:format", "includes": "cr:includes", "isLiveDataset": "cr:isLiveDataset", "jsonPath": "cr:jsonPath",
            "key": "cr:key", "md5": "cr:md5", "parentField": "cr:parentField", "path": "cr:path", "recordSet": "cr:recordSet",
            "references": "cr:references", "regex": "cr:regex", "repeated": "cr:repeated", "replace": "cr:replace",
            "separator": "cr:separator", "source": "cr:source", "subField": "cr:subField", "transform": "cr:transform",
        },
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": "physicslens-open-dataset",
        "description": ("Physical parameters measured from video by PhysicsLens, one row per parameter, each from a law "
                        "that forecast motion it was never fitted on. Every row names its clip, conditions and run."),
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "url": f"{site}/dataset",
        "citeAs": "PhysicsLens open dataset: physical parameters measured from video. https://physicslens.vercel.app/dataset",
        "keywords": ["system identification", "physics", "video", "parameter estimation", "sim-to-real"],
        "isLiveDataset": True,
        "distribution": [
            {"@type": "cr:FileObject", "@id": "dataset-csv", "name": "dataset.csv", "contentUrl": f"{base}/dataset.csv",
             "encodingFormat": "text/csv", "sha256": hashlib.sha256(csv_bytes).hexdigest()},
            {"@type": "cr:FileObject", "@id": "dataset-json", "name": "dataset.json", "contentUrl": f"{base}/dataset.json",
             "encodingFormat": "application/json", "sha256": hashlib.sha256(json_bytes).hexdigest()},
        ],
        "recordSet": [{
            "@type": "cr:RecordSet", "@id": "measurements", "name": "measurements", "key": {"@id": "measurements/id"},
            "field": [{"@type": "cr:Field", "@id": f"measurements/{f}", "name": f, "description": FIELD_NOTES[f],
                       "dataType": FIELD_TYPES.get(f, "sc:Text"),
                       "source": {"fileObject": {"@id": "dataset-csv"}, "extract": {"column": f}}} for f in FIELDS],
        }],
    }
