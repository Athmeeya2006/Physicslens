"""The open dataset: only verified, measured, honestly scaled parameters get in."""

from physicslens.export.dataset import entries, to_csv

META = {"title": "Spring cart", "description": "On an air track", "credit": "CC BY"}


def result(status="accepted", held=0.05, **param):
    p = {"name": "zeta", "value": 0.02, "std": 0.001, "unit": "", "derived": None, "uncalibrated": False, **param}
    return {"status": status, "family": "oscillator", "fit_state": {"model_id": "damped_harmonic"},
            "ranking": [{"model_id": "damped_harmonic", "heldout_nrmse": held}], "parameters": [p],
            "calibration": {"source": "none"}, "scene": {"moving_object": "cart"}}


def test_a_verified_scale_free_parameter_enters_with_its_provenance():
    rows = entries("r1", result(), META)
    assert len(rows) == 1 and rows[0]["property"] == "damping_ratio" and rows[0]["license"] == "CC BY"
    assert "damping_ratio" in to_csv(rows)


def test_the_quality_gate_keeps_out_what_was_not_measured():
    assert entries("r1", result(status="ambiguous"), META) == []
    assert entries("r1", result(held=0.4), META) == []  # did not forecast its held-out motion
    assert entries("r1", result(uncalibrated=True), META) == []  # arbitrary units
    assert entries("r1", result(derived="given by you"), META) == []  # not measured
    assert entries("r1", result(name="s0"), META) == []  # no named physical property for it


def test_gm_from_a_law_the_agent_wrote_enters_as_the_central_body():
    from physicslens.export.dataset import entries
    result = {"status": "accepted", "fit_state": {"model_id": "custom_abc"}, "family": "plane",
              "ranking": [{"model_id": "custom_abc", "heldout_nrmse": 0.001}], "calibration": {"source": "known_scale"},
              "scene": {"moving_object": "white dot"},
              "parameters": [{"name": "mu", "value": 4.494, "std": 0.001, "unit": "R_J^3/hour^2"},
                             {"name": "b", "value": 1e-7, "std": 1e-8, "unit": "1/day"}]}
    rows = entries("gallery-jupiter", result, {"title": "Weighing Jupiter", "credit": "JPL", "description": "d"})
    assert [(r["property"], r["object"]) for r in rows] == [("gravitational_parameter", "central body")]
