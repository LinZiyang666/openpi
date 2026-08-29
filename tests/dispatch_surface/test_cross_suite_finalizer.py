"""Cross-suite finalizer: four Gates across both suites, or no general claim."""

from __future__ import annotations

import pytest

from exp.dispatch_surface.analysis.finalize_cross_suite import (
    GATE_SCHEMA,
    PROTOCOL,
    REQUIRED_SUITES,
    check_one,
    finalize,
)


def _verdict(suite, *, g1=True, g2=True, **extra):
    v = {
        "protocol": PROTOCOL,
        "analysis_layer": "primary",
        "confirmatory": True,
        "suite": suite,
        "discipline": {
            "suite": suite, "protocol": PROTOCOL, "layer": "primary",
            "arm_matrix_sha256": "m", "split_manifest_sha256": "s",
            "artifact_sha256": {"dsp_sv": "a"},
            "fit_record_sha256": {"sv": "f"}, "launch_run_ids": ["r"],
        },
        "point_estimates": {"dsp_sv": {"sr": 0.7}},
        "gate_schema": GATE_SCHEMA,
        "gate1": {"pass": g1, "d_sr_p5": 0.01, "d_c_p95": -0.09,
                  "d_sr_lower_quantile": 0.05, "d_c_upper_quantile": 0.95,
                  "cost_gate": -0.05},
    }
    if g1:
        v["gate2"] = {
            "pass": g2, "dsr_lower": 0.0, "dc_upper": -0.12,
            "dsr_lower_quantile": 0.05, "dc_upper_quantile": 0.95,
            "cost_gate": -0.05,
        }
    v.update(extra)
    return v


def test_all_four_gates_pass_confirms_cross_suite():
    out = finalize({s: _verdict(s) for s in REQUIRED_SUITES})
    assert out["cross_suite_confirmed"]
    assert out["verdict"] == "cross_suite_confirmed"
    assert out["n_gates"] == 4 and out["failed_gates"] == []


@pytest.mark.parametrize("suite", sorted(REQUIRED_SUITES))
@pytest.mark.parametrize("gate", ["gate1", "gate2"])
def test_any_single_gate_failure_blocks_the_general_claim(suite, gate):
    vs = {s: _verdict(s) for s in REQUIRED_SUITES}
    if gate == "gate1":
        vs[suite] = _verdict(suite, g1=False)
    else:
        vs[suite][gate]["pass"] = False
    out = finalize(vs)
    assert not out["cross_suite_confirmed"]
    assert out["verdict"] == "suite_specific_only"
    assert out["failed_gates"] == [f"{suite}.{gate}"]


def test_one_suite_alone_is_refused_not_downgraded():
    with pytest.raises(SystemExit, match="needs exactly"):
        finalize({"libero_10": _verdict("libero_10")})


def test_the_same_suite_twice_cannot_stand_in_for_both():
    with pytest.raises(SystemExit, match="needs exactly"):
        finalize({"libero_10": _verdict("libero_10")})


def test_an_unknown_suite_is_refused():
    with pytest.raises(SystemExit, match="not in"):
        check_one(_verdict("libero_object"), "v.json")


@pytest.mark.parametrize("field", ["discipline", "point_estimates", "suite"])
def test_incomplete_provenance_is_refused(field):
    v = _verdict("libero_10")
    v.pop(field)
    with pytest.raises(SystemExit, match="missing required field|suite"):
        check_one(v, "v.json")


@pytest.mark.parametrize("gate", ["gate1", "gate2"])
def test_a_required_gate_block_is_refused_when_its_predecessor_passed(gate):
    v = _verdict("libero_10")
    v.pop(gate)
    with pytest.raises(SystemExit, match=gate):
        check_one(v, "v.json")


@pytest.mark.parametrize("marker", ["gate2_compute_slack", "dc_upper_slack"])
def test_a_retired_gate_schema_is_refused(marker):
    with pytest.raises(SystemExit, match="retired Gate 2"):
        check_one(_verdict("libero_10", **{marker: 0.05}), "v.json")


def test_a_non_negative_cost_gate_is_the_retired_rule_and_is_refused():
    v = _verdict("libero_10")
    v["gate2"]["cost_gate"] = 0.05
    with pytest.raises(SystemExit, match="schema"):
        check_one(v, "v.json")


def test_a_missing_cost_gate_is_refused():
    v = _verdict("libero_10")
    v["gate2"].pop("cost_gate")
    with pytest.raises(SystemExit, match="schema"):
        check_one(v, "v.json")


def test_gate1_failure_with_gate2_absent_is_a_valid_fixed_sequence_result():
    v = _verdict("libero_10", g1=False)
    assert check_one(v, "v.json") == "libero_10"
    out = finalize({"libero_10": v, "libero_spatial": _verdict("libero_spatial")})
    assert not out["cross_suite_confirmed"]
    assert out["not_evaluated_fixed_sequence"] == ["libero_10.gate2"]
    assert out["n_gates"] == 3
