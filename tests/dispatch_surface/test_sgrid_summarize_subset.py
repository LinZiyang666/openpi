"""Batch-run support: summarize --arms subset + H-CRD knob parsing from arm names."""
from __future__ import annotations

import json
import types

import pytest

from exp.dispatch_surface import sgrid_sweep


def test_crd_params_roundtrip():
    assert sgrid_sweep.crd_params("dsp_sv_crd_q85_g1_m2_j3_L6") == {
        "budget_mult": 2, "j_bad": 3, "l_max": 6, "gamma": 1.0}
    assert sgrid_sweep.crd_params("dsp_sv_crd_q60_g1_minf_j3_L6")["budget_mult"] == float("inf")
    pure = sgrid_sweep.crd_params("dsp_sv_crd_q925_g1_m4_jinf_Lnone")
    assert pure["j_bad"] == float("inf") and pure["l_max"] is None
    assert sgrid_sweep.crd_params("dsp_sv_p85") is None          # plain surface arm


def test_summarize_rejects_unknown_subset_arm(tmp_path):
    matrix = tmp_path / "arm_matrix_sgrid.json"
    matrix.write_text(json.dumps({"protocol": sgrid_sweep.PROTOCOL_SGRID,
                                  "arms": {"dsp_sv_crd_q85_g1_m2_j3_L6": "x.yaml"}}))
    args = types.SimpleNamespace(arm_matrix=str(matrix), arms="dsp_sv_crd_q60_g1_m2_j3_L6",
                                 journal="", per_step="", launch_manifest="", split_manifest="",
                                 trials=30, out=str(tmp_path / "out.json"))
    with pytest.raises(SystemExit, match="unknown arms"):
        sgrid_sweep.summarize(args)
