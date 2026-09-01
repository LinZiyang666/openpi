"""--gate-layer secondary: emit re-attaches the production hysteresis gate (goal-3 chain)."""
from __future__ import annotations

import json
import types

import pytest
import yaml as yamllib

from exp.dispatch_surface import sgrid_sweep
from tests.dispatch_surface.test_crd_export import world  # noqa: F401  (module-scoped fixture)
from tests.dispatch_surface.test_rev2_phase0 import TEMPLATE, run_exporter


def _record(world, name):  # noqa: F811
    out = world.tmp / name
    if not (out / "export_record.json").is_file():
        run_exporter(world.manifest_path, world.world, out, "artifact.dsp_sv", "0.85")
    return str(out / "export_record.json")


@pytest.fixture(scope="module")
def gated(world):  # noqa: F811
    out_dir = world.tmp / "sysgate_cfg"
    args = types.SimpleNamespace(rev1_package_manifest=str(world.manifest_path),
                                 export_records=_record(world, "sysgate_ex"),
                                 template=str(TEMPLATE), library_pkl=str(world.world.lib),
                                 out_dir=str(out_dir), gate_layer="secondary")
    sgrid_sweep.emit(args)
    return json.loads((out_dir / "arm_matrix_sgrid.json").read_text())


def test_gated_matrix_declares_the_sysgate_protocol(gated):
    assert gated["protocol"] == sgrid_sweep.PROTOCOL_SYSGATE
    assert gated["gate_type"] == "score_hysteresis"
    assert gated["gate_params"] == {"j": 3, "probe_interval": 3, "L": 6}


def test_gated_yaml_carries_the_frozen_production_gate(gated):
    doc = yamllib.safe_load(open(next(iter(gated["arms"].values()))))
    gate = doc["checkpoints"]["cp1"]["gate"]
    assert gate["type"] == "score_hysteresis"
    assert gate["theta_low"] == gate["theta_high"] == pytest.approx(gated["gate_theta"])
    assert (gate["j"], gate["probe_interval"], gate["L"]) == (3, 3, 6)


def test_default_emit_is_unchanged(world):  # noqa: F811
    out_dir = world.tmp / "plain_cfg"
    args = types.SimpleNamespace(rev1_package_manifest=str(world.manifest_path),
                                 export_records=_record(world, "sysgate_ex"),
                                 template=str(TEMPLATE), library_pkl=str(world.world.lib),
                                 out_dir=str(out_dir), gate_layer="primary")
    sgrid_sweep.emit(args)
    m = json.loads((out_dir / "arm_matrix_sgrid.json").read_text())
    assert m["protocol"] == sgrid_sweep.PROTOCOL_SGRID and m["gate_type"] == "always_search"
