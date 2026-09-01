"""Config-level gate binding for CRD artifacts (H-CRD G2 R1 B2).

A cumulative-risk artifact must only ever be loaded under ``gate.type ==
always_search``; every gate that can skip the search on CP1 is refused at
``load_cache_config`` time, and again by the orchestrator at assembly.
"""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from openpi.cache.config import ConfigValidationError, compute_surface_retrieval_contract, load_cache_config
from tests.cache.test_crd_judge import write_crd
from tests.cache.test_surface_binding import _base_yaml_dict, _entry, _write_pkl


def _crd_yaml(tmp_path, gate: dict):
    pkl = _write_pkl(tmp_path, [_entry("e0")])
    probe = _base_yaml_dict(tmp_path, tmp_path / "crd.npz", pkl)
    probe["checkpoints"]["cp1"]["judge"] = {"type": "always_hit"}
    p = tmp_path / "probe.yaml"
    p.write_text(yaml.safe_dump(probe, sort_keys=False))
    contract = compute_surface_retrieval_contract(load_cache_config(str(p)))
    contract.update({"library_sha256": "SHA", "library_entry_count": 3, "action_dim": 4,
                     "num_steps": 10, "h_exec": 5, "policy_fingerprint": "fp"})
    write_crd(tmp_path, contract=contract, w=np.ones(4, np.float32), k=3, name="crd.npz")
    doc = _base_yaml_dict(tmp_path, tmp_path / "crd.npz", pkl)
    doc["checkpoints"]["cp1"]["gate"] = gate
    path = tmp_path / "crd.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return str(path)


def test_crd_artifact_loads_under_always_search(tmp_path):
    cfg = load_cache_config(_crd_yaml(tmp_path, {"type": "always_search"}))
    assert cfg.checkpoints["cp1"].judge.type == "dispatch_surface"


def test_config_load_rejects_declared_crd_with_missing_controller_array(tmp_path):
    yaml_path = _crd_yaml(tmp_path, {"type": "always_search"})
    doc = yaml.safe_load(open(yaml_path))
    artifact_path = doc["checkpoints"]["cp1"]["judge"]["surface_artifact_path"]
    with np.load(artifact_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "q_hat_central"}
    np.savez(artifact_path, **arrays)
    with pytest.raises(ConfigValidationError, match="lacks q_hat"):
        load_cache_config(yaml_path)


@pytest.mark.parametrize("gate", [
    {"type": "always_skip"},
    {"type": "score_hysteresis", "theta_low": 0.9, "theta_high": 0.95, "j": 3, "probe_interval": 3},
    {"type": "client_controlled"},
])
def test_crd_artifact_refuses_skipping_gates(tmp_path, gate):
    with pytest.raises(ConfigValidationError, match="always_search"):
        load_cache_config(_crd_yaml(tmp_path, gate))
