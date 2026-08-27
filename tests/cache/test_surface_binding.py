"""Integration contract tests for the dispatch-surface binding chain.

Covers: the identity extension of ``InMemoryBackend.load_artifact`` (records,
never asserts — legacy artifacts keep loading), the surface-conditional
library binding at assembly (``_check_surface_library_binding``), yaml-level
validation with the full contract-drift matrix, and the effective-top-k lift
through the real assembly path.
"""

from __future__ import annotations

import dataclasses
import pickle

import numpy as np
import pytest
import torch
import yaml

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.config import (
    ConfigValidationError,
    _check_surface_library_binding,
    compute_surface_retrieval_contract,
    load_cache_config,
)
from openpi.cache.components.surface_judge import (
    SURFACE_ARTIFACT_SCHEMA_VERSION,
    SurfaceArtifact,
    save_surface_artifact,
)
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID

DIMS = {"robot_state": 4}
NINE_TIERS = tuple(round(1.0 - i / 10, 4) for i in range(1, 10))


def _entry(eid: str, *, with_intermediates=True, horizon=6, dim=4, num_steps=10):
    payload = CachePayload(
        action_chunk=torch.zeros(horizon, dim, dtype=torch.float32),
        intermediates=(
            {t: torch.zeros(horizon, dim) for t in NINE_TIERS}
            if with_intermediates else None
        ),
        denoising_num_steps=num_steps if with_intermediates else None,
    )
    return CacheEntry(
        id=eid, checkpoint_id=CheckpointID.CP1,
        query_keys={"robot_state": torch.zeros(4, dtype=torch.float32)},
        payload=payload, trajectory_id="traj", step_idx=0,
    )


def _write_pkl(tmp_path, entries, name="lib.pkl"):
    art = {
        "key_builder_type": "placeholder", "checkpoint_id": "CP1",
        "vector_dims": DIMS, "entries": entries,
    }
    p = tmp_path / name
    with open(p, "wb") as f:
        pickle.dump(art, f)
    return p


def _loaded_backend(tmp_path, entries):
    p = _write_pkl(tmp_path, entries)
    backend = InMemoryBackend(DIMS)
    backend.load_artifact(str(p))
    return backend, p


# ------------------------------------------------------------------
# Identity seam
# ------------------------------------------------------------------


def test_identity_meta_aggregated(tmp_path):
    backend, _ = _loaded_backend(tmp_path, [_entry(f"e{i}") for i in range(3)])
    meta = backend.artifact_meta
    assert meta["entry_count"] == 3
    assert meta["action_horizon"] == 6 and meta["action_dim"] == 4
    assert meta["denoising_num_steps"] == 10
    assert meta["schema_consensus_count"] == 3
    assert meta["intermediates_completeness"]["0.3000"] == 1.0
    assert len(meta["library_sha256"]) == 64


def test_legacy_artifact_without_intermediates_still_loads(tmp_path):
    backend, _ = _loaded_backend(
        tmp_path, [_entry(f"e{i}", with_intermediates=False) for i in range(2)]
    )
    assert backend.artifact_meta["entry_count"] == 2
    assert backend.artifact_meta["intermediates_completeness"] == {}


def test_mixed_schema_artifact_loads_but_records_dissent(tmp_path):
    entries = [_entry("e0"), _entry("e1"), _entry("e2", horizon=8)]
    backend, _ = _loaded_backend(tmp_path, entries)
    meta = backend.artifact_meta
    assert meta["entry_count"] == 3
    assert meta["schema_consensus_count"] == 2  # recorded, not rejected


# ------------------------------------------------------------------
# Surface-conditional binding (assembly point)
# ------------------------------------------------------------------


class _FakeStorage:
    def __init__(self, meta):
        self.artifact_meta = meta


class _FakeJudge:
    def __init__(self, artifact):
        self.artifact = artifact


def _artifact(tmp_path, contract_overrides=None, **art_overrides):
    contract = {
        "key_builder_digest": "kb", "search_digest": "sd", "top_k": 3,
        "library_sha256": "SHA", "library_entry_count": 3,
        "action_dim": 4, "num_steps": 10, "h_exec": 5, "policy_fingerprint": "fp",
    }
    contract.update(contract_overrides or {})
    kwargs = dict(
        schema_version=SURFACE_ARTIFACT_SCHEMA_VERSION, k=3, h_exec=5,
        w=np.ones(4, dtype=np.float32), active_mask=np.ones(4, dtype=bool),
        start_t_ws=0.3, delta=0.5, alpha=0.05, uses_disagreement=True,
        v_bin_edges=np.array([0.0, 1.0]), s_min_full=np.array([0.9]),
        s_min_warm=np.array([0.8]), conformal_c=0.01, n_calibration_episodes=100,
        retrieval_contract=contract, meta={},
    )
    kwargs.update(art_overrides)
    art = SurfaceArtifact(**kwargs)
    path = tmp_path / "s.npz"
    save_surface_artifact(art, str(path))
    return art


def _good_meta():
    return {
        "library_sha256": "SHA", "entry_count": 3, "action_dim": 4,
        "action_horizon": 6, "denoising_num_steps": 10,
        "schema_consensus_count": 3,
        "intermediates_completeness": {"0.3000": 1.0},
    }


def test_binding_passes_on_matching_library(tmp_path):
    art = _artifact(tmp_path)
    _check_surface_library_binding(
        "cp1", _FakeJudge(art), _FakeStorage(_good_meta()), effective_top_k=5,
    )


@pytest.mark.parametrize("mutation, top_k", [
    ({"library_sha256": "OTHER"}, 5),
    ({"entry_count": 99}, 5),
    ({"action_dim": 32}, 5),
    ({"denoising_num_steps": 4}, 5),
    ({"schema_consensus_count": 2}, 5),
    ({"action_horizon": 3}, 5),                       # h_exec 5 > horizon 3
    ({"intermediates_completeness": {"0.3000": 0.9}}, 5),
    ({}, 2),                                           # width below artifact k
])
def test_binding_rejects(tmp_path, mutation, top_k):
    art = _artifact(tmp_path)
    meta = {**_good_meta(), **mutation}
    with pytest.raises(ConfigValidationError):
        _check_surface_library_binding(
            "cp1", _FakeJudge(art), _FakeStorage(meta), effective_top_k=top_k,
        )


# ------------------------------------------------------------------
# Yaml-level validation + contract drift
# ------------------------------------------------------------------


def _base_yaml_dict(tmp_path, artifact_path, preload):
    return {
        "enabled": True,
        "keys": {
            "vision_0": {"enabled": False, "weight": 0.0},
            "vision_1": {"enabled": False, "weight": 0.0},
            "vision_2": {"enabled": False, "weight": 0.0},
            "prompt_emb": {"enabled": False, "weight": 0.0},
            "robot_state": {"enabled": True, "weight": 1.0},
        },
        "key_builder": {"type": "placeholder"},
        "checkpoints": {
            "cp1": {
                "enabled": True,
                "gate": {"type": "always_search"},
                "judge": {"type": "dispatch_surface",
                          "surface_artifact_path": str(artifact_path)},
                "search_strategy": {"type": "weighted_rrf_knn", "top_k": 3},
            },
        },
        "backend": {"type": "in_memory", "vector_dims": DIMS,
                    "in_memory": {"preload_path": str(preload)}},
        "write_policy": {"type": "never"},
    }


def _load(tmp_path, doc, name="cfg.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    return load_cache_config(str(p))


def _matched_artifact(tmp_path, doc):
    """Artifact whose contract digests match the given yaml doc."""
    p = tmp_path / "probe.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    config = load_cache_config(str(p))
    contract = compute_surface_retrieval_contract(config)
    contract.update({
        "library_sha256": "SHA", "library_entry_count": 3, "action_dim": 4,
        "num_steps": 10, "h_exec": 5, "policy_fingerprint": "fp",
    })
    return _artifact(tmp_path, contract_overrides=contract)


def test_yaml_valid_surface_config_loads(tmp_path):
    pkl = _write_pkl(tmp_path, [_entry("e0")])
    doc = _base_yaml_dict(tmp_path, tmp_path / "s.npz", pkl)
    doc["checkpoints"]["cp1"]["judge"]["type"] = "always_hit"
    doc["checkpoints"]["cp1"]["judge"].pop("surface_artifact_path")
    _matched_artifact(tmp_path, doc)  # writes s.npz with digests for this doc
    doc["checkpoints"]["cp1"]["judge"] = {
        "type": "dispatch_surface",
        "surface_artifact_path": str(tmp_path / "s.npz"),
    }
    cfg = _load(tmp_path, doc)
    assert cfg.checkpoints["cp1"].judge.type == "dispatch_surface"


@pytest.mark.parametrize("mutate", [
    lambda d: d["checkpoints"]["cp1"]["judge"].update(warm_tiers=[{"threshold": 0.9, "start_t": 0.3}]),
    lambda d: d["checkpoints"]["cp1"]["judge"].update(start_t=0.3),
    lambda d: d["checkpoints"]["cp1"]["judge"].pop("surface_artifact_path"),
    lambda d: d.update(write_policy={"type": "always"}),
    lambda d: d["backend"]["in_memory"].update(preload_path=None),
])
def test_yaml_rejects_invalid_surface_configs(tmp_path, mutate):
    pkl = _write_pkl(tmp_path, [_entry("e0")])
    probe = _base_yaml_dict(tmp_path, tmp_path / "s.npz", pkl)
    probe["checkpoints"]["cp1"]["judge"] = {"type": "always_hit"}
    _matched_artifact(tmp_path, probe)
    doc = _base_yaml_dict(tmp_path, tmp_path / "s.npz", pkl)
    mutate(doc)
    with pytest.raises(ConfigValidationError):
        _load(tmp_path, doc)


@pytest.mark.parametrize("drift", [
    lambda d: d["keys"]["robot_state"].update(weight=0.7),
    lambda d: d["key_builder"].update(type="cp1_mean_pool"),
    lambda d: d["checkpoints"]["cp1"]["search_strategy"].update(step_filter="window"),
    lambda d: d["checkpoints"]["cp1"]["search_strategy"].update(trajectory_depth=3),
    lambda d: d["checkpoints"]["cp1"]["search_strategy"].update(
        field_similarity={"robot_state": {"type": "l2"}}),
])
def test_yaml_rejects_every_score_affecting_drift(tmp_path, drift):
    pkl = _write_pkl(tmp_path, [_entry("e0")])
    probe = _base_yaml_dict(tmp_path, tmp_path / "s.npz", pkl)
    probe["checkpoints"]["cp1"]["judge"] = {"type": "always_hit"}
    _matched_artifact(tmp_path, probe)  # contract digests frozen for the base doc
    doc = _base_yaml_dict(tmp_path, tmp_path / "s.npz", pkl)
    drift(doc)
    with pytest.raises(ConfigValidationError):
        _load(tmp_path, doc)


def test_contract_digest_is_stable_under_field_order():
    from openpi.cache.config import CacheConfig

    c1, c2 = CacheConfig(), CacheConfig()
    d1 = compute_surface_retrieval_contract(c1)
    d2 = compute_surface_retrieval_contract(dataclasses.replace(c2))
    assert d1 == d2
