"""R-CP2 config validation, YAML round trip and the CP2 artifact binding check."""

from __future__ import annotations

import copy
import pickle

import pytest
import torch
import yaml

from exp.actioncache_baseline import libs
from exp.actioncache_baseline.export_arms import assert_arm_yaml, cp2_arm_yaml
from openpi.cache.components.cp2_vlm_key_builder import get_projection_spec
from openpi.cache.config import ConfigValidationError, build_shared_storage, load_cache_config
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID

PROJ = libs.ProjectionArgs(seed=7, d=8, p=0.25, input_dim=64)


def _doc(tier="n0", theta=0.85, preload="/tmp/does-not-matter.pkl"):
    return cp2_arm_yaml(preload_path=preload, projection=PROJ, tier=tier, theta_raw=theta)


def _write(tmp_path, doc, name="arm.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return p


def test_cp2_yaml_round_trip_both_tiers(tmp_path):
    for tier in ("n0", "n1"):
        p = _write(tmp_path, _doc(tier=tier), f"{tier}.yaml")
        cfg = load_cache_config(p)
        assert sorted(cfg.checkpoints) == ["cp2"]
        assert cfg.key_builder.type == "cp2_vlm_ternary"
        assert cfg.key_builder.cp2_vlm.seed == 7 and cfg.key_builder.cp2_vlm.d == 8
        assert cfg.keys.vlm_out.enabled and not cfg.keys.robot_state.enabled
        assert cfg.checkpoints["cp2"].search_strategy.task_scoped is False
        assert_arm_yaml(p, tier=tier, theta_raw=0.85, projection=PROJ, preload_path="/tmp/does-not-matter.pkl")
    n1 = load_cache_config(_write(tmp_path, _doc(tier="n1"), "n1b.yaml")).checkpoints["cp2"].judge
    assert n1.threshold == 1.5 and n1.warm_tiers == [{"threshold": libs.theta_norm(0.85), "start_t": 0.1}]


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda d: d["checkpoints"].update({"cp1": copy.deepcopy(d["checkpoints"]["cp2"])}), "mutually exclusive"),
        (lambda d: d["checkpoints"].update({"cp3": copy.deepcopy(d["checkpoints"]["cp2"])}), "mutually exclusive"),
        (lambda d: d["key_builder"].update({"type": "cp1_mean_pool"}), "must be configured together"),
        (lambda d: d["keys"]["robot_state"].update({"enabled": True}), "produces exactly keys.vlm_out"),
        (lambda d: d["backend"].update({"vector_dims": {"vlm_out": 9}}), "vector_dims must be exactly"),
        (lambda d: d["checkpoints"]["cp2"]["gate"].update({"type": "score_hysteresis", "theta_low": 0.5, "theta_high": 0.5, "j": 1, "probe_interval": 1}), "always_search"),
        (lambda d: d["checkpoints"]["cp2"]["judge"].update({"type": "always_hit"}), "judge.type must be 'threshold'"),
        (lambda d: d["checkpoints"]["cp2"]["search_strategy"].update({"type": "weighted_rrf_knn"}), "weighted_score_sum_knn"),
        (lambda d: d.update({"write_policy": {"type": "always"}}), "write_policy.type='never'"),
        (lambda d: d["backend"]["in_memory"].update({"preload_path": ""}), "preload_path"),
        (lambda d: d["key_builder"]["cp2_vlm"].update({"p": 1.5}), "cp2_vlm.p"),
        (lambda d: d.update({"routing": {"hit_to": "127.0.0.1:1"}}), "routing"),
    ],
)
def test_r_cp2_rules_reject(tmp_path, mutate, fragment):
    doc = _doc()
    mutate(doc)
    with pytest.raises(ConfigValidationError) as exc:
        load_cache_config(_write(tmp_path, doc))
    assert fragment in str(exc.value)


def test_vlm_out_without_cp2_builder_is_rejected(tmp_path):
    doc = _doc()
    doc["checkpoints"] = {"cp1": copy.deepcopy(doc["checkpoints"]["cp2"])}
    doc["key_builder"] = {"type": "placeholder"}
    with pytest.raises(ConfigValidationError) as exc:
        load_cache_config(_write(tmp_path, doc))
    assert "keys.vlm_out is only produced by" in str(exc.value)


def _write_cp2_artifact(path, spec, *, id_policy=libs.ID_POLICY, projection=None, checkpoint=CheckpointID.CP2):
    entries = []
    for i in range(3):
        entries.append(CacheEntry(
            id=f"e{i}", checkpoint_id=checkpoint,
            query_keys={libs.FIELD: torch.randn(spec.d)},
            payload=CachePayload(action_chunk=torch.zeros(10, 32), intermediates={0.1: torch.zeros(10, 32)},
                                 denoising_num_steps=10, task_key="t"),
            step_idx=i, trajectory_id="traj", prev_ids=[], next_ids=[],
        ))
    art = {"key_builder_type": libs.KEY_BUILDER_TYPE, "checkpoint_id": "CP2",
           "vector_dims": {libs.FIELD: spec.d}, "entries": entries,
           "projection": spec.meta() if projection is None else projection, "id_policy": id_policy}
    with open(path, "wb") as f:
        pickle.dump(art, f)


def test_projection_binding_at_storage_assembly(tmp_path):
    spec = get_projection_spec(PROJ.seed, PROJ.d, PROJ.p, PROJ.input_dim)
    ok = tmp_path / "ok.pkl"
    _write_cp2_artifact(ok, spec)
    cfg = load_cache_config(_write(tmp_path, _doc(preload=str(ok)), "ok.yaml"))
    storage = build_shared_storage(cfg)
    assert storage.artifact_meta["projection"] == spec.meta()
    assert storage.artifact_meta["id_policy"] == libs.ID_POLICY

    # Different seed in the yaml than in the artifact -> digest mismatch.
    doc = _doc(preload=str(ok))
    doc["key_builder"]["cp2_vlm"]["seed"] = 8
    with pytest.raises(ConfigValidationError, match="projection"):
        build_shared_storage(load_cache_config(_write(tmp_path, doc, "bad_seed.yaml")))

    # Artifact without projection metadata (legacy) -> rejected.
    legacy = tmp_path / "legacy.pkl"
    _write_cp2_artifact(legacy, spec, projection={"seed": PROJ.seed})
    with pytest.raises(ConfigValidationError, match="projection"):
        build_shared_storage(load_cache_config(_write(tmp_path, _doc(preload=str(legacy)), "legacy.yaml")))

    # Wrong id policy -> rejected.
    bad_policy = tmp_path / "policy.pkl"
    _write_cp2_artifact(bad_policy, spec, id_policy="rehashed")
    with pytest.raises(ConfigValidationError, match="id_policy"):
        build_shared_storage(load_cache_config(_write(tmp_path, _doc(preload=str(bad_policy)), "policy.yaml")))
