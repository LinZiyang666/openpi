"""Tests for the Phase-2 weight-search experiment package.

Covers WeightSearchStrategy.plan() (pure-eval structure + held-out init
mapping), emit_yamls.build_eval_config (C2 write_policy:never + prompt_emb
masking + per_field score_normalization), and init_holdout (exclusion +
fail-fast).
"""

from __future__ import annotations

import json

import pytest

from openpi.conductor import task as T

from exp.weighted_sum.emit_yamls import (
    build_eval_config,
    grid_weight_configs,
    isolation_weight_configs,
)
from exp.weighted_sum.init_holdout import held_out_inits, load_used_inits
from exp.weighted_sum.weight_search_strategy import WeightSearchStrategy

S1 = T.ServerEndpoint("h", 8001)

_HOLDOUT = {0: [10, 11, 12, 13, 14], 1: [20, 21, 22, 23, 24]}


def _strategy(**kw):
    base = dict(
        task_ids=[0, 1],
        eval_trials=3,
        task_suite_name="libero_spatial",
        yaml_dir="/cfg",
        held_out_inits=_HOLDOUT,
    )
    base.update(kw)
    return WeightSearchStrategy(**base)


# ------------------------------------------------------------------
# plan()
# ------------------------------------------------------------------
def test_plan_pure_eval_no_warmup():
    g = _strategy().plan(["y"], {"y": S1})
    g.validate()
    assert set(g.stages) == {"y:eval"}
    assert g.stages["y:eval"].consumes_calib_id is None
    # 2 tasks x 3 eval trials = 6 eval episodes
    assert len(g.stages["y:eval"].episodes) == 6


def test_episodes_use_held_out_inits():
    g = _strategy().plan(["y"], {"y": S1})
    eps = g.stages["y:eval"].episodes
    for ep in eps:
        assert ep.orig_init_state_idx in _HOLDOUT[ep.task_id]
    # episode_idx ep maps to held_out[task][ep]
    by_task = {0: [], 1: []}
    for ep in eps:
        by_task[ep.task_id].append((ep.episode_idx, ep.orig_init_state_idx))
    for task_id, pairs in by_task.items():
        for ep_idx, init in pairs:
            assert init == _HOLDOUT[task_id][ep_idx]


def test_eval_trials_exceeds_holdout_raises():
    strat = _strategy(eval_trials=99)
    with pytest.raises(ValueError, match="held-out init"):
        strat.plan(["y"], {"y": S1})


# ------------------------------------------------------------------
# emit_yamls.build_eval_config
# ------------------------------------------------------------------
def _fields_calib():
    return {
        "vision_0": {
            "sim_type": "cosine",
            "selected": {"method": "logit", "params": {"lo": 0.9, "hi": 0.999, "eps": 1e-4}},
        },
        "robot_state": {
            "sim_type": "l2",
            "selected": {"method": "exp_l2", "params": {"tau": 0.33}},
        },
    }


def test_build_eval_config_c2_and_prompt_emb_masked():
    cfg = build_eval_config(
        builder_type="cp1_spatial_pool_16",
        vector_dims={"vision_0": 32768, "robot_state": 32},
        preload_path="x.pkl",
        weights={"vision_0": 0.5, "robot_state": 0.5},
        fields_calib=_fields_calib(),
    )
    # C2 write-frozen
    assert cfg["write_policy"]["type"] == "never"
    # prompt_emb dropped
    assert cfg["keys"]["prompt_emb"]["enabled"] is False
    # always_hit judge (isolates retrieval)
    assert cfg["checkpoints"]["cp1"]["judge"]["type"] == "always_hit"
    ss = cfg["checkpoints"]["cp1"]["search_strategy"]
    assert ss["type"] == "weighted_score_sum_knn"
    assert ss["score_normalization"]["type"] == "per_field"
    assert set(ss["score_normalization"]["fields"]) == {"vision_0", "robot_state"}
    # only weighted fields enabled + dimensioned
    assert set(cfg["backend"]["vector_dims"]) == {"vision_0", "robot_state"}
    assert cfg["keys"]["vision_0"] == {"enabled": True, "weight": 0.5}
    # l2 field carries tau in field_similarity
    assert ss["field_similarity"]["robot_state"]["to_similarity"]["tau"] == 0.33


def test_build_eval_config_rejects_uncalibrated_field():
    with pytest.raises(ValueError, match="no Phase-1 calibration"):
        build_eval_config(
            builder_type="cp1_max_pool",
            vector_dims={"vision_1": 2048},
            preload_path="x.pkl",
            weights={"vision_1": 1.0},
            fields_calib=_fields_calib(),  # vision_1 absent
        )


def test_emit_isolation_yaml_force_enables_required_and_validates(tmp_path):
    # Modality isolation weights only one field, but cp1_*/clip key_builders
    # require vision_0 + robot_state enabled; build_eval_config force-enables
    # them at weight 0 so the YAML loads + validates.
    import yaml

    from openpi.cache.config import load_cache_config, validate_cache_config

    cfg = build_eval_config(
        builder_type="cp1_spatial_pool_16",
        vector_dims={"vision_0": 32768, "vision_1": 32768, "robot_state": 32},
        preload_path="x.pkl",
        weights={"vision_1": 1.0},  # isolation: only vision_1 carries weight
        fields_calib={
            "vision_1": {
                "sim_type": "cosine",
                "selected": {"method": "affine_clip", "params": {"lo": 0.8, "hi": 0.99}},
            }
        },
    )
    assert cfg["keys"]["vision_0"] == {"enabled": True, "weight": 0.0}
    assert cfg["keys"]["robot_state"] == {"enabled": True, "weight": 0.0}
    assert cfg["keys"]["vision_1"] == {"enabled": True, "weight": 1.0}
    # required fields are dimensioned but carry NO score_normalization entry
    assert set(cfg["checkpoints"]["cp1"]["search_strategy"]["score_normalization"]["fields"]) == {
        "vision_1"
    }
    p = tmp_path / "iso.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    loaded = load_cache_config(str(p))
    validate_cache_config(loaded)  # must not raise


def test_weight_config_generators():
    iso = isolation_weight_configs(["vision_0", "robot_state"])
    assert iso == {"iso_vision_0": {"vision_0": 1.0}, "iso_robot_state": {"robot_state": 1.0}}
    grid = grid_weight_configs(["vision_0", "robot_state"], levels=(0.5,))
    # one 2-field combo at 0.5/0.5
    assert any(set(w) == {"vision_0", "robot_state"} for w in grid.values())


# ------------------------------------------------------------------
# init_holdout
# ------------------------------------------------------------------
def test_held_out_excludes_used(tmp_path):
    init_map = [
        {"task_id": 0, "orig_init_state_idx": 13},
        {"task_id": 0, "orig_init_state_idx": 0},
        {"task_id": 1, "orig_init_state_idx": 5},
    ]
    p = tmp_path / "init_map.json"
    p.write_text(json.dumps(init_map))
    used = load_used_inits(p)
    assert used == {0: {13, 0}, 1: {5}}
    holdout = held_out_inits(p, task_ids=[0, 1], total_inits_per_task=5)
    assert holdout[0] == [1, 2, 3, 4]  # 0 and 13(>5 ignored) excluded; 13 out of range
    assert holdout[1] == [0, 1, 2, 3, 4]
    assert 5 not in holdout[1]  # used init excluded


def test_load_used_inits_missing_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_used_inits(tmp_path / "does_not_exist.json")


# ------------------------------------------------------------------
# summarize (journal -> per-yaml success rate)
# ------------------------------------------------------------------
def test_summarize_journal_eval_done_only(tmp_path):
    from exp.weighted_sum.summarize import summarize_journal

    rows = [
        {"yaml_id": "y__iso_vision_0", "phase": "eval", "status": "done", "success": True},
        {"yaml_id": "y__iso_vision_0", "phase": "eval", "status": "done", "success": False},
        {"yaml_id": "y__iso_vision_0", "phase": "warmup", "status": "done", "success": True},
        {"yaml_id": "y__iso_robot_state", "phase": "eval", "status": "failed", "success": False},
        {"yaml_id": "y__iso_robot_state", "phase": "eval", "status": "done", "success": True},
    ]
    j = tmp_path / "journal.jsonl"
    j.write_text("\n".join(json.dumps(r) for r in rows))
    res = summarize_journal(j)
    assert res["y__iso_vision_0"]["success_rate"] == 0.5  # warmup row ignored
    assert res["y__iso_vision_0"]["n"] == 2
    assert res["y__iso_robot_state"]["success_rate"] == 1.0  # failed row ignored
    assert res["y__iso_robot_state"]["n"] == 1
