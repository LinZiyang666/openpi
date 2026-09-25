"""``warm_reset:`` yaml block: parsing, round trip and every validation rule (plan §4.2, §9.3, §9.4)."""

from __future__ import annotations

import dataclasses
import os

import pytest
import yaml

from openpi.cache.config import (
    CacheConfig,
    CheckpointConfig,
    ConfigValidationError,
    GateConfig,
    JudgeConfig,
    RoutingConfig,
    SearchStrategyConfig,
    WarmResetConfig,
    WarmResetGridConfig,
    WarmResetSelfSeedConfig,
    WarmResetStartConfig,
    _warm_reset_errors,
    load_cache_config,
    validate_cache_config,
)
from openpi.cache.warm_reset.types import WarmResetSpec
from tests.cache.warm_reset._support import pi05_config

CACHE_SNAPSHOT = {
    "start": {"source": "cache", "point": "snapshot"},
    "grid": {"kind": "reset", "entry_t": 1.0},
    "num_steps": "remaining",
}
SELF_FINAL = {
    "start": {"source": "self", "point": "final"},
    "grid": {"kind": "reset", "entry_t": 0.9},
    "num_steps": "remaining",
    "self_seed": {"namespace": "sdiag_libero_self_prod"},
}


def _block(tmp_path, base: dict, **overrides) -> dict:
    block = {**base, "evidence_dir": str(tmp_path / "evidence"), **overrides}
    return block


def _errors(cfg, *, check_files: bool = True) -> str:
    """The warm_reset rule messages alone (other validators cannot mask a miss)."""
    messages = _warm_reset_errors(cfg, check_files)
    assert messages, "expected a warm_reset validation error"
    return "\n".join(messages)


def test_rules_are_part_of_validate_cache_config(tmp_path):
    cfg = pi05_config(_block(tmp_path, CACHE_SNAPSHOT, num_steps=0))
    with pytest.raises(ConfigValidationError, match="warm_reset.num_steps"):
        validate_cache_config(cfg)


# ------------------------------------------------------------------
# Absent block / parsing
# ------------------------------------------------------------------


def test_absent_block_is_none_and_validates(tmp_path):
    cfg = pi05_config(None)
    assert cfg.warm_reset is None
    validate_cache_config(cfg)
    assert CacheConfig().warm_reset is None


def test_yaml_round_trip_builds_the_dataclass_tree(tmp_path):
    raw = {
        "checkpoints": {"cp1": {
            "judge": {"type": "always_warm_start", "start_t": 0.2},
            "search_strategy": {"type": "weighted_rrf_knn"},
        }},
        "keys": {"robot_state": {"enabled": True, "weight": 1.0}},
        "backend": {"type": "in_memory", "vector_dims": {"robot_state": 8}},
        "write_policy": {"type": "never"},
        "warm_reset": _block(tmp_path, SELF_FINAL),
    }
    path = tmp_path / "selfresetfinal_t0.2.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_cache_config(path)
    wr = cfg.warm_reset
    assert isinstance(wr, WarmResetConfig)
    assert wr.start == WarmResetStartConfig(source="self", point="final")
    assert wr.grid == WarmResetGridConfig(kind="reset", entry_t=0.9)
    assert wr.self_seed == WarmResetSelfSeedConfig(
        namespace="sdiag_libero_self_prod",
        identity_keys=["experiment", "task", "orig_init_state_idx", "attempt"],
    )
    spec = WarmResetSpec.from_config(wr)
    assert spec == WarmResetSpec(
        source="self", point="final", kind="reset", level=0.9, num_steps=None,
        seed_namespace="sdiag_libero_self_prod",
        seed_keys=("experiment", "task", "orig_init_state_idx", "attempt"),
        evidence_dir=str(tmp_path / "evidence"),
    )
    # The digest is stable across loads and moves with any field.
    assert WarmResetSpec.from_config(load_cache_config(path).warm_reset).digest() == spec.digest()
    assert dataclasses.replace(spec, level=0.5).digest() != spec.digest()


def test_explicit_steps_and_shoot_block(tmp_path):
    cfg = pi05_config(
        _block(tmp_path, {"start": {"source": "cache", "point": "snapshot"},
                          "grid": {"kind": "shoot", "step_budget": 0.75}, "num_steps": 2})
    )
    validate_cache_config(cfg)
    spec = WarmResetSpec.from_config(cfg.warm_reset)
    assert (spec.kind, spec.level, spec.num_steps) == ("shoot", 0.75, 2)


# ------------------------------------------------------------------
# Rule 1 / 2: dead config and refused combinations
# ------------------------------------------------------------------


def test_block_without_a_warm_start_judge_is_dead(tmp_path):
    cfg = pi05_config(_block(tmp_path, CACHE_SNAPSHOT))
    cfg.checkpoints["cp1"].judge = JudgeConfig(type="always_hit")
    assert "never run" in _errors(cfg)


@pytest.mark.parametrize("combo,needle", [
    ("trace", "warm_reset and trace.enabled"),
    ("shadow", "warm_reset and shadow_teacher"),
    ("routing", "incompatible with a routing section"),
    ("online_rit", "incompatible with the online_rit judge"),
    ("cp2", "incompatible with an enabled cp2"),
])
def test_refused_combinations(tmp_path, combo, needle):
    cfg = pi05_config(_block(tmp_path, CACHE_SNAPSHOT))
    if combo == "trace":
        cfg.trace.enabled, cfg.trace.out_dir = True, str(tmp_path / "trace")
        cfg.write_policy.type = "never"
    elif combo == "shadow":
        cfg.shadow_teacher.enabled, cfg.shadow_teacher.path = True, str(tmp_path / "x.jsonl")
    elif combo == "routing":
        cfg.routing = RoutingConfig(miss_to="127.0.0.1:1")
    elif combo == "online_rit":
        cfg.checkpoints["cp1"].judge = JudgeConfig(type="online_rit", tiers=[0.5])
    else:
        cfg.checkpoints["cp2"] = CheckpointConfig(
            gate=GateConfig(type="always_search"),
            judge=JudgeConfig(type="threshold", threshold=0.9),
            search_strategy=SearchStrategyConfig(type="weighted_score_sum_knn"),
        )
    assert needle in _errors(cfg, check_files=False)


# ------------------------------------------------------------------
# Rule 3-6: start / grid / num_steps / self_seed / evidence_dir
# ------------------------------------------------------------------


@pytest.mark.parametrize("override,needle", [
    ({"start": {"source": "library", "point": "snapshot"}}, "warm_reset.start.source"),
    ({"start": {"source": "cache", "point": "middle"}}, "warm_reset.start.point"),
    ({"start": None}, "warm_reset.start is required"),
    ({"grid": {"kind": "exact"}}, "warm_reset.grid.kind"),
    ({"grid": None}, "warm_reset.grid is required"),
    ({"grid": {"kind": "reset"}}, "warm_reset.grid.entry_t"),
    ({"grid": {"kind": "reset", "entry_t": 0.0}}, "warm_reset.grid.entry_t"),
    ({"grid": {"kind": "reset", "entry_t": 1.5}}, "warm_reset.grid.entry_t"),
    ({"grid": {"kind": "reset", "entry_t": True}}, "warm_reset.grid.entry_t"),
    ({"grid": {"kind": "reset", "entry_t": 1.0, "step_budget": 0.5}}, "only valid with kind: shoot"),
    ({"grid": {"kind": "shoot"}}, "warm_reset.grid.step_budget"),
    ({"grid": {"kind": "shoot", "step_budget": 0.5, "entry_t": 1.0}}, "only valid with kind: reset"),
    ({"num_steps": 11}, "1 <= N <= K"),
    ({"num_steps": 0}, "1 <= N <= K"),
    ({"num_steps": True}, "warm_reset.num_steps"),
    ({"num_steps": "all"}, "warm_reset.num_steps"),
    ({"self_seed": {"namespace": "ns"}}, "only valid with start.source: self"),
    ({"evidence_dir": ""}, "evidence_dir is required"),
])
def test_cache_block_rules(tmp_path, override, needle):
    block = _block(tmp_path, CACHE_SNAPSHOT)
    block.update(override)
    cfg = pi05_config(block)
    if block.get("start") is None:
        cfg.warm_reset.start = None
    if block.get("grid") is None:
        cfg.warm_reset.grid = None
    assert needle in _errors(cfg)


def test_final_start_cannot_shoot(tmp_path):
    block = _block(tmp_path, CACHE_SNAPSHOT, start={"source": "cache", "point": "final"},
                   grid={"kind": "shoot", "step_budget": 1.0})
    assert "cannot shoot" in _errors(pi05_config(block))


@pytest.mark.parametrize("seed,needle", [
    (None, "self_seed is required"),
    ({"namespace": ""}, "self_seed.namespace"),
    ({"namespace": "ns", "identity_keys": []}, "self_seed.identity_keys"),
    ({"namespace": "ns", "identity_keys": ["experiment", "task_uid"]}, "task_uid"),
])
def test_self_seed_rules(tmp_path, seed, needle):
    block = _block(tmp_path, SELF_FINAL)
    block["self_seed"] = seed
    assert needle in _errors(pi05_config(block))


def test_self_block_validates(tmp_path):
    validate_cache_config(pi05_config(_block(tmp_path, SELF_FINAL)))
    assert _warm_reset_errors(pi05_config(_block(tmp_path, CACHE_SNAPSHOT)), True) == []


def test_num_steps_bound_follows_the_named_schedule(tmp_path):
    groot = pi05_config(_block(tmp_path, CACHE_SNAPSHOT, num_steps=8), start_t=0.75)
    groot.denoise_schedule = "groot_n15_k8_v1"
    validate_cache_config(groot, check_files=False)
    groot.warm_reset.num_steps = 9
    assert "1 <= N <= K (8)" in _errors(groot, check_files=False)


def test_evidence_dir_must_be_writable_at_load(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    os.chmod(locked, 0o500)
    try:
        if os.access(locked, os.W_OK):
            pytest.skip("running as a user that ignores directory permissions")
        cfg = pi05_config(_block(tmp_path, CACHE_SNAPSHOT, evidence_dir=str(locked / "run" / "a")))
        assert "not a writable directory" in _errors(cfg)
        # check_files=False skips only the filesystem probe.
        validate_cache_config(cfg, check_files=False)
    finally:
        os.chmod(locked, 0o700)
    # A not-yet-existing directory under a writable ancestor is fine.
    validate_cache_config(pi05_config(_block(tmp_path, CACHE_SNAPSHOT, evidence_dir=str(tmp_path / "a" / "b"))))
