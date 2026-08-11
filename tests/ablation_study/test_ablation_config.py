"""Routing allowlist coverage over the real emitted arm yamls (plan §8.2/§9)."""

from __future__ import annotations

import copy
import pathlib

import pytest
import yaml

from openpi.cache.config import ConfigValidationError
from openpi.cache.config import load_cache_config

CONFIG_ROOT = pathlib.Path("exp/ablation_study/config")
VALID_ROUTED = CONFIG_ROOT / "small_at_hit" / "libero_spatial_hit_smolvla.yaml"


def _load_mutated(tmp_path, mutate) -> None:
    raw = yaml.safe_load(VALID_ROUTED.read_text())
    mutate(raw)
    p = tmp_path / "mutated.yaml"
    p.write_text(yaml.safe_dump(raw))
    load_cache_config(p)


def test_all_emitted_yamls_load():
    for p in sorted(CONFIG_ROOT.rglob("*.yaml")):
        if p.name.startswith("arm_matrix"):
            continue
        load_cache_config(p)


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("both_targets", lambda r: r["routing"].update(miss_to="127.0.0.1:7002")),
        ("no_target", lambda r: r["routing"].update(hit_to=None)),
        ("bad_endpoint", lambda r: r["routing"].update(hit_to="not-an-endpoint")),
        ("warm_tiers", lambda r: r["checkpoints"]["cp1"]["judge"].update(
            warm_tiers=[{"threshold": 0.9, "start_t": 0.5}])),
        ("cp3_present", lambda r: r["checkpoints"].update(
            cp3=copy.deepcopy(r["checkpoints"]["cp1"]))),
        ("depth_gt_1", lambda r: r["checkpoints"]["cp1"]["search_strategy"].update(
            trajectory_depth=3, trajectory_weights=[0.5, 0.3, 0.2])),
        ("write_always", lambda r: r.update(write_policy={"type": "always"})),
        ("collection_on", lambda r: r.update(collection={"export_collect_meta": True})),
        ("bad_gate", lambda r: r["checkpoints"]["cp1"]["gate"].update(type="score_hysteresis",
            theta_low=0.5, theta_high=0.9, j=3)),
        ("bad_judge", lambda r: r["checkpoints"]["cp1"]["judge"].update(type="always_warm_start",
            start_t=0.5)),
        ("bad_timeout", lambda r: r["routing"].update(connect_timeout_s=0)),
    ],
)
def test_allowlist_rejects(tmp_path, name, mutate):
    with pytest.raises(ConfigValidationError):
        _load_mutated(tmp_path, mutate)


def test_routing_absent_stays_inert(tmp_path):
    raw = yaml.safe_load(VALID_ROUTED.read_text())
    del raw["routing"]
    # Without routing the allowlist must NOT apply: warm_tiers become legal again.
    raw["checkpoints"]["cp1"]["judge"]["warm_tiers"] = [{"threshold": 0.9, "start_t": 0.5}]
    p = tmp_path / "inert.yaml"
    p.write_text(yaml.safe_dump(raw))
    cfg = load_cache_config(p)
    assert cfg.routing is None
