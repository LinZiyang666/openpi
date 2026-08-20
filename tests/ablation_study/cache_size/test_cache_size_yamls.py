"""Arm-yaml derivation for the cache-size ablation (plan §7, §8.3, §10)."""

from __future__ import annotations

import copy

import pytest
import yaml

from exp.ablation_study.cache_size.emit_size_yamls import (
    SENSITIVITY_TIERS,
    TIERS,
    arm_name,
    make_main_arm,
    make_sensitivity_arm,
)

BASELINE_PATH = "exp/ablation_study/config/common/libero_spatial_baseline.yaml"


@pytest.fixture()
def baseline():
    with open(BASELINE_PATH) as f:
        return yaml.safe_load(f)


def _norm_fields(cfg):
    return cfg["checkpoints"]["cp1"]["search_strategy"]["score_normalization"]["fields"]


def test_main_arm_changes_only_verdict_and_preload(baseline):
    arm = make_main_arm(baseline, "/tmp/lib_S3.pkl")

    assert arm["checkpoints"]["cp1"]["judge"] == {"type": "always_hit"}
    assert arm["checkpoints"]["cp1"]["gate"] == {"type": "always_search"}
    assert arm["backend"]["in_memory"]["preload_path"] == "/tmp/lib_S3.pkl"

    # Everything else must survive byte-for-byte: normalize the two intended
    # deltas back and compare whole documents.
    restored = copy.deepcopy(arm)
    restored["checkpoints"]["cp1"]["judge"] = baseline["checkpoints"]["cp1"]["judge"]
    restored["checkpoints"]["cp1"]["gate"] = baseline["checkpoints"]["cp1"]["gate"]
    restored["backend"]["in_memory"]["preload_path"] = \
        baseline["backend"]["in_memory"]["preload_path"]
    assert restored == baseline


def test_main_arm_keeps_retrieval_chain_fixed(baseline):
    arm = make_main_arm(baseline, "/tmp/x.pkl")
    ss = arm["checkpoints"]["cp1"]["search_strategy"]
    assert ss["type"] == "weighted_score_sum_knn"
    assert ss["top_k"] == 1
    assert ss["step_filter"] == "all"
    assert arm["key_builder"]["type"] == "cp1_spatial_pool_16"
    assert arm["write_policy"]["type"] == "never"


def test_sensitivity_arm_differs_only_in_normalizer_params(baseline):
    main = make_main_arm(baseline, "/tmp/lib_S6.pkl")
    fields = _norm_fields(main)
    # to_arm_fields rows: {method, params}. Method must match the arm's own.
    recal = {
        f: {"method": fields[f]["method"], "params": {"mu": 0.5, "sigma": 0.25, "squash": "tanh"}}
        for f in fields
    }

    sens = make_sensitivity_arm(main, recal)

    # The library is deliberately shared: normalization is a scoring transform.
    assert sens["backend"]["in_memory"]["preload_path"] == \
        main["backend"]["in_memory"]["preload_path"]

    for field in fields:
        assert _norm_fields(sens)[field]["params"] == recal[field]["params"]
        assert _norm_fields(sens)[field]["method"] == _norm_fields(main)[field]["method"]

    # Restoring the params must reproduce the main arm exactly.
    restored = copy.deepcopy(sens)
    for field in fields:
        _norm_fields(restored)[field]["params"] = _norm_fields(main)[field]["params"]
    assert restored == main


def test_sensitivity_arm_rejects_unknown_field(baseline):
    main = make_main_arm(baseline, "/tmp/x.pkl")
    with pytest.raises(KeyError, match="absent from the baseline normalizer block"):
        make_sensitivity_arm(main, {"vision_9": {"method": "zscore", "params": {"mu": 0.0}}})


def test_arm_inventory_is_twelve_main_plus_four_sensitivity():
    suites = ("libero_spatial", "libero_10")
    main = [arm_name(s, t) for s in suites for t in TIERS]
    sens = [arm_name(s, t, recal=True) for s in suites for t in SENSITIVITY_TIERS]
    assert len(main) == 12
    assert len(sens) == 4
    assert len(set(main + sens)) == 16


def test_emitted_arms_load_through_cache_config(baseline, tmp_path):
    """Every arm must survive the real loader, not just yaml.safe_load."""
    from openpi.cache.config import load_cache_config

    lib = tmp_path / "lib.pkl"
    lib.write_bytes(b"")  # loader validates the path shape, not the payload
    arm = make_main_arm(baseline, str(lib))
    path = tmp_path / "arm.yaml"
    path.write_text(yaml.safe_dump(arm, sort_keys=False))

    cfg = load_cache_config(str(path))
    cp1 = cfg.checkpoints["cp1"]
    assert cp1.judge.type == "always_hit"
    assert cp1.gate.type == "always_search"


# ---------------------------------------------------------------------------
# Arm matrix
# ---------------------------------------------------------------------------


def test_arm_matrix_is_eight_per_suite_sixteen_overall():
    """6 main + 2 sensitivity per suite; the plan's 16 is the two-suite total."""
    from exp.ablation_study.cache_size.emit_arm_matrix import build_matrix

    total = 0
    for suite in ("libero_spatial", "libero_10"):
        m = build_matrix(suite, "exp/ablation_study/cache_size/config")
        assert len(m["arms"]) == 8
        assert all(a["sidecar"] is None for a in m["arms"]), "pure-cache arms need no sidecar"
        names = [a["arm"] for a in m["arms"]]
        assert len(set(names)) == 8
        assert sum(n.endswith("_recal") for n in names) == 2
        total += len(m["arms"])
    assert total == 16


def test_arm_matrix_can_drop_sensitivity_arms():
    from exp.ablation_study.cache_size.emit_arm_matrix import build_matrix

    m = build_matrix("libero_10", "d", with_sensitivity=False)
    assert len(m["arms"]) == 6


def test_arm_name_separates_the_two_library_families():
    """Two arms with different libraries must not share a yaml_id.

    The journal keys episodes on ``yaml_id``; if the success-filtered S3 and the
    unfiltered S3 both answered to ``cache_size_libero_10_S3`` the two would
    merge into one arm's ledger, and the completeness gate would see 1000
    episodes where it expected 500 -- or, worse, 500 from a mix of both.
    """
    a = arm_name("libero_10", "S3", outcome_filter="all")
    b = arm_name("libero_10", "S3", outcome_filter="success")
    assert a != b
    assert a == "cache_size_libero_10_all_S3"
    assert b == "cache_size_libero_10_success_S3"
    # recal twins stay distinguishable too
    assert arm_name("libero_10", "S6", recal=True, outcome_filter="all") == \
        "cache_size_libero_10_all_S6_recal"
    # and the single-family layout is unchanged
    assert arm_name("libero_10", "S3") == "cache_size_libero_10_S3"

