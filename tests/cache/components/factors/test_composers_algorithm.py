"""Tests for Composer algorithms (B1).

Algorithms per `docs/cache/verdict_factor_judge.md` §6 Composer
contract + plan §3.3:
- WeightedSum: orientation-aware weighted sum, NaN skip, tier mapping
- AndGate: every key must pass (orientation-aware), NaN = MISS
- OrGate: any key passes, NaN skipped
- All three honor non_monotonic + directions config
"""

from __future__ import annotations

import math

import pytest

from openpi.cache.components.factors.composers import (
    AndGateComposer,
    OrGateComposer,
    WeightedSumComposer,
)
from openpi.cache.components.judge import HitType


# ------------------------------------------------------------------
# WeightedSumComposer
# ------------------------------------------------------------------


def test_weighted_sum_safe_orientation_passes_full_hit():
    c = WeightedSumComposer(weights={"k": 1.0}, full_hit_threshold=0.5)
    c.bind_orientations({"k": "safe"})
    out = c.compose({"k": 0.8}, winner_id="w")
    assert out.hit_type is HitType.FULL_HIT
    assert out.winner_id == "w"


def test_weighted_sum_risky_orientation_flips():
    # risky: high raw factor → low contribution. value 0.9 → contrib 0.1 → MISS
    c = WeightedSumComposer(weights={"k": 1.0}, full_hit_threshold=0.5)
    c.bind_orientations({"k": "risky"})
    assert c.compose({"k": 0.9}, winner_id="w").hit_type is HitType.MISS
    # value 0.1 → contrib 0.9 → FULL_HIT
    assert c.compose({"k": 0.1}, winner_id="w").hit_type is HitType.FULL_HIT


def test_weighted_sum_nan_keys_are_skipped():
    c = WeightedSumComposer(weights={"a": 1.0, "b": 1.0}, full_hit_threshold=0.5)
    c.bind_orientations({"a": "safe", "b": "safe"})
    # NaN for 'a' is skipped; 'b' alone determines verdict
    out = c.compose({"a": float("nan"), "b": 0.9}, winner_id="w")
    assert out.hit_type is HitType.FULL_HIT


def test_weighted_sum_all_nan_returns_miss_via_zero_total_w():
    c = WeightedSumComposer(weights={"a": 1.0}, full_hit_threshold=0.5)
    c.bind_orientations({"a": "safe"})
    out = c.compose({"a": float("nan")}, winner_id="w")
    assert out.hit_type is HitType.MISS


def test_weighted_sum_warm_start_tier():
    c = WeightedSumComposer(
        weights={"k": 1.0},
        full_hit_threshold=0.9,
        warm_start_threshold=0.5,
        warm_start_t=0.3,
    )
    c.bind_orientations({"k": "safe"})
    # 0.7 is below full_hit_threshold but above warm_start_threshold
    out = c.compose({"k": 0.7}, winner_id="w")
    assert out.hit_type is HitType.WARM_START
    assert out.start_t == 0.3
    assert out.winner_id == "w"


def test_weighted_sum_non_monotonic_high_direction():
    c = WeightedSumComposer(
        weights={"k": 1.0},
        full_hit_threshold=0.5,
        directions={"k": "high"},
    )
    c.bind_orientations({"k": "non_monotonic"})
    assert c.compose({"k": 0.8}, winner_id="w").hit_type is HitType.FULL_HIT
    assert c.compose({"k": 0.2}, winner_id="w").hit_type is HitType.MISS


def test_weighted_sum_non_monotonic_range_direction():
    c = WeightedSumComposer(
        weights={"k": 1.0},
        full_hit_threshold=0.5,
        directions={"k": "range:[0.3,0.7]"},
    )
    c.bind_orientations({"k": "non_monotonic"})
    assert c.compose({"k": 0.5}, winner_id="w").hit_type is HitType.FULL_HIT  # contrib=1
    assert c.compose({"k": 0.9}, winner_id="w").hit_type is HitType.MISS      # contrib=0


def test_weighted_sum_bind_raises_on_missing_directions():
    c = WeightedSumComposer(weights={"k": 1.0}, full_hit_threshold=0.5)
    with pytest.raises(ValueError, match="missing `directions`"):
        c.bind_orientations({"k": "non_monotonic"})


def test_weighted_sum_zero_weight_non_monotonic_skipped():
    # weight=0 → skip the directions requirement (the key won't aggregate)
    c = WeightedSumComposer(
        weights={"k": 0.0, "other": 1.0},
        full_hit_threshold=0.5,
    )
    c.bind_orientations({"k": "non_monotonic", "other": "safe"})  # no raise
    out = c.compose({"k": 0.5, "other": 0.9}, winner_id="w")
    assert out.hit_type is HitType.FULL_HIT


def test_weighted_sum_full_hit_threshold_emits_warm_start_when_configured():
    # If warm_start_t is set without warm_start_threshold, FULL pass yields WARM_START
    c = WeightedSumComposer(
        weights={"k": 1.0},
        full_hit_threshold=0.5,
        warm_start_t=0.4,
    )
    c.bind_orientations({"k": "safe"})
    out = c.compose({"k": 0.9}, winner_id="w")
    assert out.hit_type is HitType.WARM_START
    assert out.start_t == 0.4


# ------------------------------------------------------------------
# AndGateComposer
# ------------------------------------------------------------------


def test_and_gate_all_pass_yields_full_hit():
    c = AndGateComposer(per_factor_thresholds={"a": 0.5, "b": 0.5})
    c.bind_orientations({"a": "safe", "b": "safe"})
    out = c.compose({"a": 0.6, "b": 0.7}, winner_id="w")
    assert out.hit_type is HitType.FULL_HIT
    assert out.winner_id == "w"


def test_and_gate_one_fail_yields_miss():
    c = AndGateComposer(per_factor_thresholds={"a": 0.5, "b": 0.5})
    c.bind_orientations({"a": "safe", "b": "safe"})
    out = c.compose({"a": 0.6, "b": 0.4}, winner_id="w")
    assert out.hit_type is HitType.MISS


def test_and_gate_nan_treated_as_fail():
    c = AndGateComposer(per_factor_thresholds={"a": 0.5})
    c.bind_orientations({"a": "safe"})
    out = c.compose({"a": float("nan")}, winner_id="w")
    assert out.hit_type is HitType.MISS


def test_and_gate_risky_orientation():
    # risky: pass when v <= thr
    c = AndGateComposer(per_factor_thresholds={"a": 0.3})
    c.bind_orientations({"a": "risky"})
    assert c.compose({"a": 0.2}, winner_id="w").hit_type is HitType.FULL_HIT
    assert c.compose({"a": 0.4}, winner_id="w").hit_type is HitType.MISS


def test_and_gate_warm_start_emit():
    c = AndGateComposer(per_factor_thresholds={"a": 0.5}, warm_start_t=0.2)
    c.bind_orientations({"a": "safe"})
    out = c.compose({"a": 0.7}, winner_id="w")
    assert out.hit_type is HitType.WARM_START
    assert out.start_t == 0.2


def test_and_gate_bind_raises_on_missing_directions():
    c = AndGateComposer(per_factor_thresholds={"k": 0.5})
    with pytest.raises(ValueError, match="missing `directions`"):
        c.bind_orientations({"k": "non_monotonic"})


# ------------------------------------------------------------------
# OrGateComposer
# ------------------------------------------------------------------


def test_or_gate_any_pass_yields_full_hit():
    c = OrGateComposer(per_factor_thresholds={"a": 0.5, "b": 0.5})
    c.bind_orientations({"a": "safe", "b": "safe"})
    # b passes — full hit
    out = c.compose({"a": 0.4, "b": 0.6}, winner_id="w")
    assert out.hit_type is HitType.FULL_HIT


def test_or_gate_all_fail_yields_miss():
    c = OrGateComposer(per_factor_thresholds={"a": 0.5, "b": 0.5})
    c.bind_orientations({"a": "safe", "b": "safe"})
    out = c.compose({"a": 0.4, "b": 0.4}, winner_id="w")
    assert out.hit_type is HitType.MISS


def test_or_gate_nan_keys_skipped():
    c = OrGateComposer(per_factor_thresholds={"a": 0.5, "b": 0.5})
    c.bind_orientations({"a": "safe", "b": "safe"})
    # a is NaN (skipped); b passes
    out = c.compose({"a": float("nan"), "b": 0.7}, winner_id="w")
    assert out.hit_type is HitType.FULL_HIT


def test_or_gate_all_nan_yields_miss():
    c = OrGateComposer(per_factor_thresholds={"a": 0.5})
    c.bind_orientations({"a": "safe"})
    out = c.compose({"a": float("nan")}, winner_id="w")
    assert out.hit_type is HitType.MISS


def test_or_gate_non_monotonic_range():
    c = OrGateComposer(
        per_factor_thresholds={"k": 0.0},        # thr unused for range
        directions={"k": "range:[0.3, 0.7]"},
    )
    c.bind_orientations({"k": "non_monotonic"})
    assert c.compose({"k": 0.5}, winner_id="w").hit_type is HitType.FULL_HIT
    assert c.compose({"k": 0.1}, winner_id="w").hit_type is HitType.MISS
