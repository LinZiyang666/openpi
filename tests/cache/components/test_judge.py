"""Tests for ThresholdJudge (T3.1–T3.9) and AlwaysWarmStartJudge."""

import pytest
import torch

from openpi.cache.components.judge import (
    AlwaysWarmStartJudge,
    HitType,
    JudgeResult,
    SimilarityJudge,
    ThresholdJudge,
)
from openpi.cache.storage_types import SearchResultLite
from openpi.cache.types import CheckpointID


def _result(id: str = "e1", score: float = 0.99, cp: CheckpointID = CheckpointID.CP1):
    return SearchResultLite(id=id, score=score, checkpoint_id=cp)


# T3.1
def test_threshold_judge_full_hit():
    j = ThresholdJudge(cp1_threshold=0.98)
    r = j([_result(score=0.99)], CheckpointID.CP1, {})
    assert r.hit_type == HitType.FULL_HIT
    assert r.winner_id == "e1"


# T3.2
def test_threshold_judge_miss_below_threshold():
    j = ThresholdJudge(cp1_threshold=0.98)
    r = j([_result(score=0.95)], CheckpointID.CP1, {})
    assert r.hit_type == HitType.MISS
    assert r.winner_id is None


# T3.3
def test_threshold_judge_miss_empty_results():
    j = ThresholdJudge()
    r = j([], CheckpointID.CP1, {})
    assert r.hit_type == HitType.MISS
    assert r.winner_id is None


# T3.4
def test_threshold_judge_exact_threshold_is_hit():
    j = ThresholdJudge(cp1_threshold=0.98)
    r = j([_result(score=0.98)], CheckpointID.CP1, {})
    assert r.hit_type == HitType.FULL_HIT
    assert r.winner_id == "e1"


# T3.5
def test_threshold_judge_cp3_uses_cp3_threshold():
    j = ThresholdJudge(cp1_threshold=0.98, cp3_threshold=0.95)
    r = j([_result(score=0.96, cp=CheckpointID.CP3)], CheckpointID.CP3, {})
    assert r.hit_type == HitType.FULL_HIT


# T3.6
def test_threshold_judge_cp3_miss():
    j = ThresholdJudge(cp3_threshold=0.95)
    r = j([_result(score=0.93, cp=CheckpointID.CP3)], CheckpointID.CP3, {})
    assert r.hit_type == HitType.MISS


# T3.7
def test_threshold_judge_unknown_cp_uses_default():
    j = ThresholdJudge()
    r = j([_result(score=0.99, cp=CheckpointID.CP2)], CheckpointID.CP2, {})
    assert r.hit_type == HitType.FULL_HIT


# T3.8
def test_threshold_judge_conforms_to_protocol():
    assert isinstance(ThresholdJudge(), SimilarityJudge)


# T3.9
def test_threshold_judge_custom_thresholds():
    j = ThresholdJudge(cp1_threshold=0.5, cp3_threshold=0.3)
    r1 = j([_result(score=0.6)], CheckpointID.CP1, {})
    assert r1.hit_type == HitType.FULL_HIT

    r3 = j([_result(score=0.4, cp=CheckpointID.CP3)], CheckpointID.CP3, {})
    assert r3.hit_type == HitType.FULL_HIT


# --- Warm tiers tests ---

_TIERS = [
    {"threshold": 0.95, "start_t": 0.3},
    {"threshold": 0.90, "start_t": 0.5},
    {"threshold": 0.85, "start_t": 0.7},
]


def test_warm_tiers_full_hit_above_threshold():
    j = ThresholdJudge(cp1_threshold=0.98, warm_tiers=_TIERS)
    r = j([_result(score=0.99)], CheckpointID.CP1, {})
    assert r.hit_type == HitType.FULL_HIT
    assert r.start_t is None


def test_warm_tiers_first_tier():
    j = ThresholdJudge(cp1_threshold=0.98, warm_tiers=_TIERS)
    r = j([_result(score=0.96)], CheckpointID.CP1, {})
    assert r.hit_type == HitType.WARM_START
    assert r.start_t == 0.3
    assert r.winner_id == "e1"


def test_warm_tiers_second_tier():
    j = ThresholdJudge(cp1_threshold=0.98, warm_tiers=_TIERS)
    r = j([_result(score=0.91)], CheckpointID.CP1, {})
    assert r.hit_type == HitType.WARM_START
    assert r.start_t == 0.5


def test_warm_tiers_third_tier():
    j = ThresholdJudge(cp1_threshold=0.98, warm_tiers=_TIERS)
    r = j([_result(score=0.86)], CheckpointID.CP1, {})
    assert r.hit_type == HitType.WARM_START
    assert r.start_t == 0.7


def test_warm_tiers_below_all_tiers_is_miss():
    j = ThresholdJudge(cp1_threshold=0.98, warm_tiers=_TIERS)
    r = j([_result(score=0.80)], CheckpointID.CP1, {})
    assert r.hit_type == HitType.MISS
    assert r.start_t is None


def test_warm_tiers_cp3_ignored():
    j = ThresholdJudge(cp1_threshold=0.98, cp3_threshold=0.95, warm_tiers=_TIERS)
    r = j([_result(score=0.91, cp=CheckpointID.CP3)], CheckpointID.CP3, {})
    assert r.hit_type == HitType.MISS


def test_warm_tiers_none_backward_compatible():
    j = ThresholdJudge(cp1_threshold=0.98, warm_tiers=None)
    r = j([_result(score=0.96)], CheckpointID.CP1, {})
    assert r.hit_type == HitType.MISS
    assert r.start_t is None


def test_warm_tiers_empty_list_backward_compatible():
    j = ThresholdJudge(cp1_threshold=0.98, warm_tiers=[])
    r = j([_result(score=0.96)], CheckpointID.CP1, {})
    assert r.hit_type == HitType.MISS


# --- AlwaysWarmStartJudge ---


def test_always_warm_start_empty_results_is_miss():
    j = AlwaysWarmStartJudge(start_t=0.5)
    r = j([], CheckpointID.CP1, {})
    assert r.hit_type == HitType.MISS
    assert r.winner_id is None
    assert r.start_t is None


def test_always_warm_start_cp1_emits_warm_start():
    j = AlwaysWarmStartJudge(start_t=0.5)
    r = j([_result(id="e42", score=0.01)], CheckpointID.CP1, {})
    assert r.hit_type == HitType.WARM_START
    assert r.winner_id == "e42"
    assert r.start_t == 0.5


def test_always_warm_start_cp3_defensive_full_hit():
    # Defensive fallback: config validation rejects this combo, but if it
    # somehow reaches runtime we fall back to FULL_HIT (observable) rather
    # than silently crashing downstream.
    j = AlwaysWarmStartJudge(start_t=0.5)
    r = j([_result(id="e9", score=0.01, cp=CheckpointID.CP3)], CheckpointID.CP3, {})
    assert r.hit_type == HitType.FULL_HIT
    assert r.winner_id == "e9"


def test_always_warm_start_non_canonical_start_t_raises():
    with pytest.raises(ValueError, match="start_t"):
        AlwaysWarmStartJudge(start_t=0.5001)


def test_always_warm_start_canonical_rounding_normalized():
    # Represents the 0.3 ≈ 0.30000000000000004 float-parsing case.
    j = AlwaysWarmStartJudge(start_t=0.30000000000000004)
    r = j([_result()], CheckpointID.CP1, {})
    assert r.start_t == 0.3


def test_always_warm_start_conforms_to_protocol():
    assert isinstance(AlwaysWarmStartJudge(start_t=0.5), SimilarityJudge)
