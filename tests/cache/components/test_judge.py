"""Tests for ThresholdJudge (T3.1–T3.9)."""

import torch

from openpi.cache.components.judge import HitType, SimilarityJudge, ThresholdJudge
from openpi.cache.storage_types import SearchResultLite
from openpi.cache.types import CheckpointID


def _result(id: str = "e1", score: float = 0.99, cp: CheckpointID = CheckpointID.CP1):
    return SearchResultLite(id=id, score=score, checkpoint_id=cp)


# T3.1
def test_threshold_judge_full_hit():
    j = ThresholdJudge(cp1_threshold=0.98)
    hit, winner = j([_result(score=0.99)], CheckpointID.CP1, {})
    assert hit == HitType.FULL_HIT
    assert winner == "e1"


# T3.2
def test_threshold_judge_miss_below_threshold():
    j = ThresholdJudge(cp1_threshold=0.98)
    hit, winner = j([_result(score=0.95)], CheckpointID.CP1, {})
    assert hit == HitType.MISS
    assert winner is None


# T3.3
def test_threshold_judge_miss_empty_results():
    j = ThresholdJudge()
    hit, winner = j([], CheckpointID.CP1, {})
    assert hit == HitType.MISS
    assert winner is None


# T3.4
def test_threshold_judge_exact_threshold_is_hit():
    j = ThresholdJudge(cp1_threshold=0.98)
    hit, winner = j([_result(score=0.98)], CheckpointID.CP1, {})
    assert hit == HitType.FULL_HIT
    assert winner == "e1"


# T3.5
def test_threshold_judge_cp3_uses_cp3_threshold():
    j = ThresholdJudge(cp1_threshold=0.98, cp3_threshold=0.95)
    hit, winner = j([_result(score=0.96, cp=CheckpointID.CP3)], CheckpointID.CP3, {})
    assert hit == HitType.FULL_HIT


# T3.6
def test_threshold_judge_cp3_miss():
    j = ThresholdJudge(cp3_threshold=0.95)
    hit, winner = j([_result(score=0.93, cp=CheckpointID.CP3)], CheckpointID.CP3, {})
    assert hit == HitType.MISS


# T3.7
def test_threshold_judge_unknown_cp_uses_default():
    j = ThresholdJudge()
    hit, _ = j([_result(score=0.99, cp=CheckpointID.CP2)], CheckpointID.CP2, {})
    assert hit == HitType.FULL_HIT


# T3.8
def test_threshold_judge_conforms_to_protocol():
    assert isinstance(ThresholdJudge(), SimilarityJudge)


# T3.9
def test_threshold_judge_custom_thresholds():
    j = ThresholdJudge(cp1_threshold=0.5, cp3_threshold=0.3)
    hit1, _ = j([_result(score=0.6)], CheckpointID.CP1, {})
    assert hit1 == HitType.FULL_HIT

    hit3, _ = j([_result(score=0.4, cp=CheckpointID.CP3)], CheckpointID.CP3, {})
    assert hit3 == HitType.FULL_HIT
