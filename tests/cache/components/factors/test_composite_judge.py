"""B0 tests for CompositeJudge structural shell.

Covers:
  - Construction collects+binds extractor metadata
  - min_required_top_k correctly picks max
  - Conflicting orientations across extractors raise at construction
  - Empty results -> MISS without invoking extractor / composer
  - Key contract assertion: extractor returning unexpected keys raises
  - Cold-start sentinel: all-NaN normalized output -> MISS
"""

from __future__ import annotations

import math

import pytest
import torch

from openpi.cache.components.factors.composers import WeightedSumComposer
from openpi.cache.components.factors.consensus import TopKActionConsensus
from openpi.cache.components.factors.normalizers import (
    PercentileRollingNormalizer,
)
from openpi.cache.components.judge import (
    CompositeJudge,
    HitType,
    JudgeResult,
    SimilarityJudge,
)
from openpi.cache.storage_types import SearchResultLite
from openpi.cache.types import CheckpointID


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _result(id: str = "e1", score: float = 0.99) -> SearchResultLite:
    return SearchResultLite(id=id, score=score, checkpoint_id=CheckpointID.CP1)


class _StubExtractor:
    """Synthetic extractor with controllable orientation map + output."""

    def __init__(
        self,
        orientations: dict[str, str],
        output: dict[str, float],
        required_top_k: int = 0,
    ):
        self.descriptor_orientations = dict(orientations)
        self._output = dict(output)
        self.required_top_k = required_top_k
        self.requires_library_stats = False
        self.requires_chain_walk = False

    @classmethod
    def describe(cls, params):
        return params.get("orientations", {})

    def extract(self, results, view, history, cached_data):
        return dict(self._output)


class _StubComposer:
    """Composer that returns a fixed JudgeResult and records calls."""

    def __init__(self, result: JudgeResult):
        self._result = result
        self.bound_orientations = None
        self.calls: list[tuple] = []

    def bind_orientations(self, orientations):
        self.bound_orientations = dict(orientations)

    def compose(self, factors, *, winner_id):
        self.calls.append((dict(factors), winner_id))
        return self._result


class _StubNormalizer:
    """Normalizer that applies a fixed transform; tracks bind_keys."""

    def __init__(self, transform):
        self._transform = transform
        self.bound_keys = None

    def bind_keys(self, keys):
        self.bound_keys = list(keys)

    def __call__(self, raw):
        return self._transform(raw)

    def on_episode_start(self):
        pass


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------


def test_composite_judge_collects_orientations_and_binds():
    e1 = _StubExtractor({"a": "safe"}, {"a": 0.5})
    e2 = _StubExtractor({"b": "risky"}, {"b": 0.2})
    composer = _StubComposer(JudgeResult(HitType.MISS))
    norm = _StubNormalizer(lambda r: r)
    cj = CompositeJudge(extractors=[e1, e2], composer=composer, normalizer=norm)
    assert composer.bound_orientations == {"a": "safe", "b": "risky"}
    assert set(norm.bound_keys) == {"a", "b"}
    assert isinstance(cj, SimilarityJudge)


def test_min_required_top_k_picks_max():
    e1 = _StubExtractor({"a": "safe"}, {"a": 0.5}, required_top_k=2)
    e2 = _StubExtractor({"b": "risky"}, {"b": 0.2}, required_top_k=7)
    composer = _StubComposer(JudgeResult(HitType.MISS))
    cj = CompositeJudge(extractors=[e1, e2], composer=composer)
    assert cj.min_required_top_k == 7


def test_conflicting_orientations_raise():
    e1 = _StubExtractor({"k": "safe"}, {"k": 0.5})
    e2 = _StubExtractor({"k": "risky"}, {"k": 0.2})
    composer = _StubComposer(JudgeResult(HitType.MISS))
    with pytest.raises(ValueError, match="conflicting"):
        CompositeJudge(extractors=[e1, e2], composer=composer)


def test_empty_extractors_raises():
    composer = _StubComposer(JudgeResult(HitType.MISS))
    with pytest.raises(ValueError, match="at least one"):
        CompositeJudge(extractors=[], composer=composer)


# ------------------------------------------------------------------
# __call__ behavior
# ------------------------------------------------------------------


def test_empty_results_returns_miss_without_invoking_extractor():
    e1 = _StubExtractor({"a": "safe"}, {"a": 0.5})
    composer = _StubComposer(JudgeResult(HitType.FULL_HIT, "winner"))
    cj = CompositeJudge(extractors=[e1], composer=composer)

    res = cj([], CheckpointID.CP1, {})
    assert res.hit_type is HitType.MISS
    assert composer.calls == []


def test_key_contract_violation_raises():
    # Extractor declares {"a"} but returns {"a", "extra"}.
    bad = _StubExtractor({"a": "safe"}, {"a": 0.5, "extra": 0.1})
    composer = _StubComposer(JudgeResult(HitType.MISS))
    cj = CompositeJudge(extractors=[bad], composer=composer)
    with pytest.raises(RuntimeError, match="key contract violation"):
        cj([_result()], CheckpointID.CP1, {})


def test_cold_start_all_nan_short_circuits_miss():
    e1 = _StubExtractor({"a": "safe", "b": "risky"}, {"a": 0.1, "b": 0.2})
    composer = _StubComposer(JudgeResult(HitType.FULL_HIT, "winner"))
    norm = _StubNormalizer(
        lambda r: {k: float("nan") for k in r}
    )
    cj = CompositeJudge(extractors=[e1], composer=composer, normalizer=norm)

    res = cj([_result()], CheckpointID.CP1, {})
    assert res.hit_type is HitType.MISS
    # Composer should NOT have been invoked.
    assert composer.calls == []


def test_partial_nan_does_not_short_circuit():
    e1 = _StubExtractor({"a": "safe", "b": "risky"}, {"a": 0.1, "b": 0.2})
    composer = _StubComposer(JudgeResult(HitType.FULL_HIT, "winner"))
    norm = _StubNormalizer(
        # Only one key NaN, the other still valid.
        lambda r: {"a": float("nan"), "b": 0.5}
    )
    cj = CompositeJudge(extractors=[e1], composer=composer, normalizer=norm)

    res = cj([_result()], CheckpointID.CP1, {})
    assert res.hit_type is HitType.FULL_HIT
    # CompositeJudge passes results[0].id as winner_id to the composer.
    assert composer.calls and composer.calls[0][1] == "e1"


def test_no_normalizer_uses_raw_factors():
    e1 = _StubExtractor({"a": "safe"}, {"a": 0.42})
    composer = _StubComposer(JudgeResult(HitType.FULL_HIT, "winner"))
    cj = CompositeJudge(extractors=[e1], composer=composer, normalizer=None)
    cj([_result()], CheckpointID.CP1, {})
    # Composer received raw value (no normalization)
    assert composer.calls[0][0]["a"] == 0.42


# ------------------------------------------------------------------
# Old judges accept new view/history kwargs
# ------------------------------------------------------------------


def test_legacy_threshold_judge_accepts_view_history_kwargs():
    from openpi.cache.components.judge import ThresholdJudge

    judge = ThresholdJudge(cp1_threshold=0.5)
    res = judge(
        [_result(score=0.9)], CheckpointID.CP1, {}, view=None, history=None
    )
    assert res.hit_type is HitType.FULL_HIT


def test_legacy_always_hit_judge_accepts_kwargs():
    from openpi.cache.components.judge import AlwaysHitJudge

    res = AlwaysHitJudge()(
        [_result()], CheckpointID.CP1, {}, view=object(), history=object()
    )
    assert res.hit_type is HitType.FULL_HIT
