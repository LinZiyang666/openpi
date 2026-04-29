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


def test_all_nan_fallback_can_emit_warm_start_for_cp1():
    e1 = _StubExtractor({"a": "safe"}, {"a": 0.1})
    composer = _StubComposer(JudgeResult(HitType.FULL_HIT, "winner"))
    norm = _StubNormalizer(lambda r: {k: float("nan") for k in r})
    cj = CompositeJudge(
        extractors=[e1],
        composer=composer,
        normalizer=norm,
        all_nan_fallback="warm_start",
        all_nan_fallback_start_t=0.7,
    )

    res = cj([_result()], CheckpointID.CP1, {})
    assert res.hit_type is HitType.WARM_START
    assert res.winner_id == "e1"
    assert res.start_t == pytest.approx(0.7)
    assert composer.calls == []


def test_all_nan_warm_start_fallback_is_cp1_only_defensively():
    e1 = _StubExtractor({"a": "safe"}, {"a": 0.1})
    composer = _StubComposer(JudgeResult(HitType.FULL_HIT, "winner"))
    norm = _StubNormalizer(lambda r: {k: float("nan") for k in r})
    cj = CompositeJudge(
        extractors=[e1],
        composer=composer,
        normalizer=norm,
        all_nan_fallback="warm_start",
        all_nan_fallback_start_t=0.7,
    )

    res = cj([_result()], CheckpointID.CP3, {})
    assert res.hit_type is HitType.MISS
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


# ------------------------------------------------------------------
# B1 end-to-end: CompositeJudge wired with real Composer + Normalizer
# ------------------------------------------------------------------


import math  # noqa: E402

from openpi.cache.components.factors.composers import (  # noqa: E402
    AndGateComposer,
    OrGateComposer,
    WeightedSumComposer,
)
from openpi.cache.components.factors.normalizers import (  # noqa: E402
    PercentileRollingNormalizer,
)
from openpi.cache.components.judge import HitType  # noqa: E402
from openpi.cache.storage_types import SearchResultLite  # noqa: E402
from openpi.cache.types import CheckpointID  # noqa: E402


def _result_lite(eid: str = "winner") -> SearchResultLite:
    return SearchResultLite(id=eid, score=1.0, checkpoint_id=CheckpointID.CP1)


def test_composite_full_hit_e2e():
    # Stub extractor producing a single 'safe' descriptor; passthrough
    # normalizer keeps the value as-is so we can land it above the
    # full_hit threshold deterministically.
    e = _StubExtractor({"k": "safe"}, {"k": 0.9})
    composer = WeightedSumComposer(weights={"k": 1.0}, full_hit_threshold=0.5)
    normalizer = PercentileRollingNormalizer(
        window_size=1, cold_start_strategy="passthrough",
    )
    cj = CompositeJudge(extractors=[e], composer=composer, normalizer=normalizer)

    out = cj([_result_lite()], CheckpointID.CP1, {})

    assert out.hit_type is HitType.FULL_HIT
    assert out.winner_id == "winner"


def test_composite_warm_start_e2e():
    e = _StubExtractor({"k": "safe"}, {"k": 0.6})
    composer = WeightedSumComposer(
        weights={"k": 1.0},
        full_hit_threshold=0.9,
        warm_start_threshold=0.5,
        warm_start_t=0.3,
    )
    normalizer = PercentileRollingNormalizer(
        window_size=1, cold_start_strategy="passthrough",
    )
    cj = CompositeJudge(extractors=[e], composer=composer, normalizer=normalizer)

    out = cj([_result_lite()], CheckpointID.CP1, {})

    assert out.hit_type is HitType.WARM_START
    assert out.start_t == 0.3


def test_composite_miss_e2e():
    e = _StubExtractor({"k": "safe"}, {"k": 0.1})
    composer = WeightedSumComposer(weights={"k": 1.0}, full_hit_threshold=0.5)
    normalizer = PercentileRollingNormalizer(
        window_size=1, cold_start_strategy="passthrough",
    )
    cj = CompositeJudge(extractors=[e], composer=composer, normalizer=normalizer)

    out = cj([_result_lite()], CheckpointID.CP1, {})

    assert out.hit_type is HitType.MISS


# ------------------------------------------------------------------
# export_factor_outputs diagnostic side-channel (CompositeJudge)
# ------------------------------------------------------------------


def test_factor_outputs_disabled_by_default():
    """Default flag off → JudgeResult.factor_outputs stays None on every path."""
    e = _StubExtractor({"k": "safe"}, {"k": 0.9})
    composer = WeightedSumComposer(weights={"k": 1.0}, full_hit_threshold=0.5)
    normalizer = PercentileRollingNormalizer(
        window_size=1, cold_start_strategy="passthrough",
    )
    cj = CompositeJudge(extractors=[e], composer=composer, normalizer=normalizer)

    out = cj([_result_lite()], CheckpointID.CP1, {})
    assert out.factor_outputs is None


def test_factor_outputs_normal_compose_path_payload_shape():
    """Normal compose → factor_outputs has raw, norm, score, sentinel=None."""
    e = _StubExtractor({"k": "safe"}, {"k": 0.9})
    composer = WeightedSumComposer(weights={"k": 1.0}, full_hit_threshold=0.5)
    normalizer = PercentileRollingNormalizer(
        window_size=1, cold_start_strategy="passthrough",
    )
    cj = CompositeJudge(
        extractors=[e], composer=composer, normalizer=normalizer,
        export_factor_outputs=True,
    )

    out = cj([_result_lite()], CheckpointID.CP1, {})
    assert out.hit_type is HitType.FULL_HIT
    fo = out.factor_outputs
    assert fo is not None
    assert set(fo.keys()) == {"raw", "norm", "score", "sentinel"}
    assert fo["raw"] == {"k": pytest.approx(0.9)}
    assert fo["norm"]["k"] == pytest.approx(0.9)
    assert fo["score"] == pytest.approx(0.9)
    assert fo["sentinel"] is None


def test_factor_outputs_all_nan_warm_fallback_payload():
    """All-NaN warm fallback → sentinel set, raw/norm NaN values become None,
    composer score absent (None)."""
    e = _StubExtractor({"k": "safe"}, {"k": float("nan")})
    composer = WeightedSumComposer(weights={"k": 1.0}, full_hit_threshold=0.5)
    norm = _StubNormalizer(lambda r: {k: float("nan") for k in r})
    cj = CompositeJudge(
        extractors=[e], composer=composer, normalizer=norm,
        all_nan_fallback="warm_start", all_nan_fallback_start_t=0.7,
        export_factor_outputs=True,
    )

    out = cj([_result_lite()], CheckpointID.CP1, {})
    assert out.hit_type is HitType.WARM_START
    fo = out.factor_outputs
    assert fo is not None
    assert fo["sentinel"] == "all_nan_warm_fallback"
    assert fo["raw"] == {"k": None}        # NaN → None
    assert fo["norm"] == {"k": None}
    assert fo["score"] is None             # composer was not invoked


def test_factor_outputs_all_nan_miss_payload():
    """All-NaN with fallback=miss → sentinel='all_nan_miss', score=None."""
    e = _StubExtractor({"k": "safe"}, {"k": float("nan")})
    composer = WeightedSumComposer(weights={"k": 1.0}, full_hit_threshold=0.5)
    norm = _StubNormalizer(lambda r: {k: float("nan") for k in r})
    cj = CompositeJudge(
        extractors=[e], composer=composer, normalizer=norm,
        export_factor_outputs=True,
    )

    out = cj([_result_lite()], CheckpointID.CP1, {})
    assert out.hit_type is HitType.MISS
    fo = out.factor_outputs
    assert fo is not None
    assert fo["sentinel"] == "all_nan_miss"
    assert fo["score"] is None


def test_factor_outputs_json_strict_serialisable():
    """factor_outputs payload must round-trip through strict JSON (no NaN
    literals leak from CompositeJudge — the producer is the one place that
    converts NaN to None)."""
    import json as _json

    e = _StubExtractor({"k": "safe", "j": "risky"}, {"k": float("nan"), "j": 0.3})
    composer = WeightedSumComposer(
        weights={"k": 1.0, "j": 1.0}, full_hit_threshold=0.5,
    )
    # Normalizer leaves the NaN in 'k' alone, returns valid 'j'.
    norm = _StubNormalizer(lambda r: {"k": float("nan"), "j": 0.3})
    cj = CompositeJudge(
        extractors=[e], composer=composer, normalizer=norm,
        export_factor_outputs=True,
    )

    out = cj([_result_lite()], CheckpointID.CP1, {})
    assert out.factor_outputs is not None
    # ``allow_nan=False`` is the strict-JSON gate every downstream reader
    # (jq, pandas, web-clients) imposes; CompositeJudge must satisfy it.
    s = _json.dumps(out.factor_outputs, allow_nan=False)
    assert "NaN" not in s
    parsed = _json.loads(s)
    assert parsed["raw"] == {"k": None, "j": pytest.approx(0.3)}


def test_composite_cold_start_short_circuits_to_miss():
    # force_miss normalizer with a window_size larger than what's been
    # seen → returns all-NaN → CompositeJudge short-circuits to MISS
    # without invoking Composer.
    e = _StubExtractor({"k": "safe"}, {"k": 0.9})
    composer = _StubComposer(JudgeResult(HitType.MISS))        # would crash if compose() is hit
    normalizer = PercentileRollingNormalizer(
        window_size=100, cold_start_strategy="force_miss",
    )
    cj = CompositeJudge(extractors=[e], composer=composer, normalizer=normalizer)

    out = cj([_result_lite()], CheckpointID.CP1, {})

    assert out.hit_type is HitType.MISS
    assert composer.calls == []            # composer never invoked


def test_composite_key_contract_violation_raises_runtime_error():
    # Extractor declares {'k'} but actually returns {'k', 'rogue'}
    e = _StubExtractor({"k": "safe"}, {"k": 0.5, "rogue": 0.5})
    composer = _StubComposer(JudgeResult(HitType.MISS))
    normalizer = PercentileRollingNormalizer(
        window_size=1, cold_start_strategy="passthrough",
    )
    cj = CompositeJudge(extractors=[e], composer=composer, normalizer=normalizer)

    with pytest.raises(RuntimeError, match="key contract violation"):
        cj([_result_lite()], CheckpointID.CP1, {})


def test_composite_andgate_e2e_all_pass():
    e1 = _StubExtractor({"a": "safe"}, {"a": 0.7})
    e2 = _StubExtractor({"b": "safe"}, {"b": 0.8})
    composer = AndGateComposer(per_factor_thresholds={"a": 0.5, "b": 0.5})
    cj = CompositeJudge(extractors=[e1, e2], composer=composer)
    out = cj([_result_lite()], CheckpointID.CP1, {})
    assert out.hit_type is HitType.FULL_HIT


def test_composite_orgate_e2e_one_passes():
    e1 = _StubExtractor({"a": "safe"}, {"a": 0.1})
    e2 = _StubExtractor({"b": "safe"}, {"b": 0.9})
    composer = OrGateComposer(per_factor_thresholds={"a": 0.5, "b": 0.5})
    cj = CompositeJudge(extractors=[e1, e2], composer=composer)
    out = cj([_result_lite()], CheckpointID.CP1, {})
    assert out.hit_type is HitType.FULL_HIT


def test_composite_empty_results_skips_extractors():
    # Empty results → CompositeJudge short-circuits to MISS without
    # calling extract / composer (per the early `if not results` guard).
    e = _StubExtractor({"k": "safe"}, {"k": 0.9})
    composer = _StubComposer(JudgeResult(HitType.MISS))
    cj = CompositeJudge(extractors=[e], composer=composer)
    out = cj([], CheckpointID.CP1, {})
    assert out.hit_type is HitType.MISS
    assert composer.calls == []
