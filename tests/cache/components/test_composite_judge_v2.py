"""Unit tests for the refactored 4-layer ``CompositeJudge`` (B5)."""

from __future__ import annotations

import math

import pytest
import torch

from openpi.cache.components.composite_judge import CompositeJudge
from openpi.cache.components.factors.base import (
    CalibrationSamples,
    Factor,
    FactorContext,
    HistoryView,
    LibraryStats,
)
from openpi.cache.components.factors.calibrations import PercentileRollingCalibration
from openpi.cache.components.factors.composers import (
    AndGateComposer,
    WeightedSumComposer,
    WeightedSumWithWarmFallbackComposer,
)
from openpi.cache.components.factors.normalization import ZScoreNormalization
from openpi.cache.components.judge import HitType
from openpi.cache.storage_types import CachePayload, SearchResultLite
from openpi.cache.types import CheckpointID


# ----------------------------------------------------------------------
# Lightweight dummy factor for assembly testing
# ----------------------------------------------------------------------


class _DummyFactor:
    """A factor that emits a fixed value (or NaN) for one safe key."""

    requires_chain_walk = False
    required_top_k = 0

    def __init__(self, *, key: str = "dummy_safe", value: float = 0.5,
                 orientation: str = "safe") -> None:
        self._key = key
        self._value = value
        self.descriptor_orientations = {key: orientation}

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        return {params.get("key", "dummy_safe"): params.get("orientation", "safe")}

    def extract(self, ctx: FactorContext) -> dict[str, float]:
        return {self._key: self._value}


def _make_lib_stats() -> LibraryStats:
    a = torch.ones(2, dtype=torch.float32)
    s = torch.ones(2, dtype=torch.float32)
    return LibraryStats(
        action_sigma=a, action_active_mask=torch.ones_like(a, dtype=torch.bool),
        state_sigma=s, state_active_mask=torch.ones_like(s, dtype=torch.bool),
    )


def _make_calibration(keys: list[str], window_size: int = 10) -> PercentileRollingCalibration:
    # Samples in [0, 1] uniformly so a fresh 0.5 maps to ~50th percentile,
    # 0.95 maps near the top, and 0.05 maps near the bottom.
    samples = CalibrationSamples({
        k: [float(i) / window_size for i in range(window_size)] for k in keys
    })
    return PercentileRollingCalibration(samples, window_size=window_size)


def _make_results(eid: str = "w0") -> list[SearchResultLite]:
    return [SearchResultLite(id=eid, score=1.0, checkpoint_id=CheckpointID.CP1)]


# ----------------------------------------------------------------------
# Construction — orientation conflict + missing dependency
# ----------------------------------------------------------------------


def test_construction_rejects_conflicting_orientations() -> None:
    f1 = _DummyFactor(key="k", orientation="safe")
    f2 = _DummyFactor(key="k", orientation="risky")
    norm = ZScoreNormalization(_make_lib_stats())
    calib = _make_calibration(["k"])
    composer = WeightedSumComposer(weights={"k": 1.0}, full_hit_threshold=0.5)
    with pytest.raises(ValueError, match="conflicting orientations"):
        CompositeJudge(norm, [f1, f2], calib, composer)


def test_construction_rejects_composer_missing_dependency() -> None:
    f = _DummyFactor(key="k1")
    norm = ZScoreNormalization(_make_lib_stats())
    calib = _make_calibration(["k1", "k_missing"])
    composer = WeightedSumComposer(
        weights={"k1": 1.0, "k_missing": 1.0}, full_hit_threshold=0.5,
    )
    with pytest.raises(ValueError, match="requires factor keys not"):
        CompositeJudge(norm, [f], calib, composer)


def test_construction_rejects_empty_factors_list() -> None:
    norm = ZScoreNormalization(_make_lib_stats())
    calib = _make_calibration([])
    composer = WeightedSumComposer(weights={}, full_hit_threshold=0.5)
    with pytest.raises(ValueError, match="at least one Factor"):
        CompositeJudge(norm, [], calib, composer)


def test_construction_calls_calibration_bind_keys_with_union() -> None:
    """If calibration samples lack one of the bound keys, bind_keys raises."""
    f = _DummyFactor(key="k1")
    norm = ZScoreNormalization(_make_lib_stats())
    # Samples cover only k_other; calibration.bind_keys should fail on k1.
    calib = PercentileRollingCalibration(
        CalibrationSamples({"k_other": [0.0] * 20}),
        window_size=10,
    )
    composer = WeightedSumComposer(weights={"k1": 1.0}, full_hit_threshold=0.5)
    with pytest.raises(KeyError, match="missing key 'k1'"):
        CompositeJudge(norm, [f], calib, composer)


# ----------------------------------------------------------------------
# min_required_top_k aggregation
# ----------------------------------------------------------------------


class _FactorWithTopK:
    requires_chain_walk = False

    def __init__(self, K: int) -> None:
        self.required_top_k = K
        self.descriptor_orientations = {"k": "risky"}

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        return {"k": "risky"}

    def extract(self, ctx: FactorContext) -> dict[str, float]:
        return {"k": 0.5}


def test_min_required_top_k_takes_max_across_factors() -> None:
    f1 = _DummyFactor(key="a")
    f2 = _FactorWithTopK(K=5)
    f1.descriptor_orientations = {"a": "safe"}
    norm = ZScoreNormalization(_make_lib_stats())
    calib = _make_calibration(["a", "k"])
    composer = WeightedSumComposer(
        weights={"a": 1.0, "k": 0.5},
        full_hit_threshold=0.5,
        directions={},
    )
    composer.bind_orientations({"a": "safe", "k": "risky"})  # cleared by judge below
    judge = CompositeJudge(norm, [f1, f2], calib, composer)
    assert judge.min_required_top_k == 5


# ----------------------------------------------------------------------
# Verdict path — empty results short-circuit
# ----------------------------------------------------------------------


def test_empty_results_short_circuits_to_miss() -> None:
    f = _DummyFactor()
    norm = ZScoreNormalization(_make_lib_stats())
    calib = _make_calibration(["dummy_safe"])
    composer = WeightedSumComposer(
        weights={"dummy_safe": 1.0}, full_hit_threshold=0.5,
    )
    judge = CompositeJudge(norm, [f], calib, composer)
    out = judge([], CheckpointID.CP1, {})
    assert out.hit_type == HitType.MISS
    assert out.winner_id is None


def test_empty_results_does_not_invoke_warm_fallback_composer() -> None:
    """Even with WarmFallback composer, empty results still MISS at CompositeJudge level."""
    f = _DummyFactor(value=float("nan"))
    norm = ZScoreNormalization(_make_lib_stats())
    calib = _make_calibration(["dummy_safe"])
    composer = WeightedSumWithWarmFallbackComposer(
        weights={"dummy_safe": 1.0},
        full_hit_threshold=0.5,
        warm_fallback_start_t=0.7,
    )
    judge = CompositeJudge(norm, [f], calib, composer)
    out = judge([], CheckpointID.CP1, {})
    assert out.hit_type == HitType.MISS
    assert out.winner_id is None


# ----------------------------------------------------------------------
# Verdict path — happy + factor_outputs schema
# ----------------------------------------------------------------------


def test_full_hit_path() -> None:
    """Single safe factor at value 0.9 → calibrated ≈ 1.0 → FULL_HIT."""
    f = _DummyFactor(value=0.95)
    norm = ZScoreNormalization(_make_lib_stats())
    calib = _make_calibration(["dummy_safe"])
    composer = WeightedSumComposer(
        weights={"dummy_safe": 1.0}, full_hit_threshold=0.5,
    )
    judge = CompositeJudge(norm, [f], calib, composer)
    out = judge(_make_results(), CheckpointID.CP1, {})
    assert out.hit_type == HitType.FULL_HIT
    assert out.winner_id == "w0"


def test_factor_outputs_schema_version_2_when_export_enabled() -> None:
    f = _DummyFactor(value=0.5)
    norm = ZScoreNormalization(_make_lib_stats())
    calib = _make_calibration(["dummy_safe"])
    composer = WeightedSumComposer(
        weights={"dummy_safe": 1.0}, full_hit_threshold=0.5,
    )
    judge = CompositeJudge(norm, [f], calib, composer, export_factor_outputs=True)
    out = judge(_make_results(), CheckpointID.CP1, {})
    assert out.factor_outputs is not None
    assert out.factor_outputs["schema_version"] == 2
    assert "raw" in out.factor_outputs
    assert "calibrated" in out.factor_outputs   # renamed from "norm"
    assert "norm" not in out.factor_outputs
    assert "composer_score" in out.factor_outputs
    assert "score" not in out.factor_outputs            # renamed
    assert "sentinel" not in out.factor_outputs         # dropped (plan §6.10)


def test_factor_outputs_omitted_when_export_disabled() -> None:
    f = _DummyFactor()
    norm = ZScoreNormalization(_make_lib_stats())
    calib = _make_calibration(["dummy_safe"])
    composer = WeightedSumComposer(
        weights={"dummy_safe": 1.0}, full_hit_threshold=0.5,
    )
    judge = CompositeJudge(norm, [f], calib, composer)   # export_factor_outputs=False default
    out = judge(_make_results(), CheckpointID.CP1, {})
    assert out.factor_outputs is None


# ----------------------------------------------------------------------
# Key contract assertion
# ----------------------------------------------------------------------


class _MisbehavingFactor:
    requires_chain_walk = False
    required_top_k = 0
    descriptor_orientations = {"declared_a": "safe"}

    @classmethod
    def describe(cls, params: dict) -> dict[str, str]:
        return {"declared_a": "safe"}

    def extract(self, ctx: FactorContext) -> dict[str, float]:
        # Returns a different key than declared → contract violation.
        return {"surprise_b": 0.5}


def test_key_contract_violation_raises_runtime_error() -> None:
    f = _MisbehavingFactor()
    norm = ZScoreNormalization(_make_lib_stats())
    calib = _make_calibration(["declared_a"])
    composer = WeightedSumComposer(
        weights={"declared_a": 1.0}, full_hit_threshold=0.5,
    )
    judge = CompositeJudge(norm, [f], calib, composer)
    with pytest.raises(RuntimeError, match="key contract violation"):
        judge(_make_results(), CheckpointID.CP1, {})


# ----------------------------------------------------------------------
# Lifecycle hooks
# ----------------------------------------------------------------------


def test_on_episode_start_does_not_reset_buffers() -> None:
    """Plan §6.11 #9: rolling buffer persists across episodes."""
    f = _DummyFactor(value=0.5)
    norm = ZScoreNormalization(_make_lib_stats())
    calib = _make_calibration(["dummy_safe"], window_size=5)
    composer = WeightedSumComposer(
        weights={"dummy_safe": 1.0}, full_hit_threshold=0.5,
    )
    judge = CompositeJudge(norm, [f], calib, composer, export_factor_outputs=True)
    # First verdict
    out1 = judge(_make_results(), CheckpointID.CP1, {})
    cal1 = out1.factor_outputs["calibrated"]["dummy_safe"]
    # Reset (no-op per plan)
    judge.on_episode_start()
    # Second verdict — calibration should be deterministic given identical input
    out2 = judge(_make_results(), CheckpointID.CP1, {})
    cal2 = out2.factor_outputs["calibrated"]["dummy_safe"]
    # Both finite (buffer was prefilled, no cold-start).
    assert cal1 is not None and cal2 is not None
    assert math.isfinite(cal1) and math.isfinite(cal2)


def test_record_action_is_noop() -> None:
    f = _DummyFactor()
    norm = ZScoreNormalization(_make_lib_stats())
    calib = _make_calibration(["dummy_safe"])
    composer = WeightedSumComposer(
        weights={"dummy_safe": 1.0}, full_hit_threshold=0.5,
    )
    judge = CompositeJudge(norm, [f], calib, composer)
    judge.record_action(torch.zeros(2))   # should not raise / mutate anything
