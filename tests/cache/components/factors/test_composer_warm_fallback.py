"""Unit tests for WeightedSumWithWarmFallbackComposer (B4 refactor).

Plan §6.5 / §6.11 #2: framework no longer short-circuits on all-NaN; the
``all_nan_fallback: warm_start@0.7`` yaml field is gone. Equivalent
behaviour now lives in this composer subclass."""

from __future__ import annotations

import math

import pytest

from openpi.cache.components.factors.composers import (
    WeightedSumComposer,
    WeightedSumWithWarmFallbackComposer,
)
from openpi.cache.components.factors.composers.base import Composer
from openpi.cache.components.judge import HitType


# ----------------------------------------------------------------------
# Construction + protocol conformance
# ----------------------------------------------------------------------


def test_satisfies_composer_protocol() -> None:
    inst = WeightedSumWithWarmFallbackComposer(
        weights={"a": 1.0},
        full_hit_threshold=0.5,
        warm_fallback_start_t=0.7,
    )
    assert isinstance(inst, Composer)


def test_declared_dependencies_excludes_zero_weight_keys() -> None:
    inst = WeightedSumWithWarmFallbackComposer(
        weights={"a": 1.0, "b": 0.0, "c": 0.5},
        full_hit_threshold=0.5,
        warm_fallback_start_t=0.7,
    )
    assert inst.declared_dependencies == {"a", "c"}


# ----------------------------------------------------------------------
# Fallback behaviour — all-NaN
# ----------------------------------------------------------------------


def test_all_nan_weighted_keys_emit_warm_start_with_fallback_t() -> None:
    inst = WeightedSumWithWarmFallbackComposer(
        weights={"a": 1.0, "b": 1.0},
        full_hit_threshold=0.5,
        warm_fallback_start_t=0.7,
    )
    inst.bind_orientations({"a": "safe", "b": "safe"})
    out = inst.compose({"a": float("nan"), "b": float("nan")}, winner_id="w0")
    assert out.hit_type == HitType.WARM_START
    assert out.winner_id == "w0"
    assert math.isclose(out.start_t, 0.7)


def test_zero_weight_nans_do_not_trigger_fallback() -> None:
    """A zero-weight NaN must not count toward the all-NaN check."""
    inst = WeightedSumWithWarmFallbackComposer(
        weights={"a": 1.0, "ignore_me": 0.0},
        full_hit_threshold=0.5,
        warm_fallback_start_t=0.7,
    )
    inst.bind_orientations({"a": "safe", "ignore_me": "safe"})
    # `a` is finite at 0.9 → score 0.9 → FULL_HIT (above threshold).
    out = inst.compose({"a": 0.9, "ignore_me": float("nan")}, winner_id="w0")
    assert out.hit_type == HitType.FULL_HIT


def test_partial_nan_delegates_to_base_class_scoring() -> None:
    """If at least one weighted key is finite, fall through to the base
    WeightedSum scoring (does NOT trigger the warm fallback)."""
    inst = WeightedSumWithWarmFallbackComposer(
        weights={"a": 1.0, "b": 1.0},
        full_hit_threshold=0.5,
        warm_fallback_start_t=0.7,
    )
    inst.bind_orientations({"a": "safe", "b": "safe"})
    # a=0.9 finite, b=NaN → score = 0.9 (b skipped). 0.9 >= 0.5 → FULL_HIT.
    out = inst.compose({"a": 0.9, "b": float("nan")}, winner_id="w0")
    assert out.hit_type == HitType.FULL_HIT


def test_finite_below_threshold_returns_miss_not_fallback() -> None:
    """Fallback is for "all NaN", not for "score below threshold"."""
    inst = WeightedSumWithWarmFallbackComposer(
        weights={"a": 1.0},
        full_hit_threshold=0.5,
        warm_fallback_start_t=0.7,
    )
    inst.bind_orientations({"a": "safe"})
    out = inst.compose({"a": 0.1}, winner_id="w0")
    assert out.hit_type == HitType.MISS


# ----------------------------------------------------------------------
# Behavioural equivalence to WeightedSumComposer when there is signal
# ----------------------------------------------------------------------


def test_finite_inputs_match_base_class_decision() -> None:
    """When every weighted key is finite, both composers must agree."""
    weights = {"a": 1.0, "b": 1.0}
    base = WeightedSumComposer(weights=weights, full_hit_threshold=0.5)
    fb = WeightedSumWithWarmFallbackComposer(
        weights=weights, full_hit_threshold=0.5, warm_fallback_start_t=0.7
    )
    base.bind_orientations({"a": "safe", "b": "safe"})
    fb.bind_orientations({"a": "safe", "b": "safe"})

    factors = {"a": 0.6, "b": 0.4}
    out_base = base.compose(factors, winner_id="w0")
    out_fb = fb.compose(factors, winner_id="w0")
    assert out_base.hit_type == out_fb.hit_type
    assert out_base.composer_score == out_fb.composer_score


def test_warm_fallback_does_not_alter_warm_start_threshold_path() -> None:
    """If the regular WARM_START tier fires (score in [warm, full)), use
    the regular ``warm_start_t``, not the fallback ``warm_fallback_start_t``."""
    inst = WeightedSumWithWarmFallbackComposer(
        weights={"a": 1.0},
        full_hit_threshold=0.7,
        warm_start_threshold=0.3,
        warm_start_t=0.5,                 # regular warm tier
        warm_fallback_start_t=0.9,        # different from regular
    )
    inst.bind_orientations({"a": "safe"})
    out = inst.compose({"a": 0.4}, winner_id="w0")
    assert out.hit_type == HitType.WARM_START
    assert math.isclose(out.start_t, 0.5)   # regular warm, not fallback


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------


def test_no_weighted_keys_does_not_trigger_fallback() -> None:
    """If every weight is zero, ``weighted_keys`` is empty; the all-NaN
    guard short-circuits to ``False`` and we delegate to the base class
    (which itself returns MISS for total_w=0)."""
    inst = WeightedSumWithWarmFallbackComposer(
        weights={"a": 0.0, "b": 0.0},
        full_hit_threshold=0.5,
        warm_fallback_start_t=0.7,
    )
    inst.bind_orientations({"a": "safe", "b": "safe"})
    out = inst.compose({"a": float("nan"), "b": float("nan")}, winner_id="w0")
    assert out.hit_type == HitType.MISS
