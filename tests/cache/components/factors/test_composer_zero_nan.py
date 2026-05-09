"""Unit tests for WeightedSumZeroNanComposer (Phase 3/4 sweep composer).

Phase 3 plan §2 / §5.5: weighted sum (Sum w_k * contrib_k / Sum w_k)
where NaN raw values contribute 0 to the numerator but their weight is
retained in the denominator (zero-NaN). Two-tier inclusive cascade. No
all-NaN warm fallback path. Distinct from WeightedSumComposer (NaN-skip
denominator) and WeightedSumWithWarmFallbackComposer (all-NaN warm
fallback).

Phase 4 turned _score_only into a true weighted sum (was equal-average
pre-phase4). Tests #16-#20 lock in numeric weight sensitivity and
backward compatibility for the all-equal-weights case (= phase 3 yamls).

Test groups (mirrors plan §8.1):
  Core behaviour                  — #1-#6
  Boundary semantics              — #7-#12
  _score_only helper              — #13-#15
  Phase 4 weight sensitivity      — #16-#20
"""

from __future__ import annotations

import math

import pytest

from openpi.cache.components.factors.composers import (
    WeightedSumZeroNanComposer,
)
from openpi.cache.components.factors.composers.base import Composer
from openpi.cache.components.judge import HitType


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make(
    *,
    weights: dict[str, float] | None = None,
    fh: float = 0.5,
    ws: float = 0.2,
    warm_t: float = 0.5,
    directions: dict[str, str] | None = None,
    orientations: dict[str, str] | None = None,
) -> WeightedSumZeroNanComposer:
    weights = weights or {"a": 1.0, "b": 1.0}
    inst = WeightedSumZeroNanComposer(
        weights=weights,
        full_hit_threshold=fh,
        warm_start_threshold=ws,
        warm_start_t=warm_t,
        directions=directions,
    )
    inst.bind_orientations(orientations or {k: "safe" for k in weights})
    return inst


# ----------------------------------------------------------------------
# Construction + protocol conformance
# ----------------------------------------------------------------------


def test_satisfies_composer_protocol() -> None:
    inst = WeightedSumZeroNanComposer(
        weights={"a": 1.0},
        full_hit_threshold=0.5,
        warm_start_threshold=0.2,
        warm_start_t=0.5,
    )
    assert isinstance(inst, Composer)


def test_declared_dependencies_excludes_zero_weight_keys() -> None:
    inst = WeightedSumZeroNanComposer(
        weights={"a": 1.0, "b": 0.0, "c": 0.5},
        full_hit_threshold=0.5,
        warm_start_threshold=0.2,
        warm_start_t=0.5,
    )
    assert inst.declared_dependencies == {"a", "c"}


# ----------------------------------------------------------------------
# Core behaviour (plan §8.1 #1-#6)
# ----------------------------------------------------------------------


def test_1_equal_weight_sum_safe_orientation() -> None:
    """All keys numeric, safe orientation -> s = mean(v)."""
    inst = _make(fh=0.5, ws=0.2)
    out = inst.compose({"a": 0.6, "b": 0.8}, winner_id="w0")
    # mean = 0.7 >= 0.5 -> FULL_HIT
    assert out.hit_type == HitType.FULL_HIT
    assert out.composer_score == pytest.approx(0.7)


def test_2_nan_keys_contribute_zero_with_full_denominator() -> None:
    """NaN -> contrib 0 but still in denominator (matches §2.2)."""
    inst = _make(fh=0.5, ws=0.2)
    # contrib = (0.8 + 0) / 2 = 0.4 -> WS (0.2 <= 0.4 < 0.5)
    out = inst.compose({"a": 0.8, "b": float("nan")}, winner_id="w0")
    assert out.hit_type == HitType.WARM_START
    assert out.composer_score == pytest.approx(0.4)
    assert out.start_t == pytest.approx(0.5)


def test_3_directions_dispersion_range_binary_path_length_high_linear() -> None:
    """`range:[lo,hi]` -> 0/1 step; `high` -> linear v."""
    inst = _make(
        weights={"d": 1.0, "p": 1.0},
        fh=0.5,
        ws=0.2,
        directions={"d": "range:[0.3,0.7]", "p": "high"},
        orientations={"d": "non_monotonic", "p": "non_monotonic"},
    )
    # d=0.5 -> contrib 1 (in range); p=0.4 -> contrib 0.4 (high)
    # mean = (1 + 0.4) / 2 = 0.7 >= 0.5 -> FULL_HIT
    out = inst.compose({"d": 0.5, "p": 0.4}, winner_id="w0")
    assert out.hit_type == HitType.FULL_HIT
    assert out.composer_score == pytest.approx(0.7)
    # d=0.1 -> contrib 0 (out of range); p=0.6 -> contrib 0.6
    # mean = 0.3 -> WS
    out = inst.compose({"d": 0.1, "p": 0.6}, winner_id="w0")
    assert out.hit_type == HitType.WARM_START
    assert out.composer_score == pytest.approx(0.3)


def test_4_warm_start_emit_attaches_start_t_05() -> None:
    inst = _make(fh=0.7, ws=0.3, warm_t=0.5)
    out = inst.compose({"a": 0.5, "b": 0.5}, winner_id="w0")
    assert out.hit_type == HitType.WARM_START
    assert out.start_t == pytest.approx(0.5)
    assert out.winner_id == "w0"


def test_5_bind_orientations_missing_direction_on_non_monotonic_raises() -> None:
    inst = WeightedSumZeroNanComposer(
        weights={"a": 1.0},
        full_hit_threshold=0.5,
        warm_start_threshold=0.2,
        warm_start_t=0.5,
        directions=None,                # missing
    )
    with pytest.raises(ValueError, match="missing `directions`"):
        inst.bind_orientations({"a": "non_monotonic"})


def test_6_factor_outputs_composer_score_matches_score() -> None:
    """factor_outputs.composer_score == _score_only(...) on emit paths."""
    inst = _make(fh=0.5, ws=0.2)
    raw = {"a": 0.5, "b": 0.7}
    out = inst.compose(raw, winner_id="w0")
    s = inst._score_only(raw)
    assert out.composer_score == pytest.approx(s)


# ----------------------------------------------------------------------
# Boundary semantics (plan §8.1 #7-#12)
# ----------------------------------------------------------------------


def test_7_s_equal_full_hit_threshold_is_full_hit() -> None:
    """Inclusive-on-FH cascade: s == FH_thr -> FULL_HIT."""
    inst = _make(fh=0.5, ws=0.2)
    # mean(0.5, 0.5) = 0.5 == FH_thr
    out = inst.compose({"a": 0.5, "b": 0.5}, winner_id="w0")
    assert out.hit_type == HitType.FULL_HIT


def test_8_s_equal_warm_start_threshold_is_warm_start() -> None:
    """Inclusive-on-WS cascade: s == WS_thr (and < FH_thr) -> WARM_START."""
    inst = _make(fh=0.5, ws=0.2)
    # mean(0.2, 0.2) = 0.2 == WS_thr, < FH_thr
    out = inst.compose({"a": 0.2, "b": 0.2}, winner_id="w0")
    assert out.hit_type == HitType.WARM_START


def test_9_all_nan_with_positive_ws_thr_is_miss() -> None:
    """All-NaN -> s=0; with WS_thr > 0 -> MISS."""
    inst = _make(fh=0.5, ws=0.1)
    out = inst.compose({"a": float("nan"), "b": float("nan")}, winner_id="w0")
    assert out.hit_type == HitType.MISS
    assert out.composer_score == pytest.approx(0.0)


def test_10_all_nan_with_zero_ws_thr_is_warm_start() -> None:
    """The (0.5, 0.5) cell zero-MISS intent: s=0, WS_thr=0 -> WARM_START."""
    inst = _make(fh=0.5, ws=0.0)
    out = inst.compose({"a": float("nan"), "b": float("nan")}, winner_id="w0")
    # s = 0 >= WS_thr (0) and < FH_thr (0.5) -> WS
    assert out.hit_type == HitType.WARM_START
    assert out.composer_score == pytest.approx(0.0)


def test_11_all_nan_with_zero_thresholds_is_full_hit() -> None:
    """Degenerate cell WS=FH=0: s=0 >= FH -> FULL_HIT (FH wins)."""
    inst = _make(fh=0.0, ws=0.0)
    out = inst.compose({"a": float("nan"), "b": float("nan")}, winner_id="w0")
    assert out.hit_type == HitType.FULL_HIT


def test_12_negative_near_zero_score_is_miss_via_strict_lt() -> None:
    """s = -1e-12 vs WS_thr = 0 -> MISS (strict `<`)."""
    inst = _make(fh=0.5, ws=0.0)
    # Construct a contrib that sums to exactly -1e-12 by abusing risky
    # orientation: contrib = 1 - v on safe NaN-pad. Easier: bypass via
    # a single key whose orientation produces -1e-12. We'll verify the
    # cascade math directly: any s < 0 with WS_thr=0 must be MISS.
    # Simplest construction: monkey-feed _score_only output through
    # compose by making composer believe the inputs yield s = -1e-12.
    # Use risky orientation: 1 - 1.0 - epsilon path is awkward; instead
    # use safe orientation with one finite negative-summing input.
    # The composer has no input validation on raw values being in [0,1],
    # so feeding raw = -1e-12 produces contrib = -1e-12, mean = -5e-13 < 0.
    inst_safe = _make(weights={"a": 1.0}, fh=0.5, ws=0.0)
    out = inst_safe.compose({"a": -1e-12}, winner_id="w0")
    assert out.hit_type == HitType.MISS


# ----------------------------------------------------------------------
# _score_only helper (plan §8.1 #13-#15)
# ----------------------------------------------------------------------


def test_13_score_only_excludes_zero_weight_keys() -> None:
    """Zero-weight keys not in numerator nor denominator (§2.2 contract)."""
    inst = WeightedSumZeroNanComposer(
        weights={"a": 1.0, "b": 0.0, "c": 1.0},
        full_hit_threshold=0.5,
        warm_start_threshold=0.2,
        warm_start_t=0.5,
    )
    inst.bind_orientations({"a": "safe", "b": "safe", "c": "safe"})
    # Denominator = 2 (a, c), b ignored even when finite.
    s = inst._score_only({"a": 0.6, "b": 999.0, "c": 0.8})
    assert s == pytest.approx(0.7)


def test_14_score_only_all_zero_weights_returns_nan() -> None:
    """No non-zero weights -> NaN (degenerate construction)."""
    inst = WeightedSumZeroNanComposer(
        weights={"a": 0.0, "b": 0.0},
        full_hit_threshold=0.5,
        warm_start_threshold=0.2,
        warm_start_t=0.5,
    )
    inst.bind_orientations({"a": "safe", "b": "safe"})
    s = inst._score_only({"a": 0.5, "b": 0.5})
    assert math.isnan(s)


def test_15_compose_emits_miss_on_degenerate_all_zero_weight() -> None:
    """Degenerate composer (no non-zero weights) -> MISS at compose."""
    inst = WeightedSumZeroNanComposer(
        weights={"a": 0.0},
        full_hit_threshold=0.5,
        warm_start_threshold=0.2,
        warm_start_t=0.5,
    )
    inst.bind_orientations({"a": "safe"})
    out = inst.compose({"a": 0.5}, winner_id="w0")
    assert out.hit_type == HitType.MISS
    assert out.composer_score is None


# ----------------------------------------------------------------------
# Phase 4: numeric weight sensitivity
# ----------------------------------------------------------------------
#
# Pre-phase4 _score_only collapsed all non-zero weights to an equal
# average; phase4 changes the formula to true weighted sum. These tests
# lock in the post-phase4 contract and the backward compatibility of the
# all-equal-weights case (= phase 3 yamls keep producing identical
# numerical scores).


def test_16_score_only_uses_real_weighted_sum_not_equal_average() -> None:
    """Different non-zero weights produce different scores on the same
    factor input. Pre-phase4 (broken): both would produce the equal
    average 0.5. Post-phase4: a-heavy = (2*0.9 + 1*0.1) / 3 ~= 0.633.
    """
    composer_uni = _make(weights={"a": 1.0, "b": 1.0})
    composer_ah = _make(weights={"a": 2.0, "b": 1.0})
    factors = {"a": 0.9, "b": 0.1}
    s_uni = composer_uni._score_only(factors)
    s_ah = composer_ah._score_only(factors)
    assert s_uni == pytest.approx(0.5)                     # (0.9 + 0.1) / 2
    assert s_ah == pytest.approx((2 * 0.9 + 1 * 0.1) / 3)  # 1.9 / 3
    assert s_ah > s_uni + 0.05                             # numerical sensitivity >= 5e-2


def test_17_score_only_zero_weight_excluded_from_denominator() -> None:
    """Zero-weight keys do not contribute to numerator and do not pad
    the denominator. weights {a:0.5, b:0.0, c:0.5} ignores b regardless
    of b's factor value, so result equals (0.5*0.6 + 0.5*0.8) / (0.5+0.5)
    = 0.7.
    """
    inst = _make(weights={"a": 0.5, "b": 0.0, "c": 0.5})
    s = inst._score_only({"a": 0.6, "b": 999.0, "c": 0.8})
    assert s == pytest.approx(0.7)


def test_18_score_only_nan_keeps_weight_in_denominator() -> None:
    """NaN raw value contributes 0 to the numerator but its weight is
    retained in the denominator (zero-NaN semantics). weights {a:1, b:1}
    with a=0.8, b=NaN -> (1*0.8 + 1*0) / (1+1) = 0.4.

    Distinguishes this composer from regular WeightedSumComposer which
    would NaN-skip both numerator and denominator (giving 0.8).
    """
    inst = _make(weights={"a": 1.0, "b": 1.0})
    s = inst._score_only({"a": 0.8, "b": float("nan")})
    assert s == pytest.approx(0.4)


def test_19_score_only_all_zero_weights_returns_nan() -> None:
    """Degenerate construction: all non-zero weights stripped -> NaN.
    Mirrors test #14 but explicitly under the post-phase4 weighted-sum
    formula to lock the degenerate path.
    """
    inst = WeightedSumZeroNanComposer(
        weights={"a": 0.0, "b": 0.0},
        full_hit_threshold=0.5,
        warm_start_threshold=0.2,
        warm_start_t=0.5,
    )
    inst.bind_orientations({"a": "safe", "b": "safe"})
    s = inst._score_only({"a": 0.5, "b": 0.5})
    assert math.isnan(s)


def test_20_score_only_backward_compat_uniform_weights_match_phase3_behavior() -> None:
    """Phase 3 regression guard: when all non-zero weights are equal
    (e.g. {k: 1.0 for k}), the post-phase4 weighted sum is numerically
    identical to the pre-phase4 equal average over the same input.

    Pre-phase4 = sum(contrib_k) / N over non-zero-weight keys.
    Post-phase4 = sum(w_k * contrib_k) / sum(w_k); when all w_k = c, this
    reduces to c * sum(contrib_k) / (c * N) = sum(contrib_k) / N.
    """
    factors = {"a": 0.3, "b": 0.7, "c": 0.5, "d": 0.9}
    expected = sum(factors.values()) / len(factors)        # = 0.6, the pre-phase4 equal average
    inst_unit = _make(weights={k: 1.0 for k in factors})
    inst_scaled = _make(weights={k: 5.0 for k in factors})
    assert inst_unit._score_only(factors) == pytest.approx(expected)
    assert inst_scaled._score_only(factors) == pytest.approx(expected)
    assert inst_unit._score_only(factors) == pytest.approx(
        inst_scaled._score_only(factors)
    )
