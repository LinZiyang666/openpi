"""Tests for `PercentileRollingNormalizer.__call__` algorithm (B1).

Cold-start strategies per docs §2.8.8:
- force_miss (default): all-NaN until window_size buffered samples
- passthrough: skip normalization entirely
- lenient: compute percentile from partial buffer once N >= 10
"""

from __future__ import annotations

import math

import pytest

from openpi.cache.components.factors.normalizers import PercentileRollingNormalizer


# ------------------------------------------------------------------
# force_miss strategy
# ------------------------------------------------------------------


def test_force_miss_returns_nan_until_window_full():
    n = PercentileRollingNormalizer(window_size=5, cold_start_strategy="force_miss")
    n.bind_keys(["a"])
    # First 4 calls: still cold
    for _ in range(4):
        out = n({"a": 0.5})
        assert math.isnan(out["a"])
    # 5th call: window fills, returns percentile
    out = n({"a": 0.5})
    assert not math.isnan(out["a"])
    assert 0.0 < out["a"] <= 1.0


def test_force_miss_percentile_calculation():
    n = PercentileRollingNormalizer(window_size=4, cold_start_strategy="force_miss")
    n.bind_keys(["a"])
    n({"a": 1.0})
    n({"a": 2.0})
    n({"a": 3.0})
    out = n({"a": 4.0})
    # window now full: [1, 2, 3, 4]; rank of v=4 → 4/4 = 1.0
    assert out["a"] == pytest.approx(1.0)
    # next: insert 2.5 → window slides to [2, 3, 4, 2.5]; rank of 2.5 → 2/4
    out = n({"a": 2.5})
    assert out["a"] == pytest.approx(2.0 / 4.0)


# ------------------------------------------------------------------
# passthrough strategy
# ------------------------------------------------------------------


def test_passthrough_returns_raw_value():
    n = PercentileRollingNormalizer(window_size=10, cold_start_strategy="passthrough")
    n.bind_keys(["a"])
    out = n({"a": 0.42})
    assert out["a"] == 0.42


def test_passthrough_propagates_nan():
    n = PercentileRollingNormalizer(window_size=10, cold_start_strategy="passthrough")
    n.bind_keys(["a"])
    out = n({"a": float("nan")})
    assert math.isnan(out["a"])


# ------------------------------------------------------------------
# lenient strategy
# ------------------------------------------------------------------


def test_lenient_returns_nan_below_min_samples():
    n = PercentileRollingNormalizer(window_size=200, cold_start_strategy="lenient")
    n.bind_keys(["a"])
    # First 9 → still under the lenient floor (10)
    for _ in range(9):
        out = n({"a": 0.5})
        assert math.isnan(out["a"])
    # 10th → finally a percentile
    out = n({"a": 0.5})
    assert not math.isnan(out["a"])


def test_lenient_uses_partial_buffer():
    n = PercentileRollingNormalizer(window_size=100, cold_start_strategy="lenient")
    n.bind_keys(["a"])
    # Pre-fill 10 small values
    for v in [0.1] * 10:
        n({"a": v})
    # Insert a much larger value → percentile should be 1.0 (top of pool)
    out = n({"a": 5.0})
    assert out["a"] == pytest.approx(1.0)


# ------------------------------------------------------------------
# NaN handling
# ------------------------------------------------------------------


def test_nan_input_does_not_advance_window():
    n = PercentileRollingNormalizer(window_size=3, cold_start_strategy="force_miss")
    n.bind_keys(["a"])
    # NaN inputs are NOT enqueued (so they don't pollute the rolling window),
    # but the call still returns NaN as cold-start.
    for _ in range(5):
        out = n({"a": float("nan")})
        assert math.isnan(out["a"])
    # Now feed real values — window still cold (NaN didn't advance it)
    n({"a": 1.0})
    n({"a": 2.0})
    out = n({"a": 3.0})
    assert not math.isnan(out["a"])
    assert out["a"] == pytest.approx(1.0)               # 3/3 = top of pool


# ------------------------------------------------------------------
# Multiple keys, independent windows
# ------------------------------------------------------------------


def test_multiple_keys_each_have_independent_window():
    n = PercentileRollingNormalizer(window_size=3, cold_start_strategy="force_miss")
    n.bind_keys(["a", "b"])
    # Fill 'a' but not 'b'
    for v in [1.0, 2.0, 3.0]:
        n({"a": v, "b": float("nan")})
    out = n({"a": 4.0, "b": 1.0})
    assert not math.isnan(out["a"])
    # 'b' has only 1 sample → still cold
    assert math.isnan(out["b"])


# ------------------------------------------------------------------
# Constructor validation
# ------------------------------------------------------------------


def test_invalid_strategy_raises():
    with pytest.raises(ValueError, match="cold_start_strategy"):
        PercentileRollingNormalizer(cold_start_strategy="banana")


def test_invalid_window_size_raises():
    with pytest.raises(ValueError, match="window_size"):
        PercentileRollingNormalizer(window_size=0)


# ------------------------------------------------------------------
# Lifecycle hooks (no-op default)
# ------------------------------------------------------------------


def test_on_episode_start_does_not_clear_window():
    n = PercentileRollingNormalizer(window_size=2, cold_start_strategy="force_miss")
    n.bind_keys(["a"])
    n({"a": 1.0})
    n({"a": 2.0})
    n.on_episode_start()
    out = n({"a": 3.0})
    # If window had cleared, we'd be cold again (NaN). It doesn't.
    assert not math.isnan(out["a"])
