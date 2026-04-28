"""Tests for `PercentileRollingNormalizer.preload_buffer` (B2.1).

Covers the warmup -> preload workflow used by verdict_factor_judge to bypass
cold-start NaN sentinels:
- Preloaded buffers skip the cold-start window on the next ``__call__``.
- Overfilled preloads truncate from the left via deque ``maxlen``.
- NaN entries in preload payloads are silently dropped (mirrors ``__call__``).
- Preloading an unbound key raises ``KeyError``.
- Partial preload (subset of bound keys) leaves unspecified keys untouched.
- Subsequent ``__call__`` continues appending into the preloaded buffer.
"""

from __future__ import annotations

import math

import pytest

from openpi.cache.components.factors.normalizers import PercentileRollingNormalizer


def test_preload_then_call_skips_cold_start():
    n = PercentileRollingNormalizer(window_size=10, cold_start_strategy="force_miss")
    n.bind_keys(["a"])
    n.preload_buffer({"a": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]})
    out = n({"a": 0.55})
    # Buffer was full at the moment of the call, so force_miss should NOT fire.
    assert not math.isnan(out["a"])
    assert 0.0 < out["a"] <= 1.0


def test_preload_truncates_to_window_size():
    n = PercentileRollingNormalizer(window_size=5, cold_start_strategy="force_miss")
    n.bind_keys(["a"])
    n.preload_buffer({"a": [float(i) for i in range(50)]})
    # Inspect the buffer state immediately after preload (before __call__'s own
    # append) so we can pin down truncation behaviour without conflating it
    # with the call-time append that evicts one more element from the left.
    buf = n._buffers["a"]
    assert buf.maxlen == 5
    assert len(buf) == 5
    # Preload appended 0..49 in order; deque maxlen=5 keeps only the most
    # recent five (oldest ~45 dropped from the left).
    assert list(buf) == [45.0, 46.0, 47.0, 48.0, 49.0]
    out = n({"a": 49.0})
    # Top of pool: 49 is the largest value in the truncated buffer.
    assert out["a"] == pytest.approx(1.0)
    # __call__ appended 49.0, evicting 45.0 from the left.
    assert len(buf) == 5
    assert list(buf) == [46.0, 47.0, 48.0, 49.0, 49.0]


def test_preload_skips_nan():
    n = PercentileRollingNormalizer(window_size=10, cold_start_strategy="force_miss")
    n.bind_keys(["a"])
    n.preload_buffer({"a": [1.0, float("nan"), 2.0, float("nan"), 3.0]})
    # NaN entries are silently dropped (matches __call__'s enqueue policy).
    assert len(n._buffers["a"]) == 3
    assert list(n._buffers["a"]) == [1.0, 2.0, 3.0]


def test_preload_unbound_key_raises_key_error():
    n = PercentileRollingNormalizer(window_size=10, cold_start_strategy="force_miss")
    n.bind_keys(["a"])
    with pytest.raises(KeyError):
        n.preload_buffer({"b": [1.0]})


def test_preload_subset_of_keys_leaves_others_alone():
    n = PercentileRollingNormalizer(window_size=10, cold_start_strategy="force_miss")
    n.bind_keys(["a", "b"])
    n.preload_buffer({"a": [1.0, 2.0, 3.0]})
    assert len(n._buffers["a"]) == 3
    assert len(n._buffers["b"]) == 0


def test_preload_then_call_appends_more():
    n = PercentileRollingNormalizer(window_size=10, cold_start_strategy="force_miss")
    n.bind_keys(["a"])
    n.preload_buffer({"a": [1.0, 2.0]})
    out = n({"a": 3.0})
    # Buffer has only 3 samples, below window_size=10, so force_miss fires.
    assert math.isnan(out["a"])
    # __call__ appended its raw value on top of the 2 preloaded ones.
    assert len(n._buffers["a"]) == 3
    assert list(n._buffers["a"]) == [1.0, 2.0, 3.0]
