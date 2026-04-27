"""B0 protocol-shape tests for Normalizers.

Algorithm body raises NotImplementedError in B0; only constructor +
bind_keys + Protocol structural typing + on_episode_start no-op are
exercised here.
"""

from __future__ import annotations

import pytest

from openpi.cache.components.factors.normalizers import (
    Normalizer,
    PercentileRollingNormalizer,
)


def test_percentile_rolling_implements_protocol():
    n = PercentileRollingNormalizer()
    assert isinstance(n, Normalizer)


def test_bind_keys_stores_keys():
    n = PercentileRollingNormalizer()
    keys = ["f2_var", "f1a_a_jerk"]
    n.bind_keys(keys)
    assert n._keys == keys


def test_call_b0_stub_raises():
    n = PercentileRollingNormalizer()
    n.bind_keys(["f2_var"])
    with pytest.raises(NotImplementedError, match="B1"):
        n({"f2_var": 0.5})


def test_on_episode_start_is_noop():
    n = PercentileRollingNormalizer()
    # Should not raise; no return value contract.
    assert n.on_episode_start() is None


def test_constructor_defaults():
    n = PercentileRollingNormalizer()
    assert n._window_size == 200
    assert n._cold_start_strategy == "force_miss"


def test_constructor_overrides():
    n = PercentileRollingNormalizer(window_size=50, cold_start_strategy="passthrough")
    assert n._window_size == 50
    assert n._cold_start_strategy == "passthrough"
