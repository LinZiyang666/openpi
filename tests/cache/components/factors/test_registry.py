"""Tests for the verdict-factor registry (refactor: 17 factors).

Replaces the legacy test of 5 names with a check on the new 17 factor
names produced by ``factors/online.py + offline.py + topk.py`` and
imported eagerly by ``factors/registry.py``.
"""

from __future__ import annotations

import pytest

from openpi.cache.components.factors import registry

# Refactor: descriptor orientation table now lives in _descriptor_kernel.
from openpi.cache.components.factors._descriptor_kernel import (
    _DESCRIPTOR_ORIENTATIONS,
)
from openpi.cache.components.factors.base import normalize_windows


# 17 names = 4 desc × 2 source × 2 channel + topk_action_variance.
_EXPECTED_NAMES: set[str] = {
    f"{desc}_{source}_{channel}"
    for desc in ("jerk", "direction", "dispersion", "path_length")
    for source in ("online", "offline")
    for channel in ("action", "state")
} | {"topk_action_variance"}


def test_known_contains_exactly_17_refactor_factors() -> None:
    assert registry.known() == _EXPECTED_NAMES


def test_legacy_names_no_longer_registered() -> None:
    """Legacy 5 names (f1a_a / f1a_t / f1b_a / f1b_t / f2) are gone."""
    legacy = {"f1a_a", "f1a_t", "f1b_a", "f1b_t", "f2"}
    assert legacy.isdisjoint(registry.known())


def test_get_class_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown factor name"):
        registry.get_class("nonexistent_factor")


def test_get_class_resolves_each_refactor_name() -> None:
    for name in _EXPECTED_NAMES:
        cls = registry.get_class(name)
        assert isinstance(cls, type), f"registry.get_class({name!r}) returned {cls!r}"


def test_register_duplicate_raises() -> None:
    @registry.register("__test_temp_unique__")
    class _Foo:
        pass

    with pytest.raises(ValueError, match="already registered"):
        @registry.register("__test_temp_unique__")
        class _Bar:
            pass


# ----------------------------------------------------------------------
# Refactored shared helpers
# ----------------------------------------------------------------------


def test_normalize_windows_dict() -> None:
    assert normalize_windows([{"past": 0, "future": 5}]) == [(0, 5)]


def test_normalize_windows_tuple() -> None:
    assert normalize_windows([(0, 5), (5, 10)]) == [(0, 5), (5, 10)]


def test_normalize_windows_mixed() -> None:
    out = normalize_windows([{"past": 0, "future": 3}, (1, 2), [4, 6]])
    assert out == [(0, 3), (1, 2), (4, 6)]


def test_descriptor_orientation_table_includes_refactor_names() -> None:
    assert _DESCRIPTOR_ORIENTATIONS["jerk"] == "risky"
    assert _DESCRIPTOR_ORIENTATIONS["direction"] == "safe"
    assert _DESCRIPTOR_ORIENTATIONS["dispersion"] == "non_monotonic"
    assert _DESCRIPTOR_ORIENTATIONS["path_length"] == "non_monotonic"
