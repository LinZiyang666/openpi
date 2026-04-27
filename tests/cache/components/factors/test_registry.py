"""Tests for the verdict factor registry (B0): register / get_class /
build / known + duplicate-name raise + unknown-name raise + factor
metadata + describe(cls, params) consistency.
"""

from __future__ import annotations

import pytest

from openpi.cache.components.factors import registry
from openpi.cache.components.factors.consensus import TopKActionConsensus
from openpi.cache.components.factors.runtime_continuity import (
    RuntimeContinuityAction,
    RuntimeContinuityState,
    _RuntimeContinuityBase,
)
from openpi.cache.components.factors.source_window import (
    SourceWindowSmoothnessAction,
    SourceWindowSmoothnessState,
    _DESCRIPTOR_ORIENTATIONS,
    _SourceWindowSmoothnessBase,
    _normalize_windows,
)


# ------------------------------------------------------------------
# known() / get_class basics
# ------------------------------------------------------------------


def test_known_contains_all_b0_factors():
    names = registry.known()
    assert {"f1a_a", "f1a_t", "f1b_a", "f1b_t", "f2"} <= names


def test_get_class_returns_correct_classes():
    assert registry.get_class("f1a_a") is RuntimeContinuityAction
    assert registry.get_class("f1a_t") is RuntimeContinuityState
    assert registry.get_class("f1b_a") is SourceWindowSmoothnessAction
    assert registry.get_class("f1b_t") is SourceWindowSmoothnessState
    assert registry.get_class("f2") is TopKActionConsensus


def test_get_class_unknown_raises():
    with pytest.raises(ValueError, match="Unknown factor name"):
        registry.get_class("nonexistent_factor")


def test_register_duplicate_raises():
    @registry.register("__test_temp__")
    class _Foo:
        pass

    with pytest.raises(ValueError, match="already registered"):
        @registry.register("__test_temp__")
        class _Bar:
            pass


# ------------------------------------------------------------------
# Class-level capability flags
# ------------------------------------------------------------------


def test_runtime_continuity_capability_flags():
    assert RuntimeContinuityAction.source == "action"
    assert RuntimeContinuityAction.requires_chain_walk is False
    assert RuntimeContinuityAction.requires_library_stats is True
    assert RuntimeContinuityState.source == "state"
    assert RuntimeContinuityState.requires_chain_walk is True
    assert RuntimeContinuityState.requires_library_stats is True


def test_source_window_capability_flags():
    assert SourceWindowSmoothnessAction.source == "action"
    assert SourceWindowSmoothnessAction.requires_chain_walk is False
    assert SourceWindowSmoothnessAction.requires_library_stats is True
    assert SourceWindowSmoothnessState.source == "state"
    assert SourceWindowSmoothnessState.requires_chain_walk is False
    assert SourceWindowSmoothnessState.requires_library_stats is True


def test_consensus_capability_flags():
    assert TopKActionConsensus.requires_library_stats is False
    assert TopKActionConsensus.requires_chain_walk is False


# ------------------------------------------------------------------
# describe(cls, params) classmethod — pure, no construction
# ------------------------------------------------------------------


def test_runtime_continuity_describe_action():
    out = RuntimeContinuityAction.describe(
        {"window_k": 3, "descriptors": ["jerk", "dir"]}
    )
    assert out == {"f1a_a_jerk": "risky", "f1a_a_dir": "safe"}


def test_runtime_continuity_describe_state():
    out = RuntimeContinuityState.describe(
        {"window_k": 5, "descriptors": ["jerk"]}
    )
    # State factor uses key namespace "t" — matches the registered name
    # `f1a_t` so YAML configs that reference the registered factor name
    # find descriptor keys under the same namespace.
    assert out == {"f1a_t_jerk": "risky"}


def test_runtime_continuity_describe_state_multiple_descriptors():
    out = RuntimeContinuityState.describe(
        {"window_k": 4, "descriptors": ["jerk", "dir", "curv_radius"]}
    )
    assert out == {
        "f1a_t_jerk":        "risky",
        "f1a_t_dir":         "safe",
        "f1a_t_curv_radius": "non_monotonic",
    }


def test_runtime_continuity_key_initial_attributes():
    assert RuntimeContinuityAction.key_initial == "a"
    assert RuntimeContinuityState.key_initial == "t"


def test_runtime_continuity_describe_unknown_descriptor_raises():
    with pytest.raises(ValueError, match="Unknown F1a descriptor"):
        RuntimeContinuityAction.describe(
            {"window_k": 3, "descriptors": ["nonexistent"]}
        )


def test_source_window_describe_dict_form():
    out = SourceWindowSmoothnessAction.describe(
        {
            "windows": [{"past": 0, "future": 5}, {"past": 5, "future": 5}],
            "descriptors": ["jerk", "curv_radius"],
        }
    )
    assert out == {
        "f1b_a_jerk__p0_f5":         "risky",
        "f1b_a_jerk__p5_f5":         "risky",
        "f1b_a_curv_radius__p0_f5":  "non_monotonic",
        "f1b_a_curv_radius__p5_f5":  "non_monotonic",
    }


def test_source_window_describe_state_uses_t_namespace():
    out = SourceWindowSmoothnessState.describe(
        {
            "windows": [{"past": 0, "future": 5}, (5, 10)],
            "descriptors": ["jerk", "dir"],
        }
    )
    # State factor uses key namespace "t" — matches the registered name
    # `f1b_t` so YAML configs find descriptor keys under the same namespace.
    assert out == {
        "f1b_t_jerk__p0_f5":  "risky",
        "f1b_t_jerk__p5_f10": "risky",
        "f1b_t_dir__p0_f5":   "safe",
        "f1b_t_dir__p5_f10":  "safe",
    }


def test_source_window_key_initial_attributes():
    assert SourceWindowSmoothnessAction.key_initial == "a"
    assert SourceWindowSmoothnessState.key_initial == "t"


def test_source_window_describe_tuple_form_matches_dict_form():
    p_dict = SourceWindowSmoothnessAction.describe(
        {"windows": [{"past": 0, "future": 5}], "descriptors": ["dir"]}
    )
    p_tuple = SourceWindowSmoothnessAction.describe(
        {"windows": [(0, 5)], "descriptors": ["dir"]}
    )
    assert p_dict == p_tuple


def test_source_window_describe_unknown_descriptor_raises():
    with pytest.raises(ValueError, match="Unknown F1b descriptor"):
        SourceWindowSmoothnessAction.describe(
            {"windows": [(0, 5)], "descriptors": ["totally_made_up"]}
        )


def test_consensus_describe_constant():
    assert TopKActionConsensus.describe({"K": 1}) == {"f2_var": "risky"}
    assert TopKActionConsensus.describe({"K": 100}) == {"f2_var": "risky"}


# ------------------------------------------------------------------
# build() + instance descriptor_orientations agree with describe()
# ------------------------------------------------------------------


def test_consensus_instance_orientations_match_describe():
    inst = registry.build("f2", K=5)
    assert inst.descriptor_orientations == TopKActionConsensus.describe({"K": 5})
    assert inst.required_top_k == 5


def test_consensus_invalid_K_raises():
    with pytest.raises(ValueError):
        TopKActionConsensus(K=0)


# ------------------------------------------------------------------
# Helper: _normalize_windows
# ------------------------------------------------------------------


def test_normalize_windows_dict():
    assert _normalize_windows([{"past": 0, "future": 5}]) == [(0, 5)]


def test_normalize_windows_tuple():
    assert _normalize_windows([(0, 5), (5, 10)]) == [(0, 5), (5, 10)]


def test_normalize_windows_mixed():
    out = _normalize_windows([{"past": 0, "future": 3}, (1, 2), [4, 6]])
    assert out == [(0, 3), (1, 2), (4, 6)]


# ------------------------------------------------------------------
# Descriptor orientation table
# ------------------------------------------------------------------


def test_descriptor_orientation_table():
    assert _DESCRIPTOR_ORIENTATIONS["jerk"] == "risky"
    assert _DESCRIPTOR_ORIENTATIONS["dir"] == "safe"
    assert _DESCRIPTOR_ORIENTATIONS["curv_radius"] == "non_monotonic"
    assert _DESCRIPTOR_ORIENTATIONS["cum_disp"] == "non_monotonic"
