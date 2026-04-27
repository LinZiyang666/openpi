"""B0 protocol-shape tests for Composers.

The algorithm bodies (`compose`) raise NotImplementedError in B0; only
the constructor signatures + bind_orientations + Protocol structural
typing are exercised here. Algorithm-level behavior is covered in B1.
"""

from __future__ import annotations

from openpi.cache.components.factors.composers import (
    AndGateComposer,
    Composer,
    OrGateComposer,
    WeightedSumComposer,
)


# ------------------------------------------------------------------
# Protocol conformance
# ------------------------------------------------------------------


def test_weighted_sum_implements_composer_protocol():
    c = WeightedSumComposer(weights={"a": 1.0}, full_hit_threshold=0.5)
    assert isinstance(c, Composer)


def test_and_gate_implements_composer_protocol():
    c = AndGateComposer(per_factor_thresholds={"a": 0.5})
    assert isinstance(c, Composer)


def test_or_gate_implements_composer_protocol():
    c = OrGateComposer(per_factor_thresholds={"a": 0.5})
    assert isinstance(c, Composer)


# ------------------------------------------------------------------
# bind_orientations stores orientations
# ------------------------------------------------------------------


def test_bind_orientations_stores():
    c = WeightedSumComposer(weights={"a": 1.0}, full_hit_threshold=0.5)
    orient = {"a": "safe", "b": "risky"}
    c.bind_orientations(orient)
    assert c._orientations == orient


# ------------------------------------------------------------------
# Constructor accepts optional fields
# ------------------------------------------------------------------


def test_weighted_sum_optional_fields():
    c = WeightedSumComposer(
        weights={"a": 1.0},
        full_hit_threshold=0.8,
        warm_start_threshold=0.5,
        warm_start_t=0.5,
        directions={"non_mono_key": "high"},
    )
    assert c._warm_start_threshold == 0.5
    assert c._warm_start_t == 0.5
    assert c._directions == {"non_mono_key": "high"}
