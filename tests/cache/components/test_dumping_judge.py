"""Unit tests for DumpingJudge transparent wrapper.

Covers:
  - Equivalence across 3 inner judge types: AlwaysHit, Threshold, CompositeJudge
    (verdict output byte-identical with dump enabled vs disabled)
  - JSONL schema: 8-field row with correct types
  - Per-extractor exception isolation (failed extract -> NaN row, verdict OK)
  - Identity injection: both `task_id` and `orig_init_state_idx` carried through
    `on_episode_start(extra_metadata=...)` to JSONL row; reset on next episode
  - Lifecycle forwarding: inner CompositeJudge normalizer reset triggered;
    record_action forwards to inner
  - min_required_top_k merging: max(inner_hint, max(dump-extractor required_top_k))
  - __getattr__ fallback transparent + non-recursive on internal attrs
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from openpi.cache.components.factors.composers import WeightedSumComposer
from openpi.cache.components.factors.normalizers import PercentileRollingNormalizer
from openpi.cache.components.judge import (
    AlwaysHitJudge,
    CompositeJudge,
    DumpingJudge,
    HitType,
    JudgeResult,
    ThresholdJudge,
)
from openpi.cache.storage_types import SearchResultLite
from openpi.cache.types import CheckpointID


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _result(id: str = "e1", score: float = 0.99) -> SearchResultLite:
    return SearchResultLite(id=id, score=score, checkpoint_id=CheckpointID.CP1)


class _StubExtractor:
    """Minimal OnlineExtractor with controllable output."""

    def __init__(
        self,
        orientations: dict[str, str],
        output: dict[str, float],
        required_top_k: int = 0,
        raises: bool = False,
    ):
        self.descriptor_orientations = dict(orientations)
        self._output = dict(output)
        self.required_top_k = required_top_k
        self.requires_library_stats = False
        self.requires_chain_walk = False
        self._raises = raises

    @classmethod
    def describe(cls, params):
        return params.get("orientations", {})

    def extract(self, results, view, history, cached_data):
        if self._raises:
            raise RuntimeError("synthetic extractor failure")
        return dict(self._output)


# ------------------------------------------------------------------
# Equivalence: dump on/off must produce identical JudgeResult for 3 inner types
# ------------------------------------------------------------------


def _verdict_equivalent(inner, results, dump_path):
    """Wrapper verdict must equal naked inner verdict (byte-identical)."""
    naked = inner(results, CheckpointID.CP1, {})
    wrapper = DumpingJudge(inner, [], dump_path, "cfg_a")
    wrapped = wrapper(results, CheckpointID.CP1, {})
    assert wrapped.hit_type is naked.hit_type
    assert wrapped.winner_id == naked.winner_id
    assert wrapped.start_t == naked.start_t


def test_equivalence_always_hit(tmp_path):
    _verdict_equivalent(AlwaysHitJudge(), [_result("e1", 0.5)], tmp_path / "d.jsonl")


def test_equivalence_threshold(tmp_path):
    _verdict_equivalent(
        ThresholdJudge(cp1_threshold=0.9),
        [_result("e1", 0.95)],
        tmp_path / "d.jsonl",
    )


def test_equivalence_threshold_miss_path(tmp_path):
    _verdict_equivalent(
        ThresholdJudge(cp1_threshold=0.99),
        [_result("e1", 0.5)],  # below threshold -> MISS
        tmp_path / "d.jsonl",
    )


def test_equivalence_composite_judge(tmp_path):
    """Wrapping CompositeJudge must not break its decision pipeline."""
    ext = _StubExtractor({"k": "safe"}, {"k": 0.9})
    composer = WeightedSumComposer(
        weights={"k": 1.0}, full_hit_threshold=0.5,
    )
    inner = CompositeJudge(extractors=[ext], composer=composer, normalizer=None)
    _verdict_equivalent(inner, [_result("e1", 0.99)], tmp_path / "d.jsonl")


# ------------------------------------------------------------------
# JSONL schema
# ------------------------------------------------------------------


def _read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_jsonl_schema_has_eight_fields(tmp_path):
    dump_path = tmp_path / "d.jsonl"
    ext = _StubExtractor({"f": "safe"}, {"f": 0.7})
    wrapper = DumpingJudge(AlwaysHitJudge(), [ext], dump_path, "cfg_a")
    wrapper([_result("e1", 0.8)], CheckpointID.CP1, {})

    rows = _read_rows(dump_path)
    assert len(rows) == 1
    row = rows[0]
    expected = {"config_id", "task_id", "orig_init_state_idx", "step_idx",
                "winner_id", "cp1_score", "factor_raw", "factor_nan"}
    assert set(row.keys()) == expected
    assert row["config_id"] == "cfg_a"
    assert row["winner_id"] == "e1"
    assert row["cp1_score"] == pytest.approx(0.8)
    assert row["factor_raw"] == {"f": 0.7}
    assert row["factor_nan"] == {"f": False}
    assert row["step_idx"] == 0


# ------------------------------------------------------------------
# Per-extractor exception isolation
# ------------------------------------------------------------------


def test_failing_extractor_yields_nan_but_verdict_ok(tmp_path):
    dump_path = tmp_path / "d.jsonl"
    ok = _StubExtractor({"f1": "safe"}, {"f1": 0.5})
    bad = _StubExtractor({"f2": "risky"}, {"f2": 0.5}, raises=True)
    wrapper = DumpingJudge(AlwaysHitJudge(), [ok, bad], dump_path, "cfg_a")

    result = wrapper([_result("e1", 0.5)], CheckpointID.CP1, {})
    assert result.hit_type is HitType.FULL_HIT  # verdict not perturbed

    row = _read_rows(dump_path)[0]
    assert row["factor_raw"]["f1"] == pytest.approx(0.5)
    assert math.isnan(row["factor_raw"]["f2"])
    assert row["factor_nan"]["f1"] is False
    assert row["factor_nan"]["f2"] is True


def test_dump_path_write_failure_does_not_break_verdict(tmp_path):
    """Even if write itself fails (bad path), verdict must still return."""
    bad_path = tmp_path / "no_such_dir" / "d.jsonl"
    wrapper = DumpingJudge(AlwaysHitJudge(), [], bad_path, "cfg_a")
    result = wrapper([_result("e1", 0.5)], CheckpointID.CP1, {})
    assert result.hit_type is HitType.FULL_HIT


# ------------------------------------------------------------------
# Identity injection (two-field contract)
# ------------------------------------------------------------------


def test_identity_both_fields_carried_through(tmp_path):
    dump_path = tmp_path / "d.jsonl"
    wrapper = DumpingJudge(AlwaysHitJudge(), [], dump_path, "cfg_a")
    wrapper.on_episode_start(extra_metadata={"task_id": 3, "orig_init_state_idx": 7})
    wrapper([_result("e1", 0.5)], CheckpointID.CP1, {})

    row = _read_rows(dump_path)[0]
    assert row["task_id"] == 3 and isinstance(row["task_id"], int)
    assert row["orig_init_state_idx"] == 7 and isinstance(row["orig_init_state_idx"], int)


def test_identity_resets_between_episodes(tmp_path):
    dump_path = tmp_path / "d.jsonl"
    wrapper = DumpingJudge(AlwaysHitJudge(), [], dump_path, "cfg_a")
    wrapper.on_episode_start(extra_metadata={"task_id": 3, "orig_init_state_idx": 7})
    wrapper([_result("e1", 0.5)], CheckpointID.CP1, {})
    wrapper.on_episode_start(extra_metadata={})  # reset
    wrapper([_result("e2", 0.5)], CheckpointID.CP1, {})

    rows = _read_rows(dump_path)
    assert rows[0]["task_id"] == 3
    assert rows[0]["orig_init_state_idx"] == 7
    assert rows[1]["task_id"] is None
    assert rows[1]["orig_init_state_idx"] is None


def test_identity_partial_field_does_not_raise(tmp_path):
    dump_path = tmp_path / "d.jsonl"
    wrapper = DumpingJudge(AlwaysHitJudge(), [], dump_path, "cfg_a")
    wrapper.on_episode_start(extra_metadata={"task_id": 5})
    wrapper([_result("e1", 0.5)], CheckpointID.CP1, {})

    row = _read_rows(dump_path)[0]
    assert row["task_id"] == 5
    assert row["orig_init_state_idx"] is None


def test_step_idx_resets_between_episodes(tmp_path):
    dump_path = tmp_path / "d.jsonl"
    wrapper = DumpingJudge(AlwaysHitJudge(), [], dump_path, "cfg_a")
    wrapper.on_episode_start()
    wrapper([_result("e1", 0.5)], CheckpointID.CP1, {})
    wrapper([_result("e2", 0.5)], CheckpointID.CP1, {})
    wrapper.on_episode_start()
    wrapper([_result("e3", 0.5)], CheckpointID.CP1, {})

    rows = _read_rows(dump_path)
    assert [r["step_idx"] for r in rows] == [0, 1, 0]


# ------------------------------------------------------------------
# Lifecycle forwarding
# ------------------------------------------------------------------


def test_on_episode_start_forwards_to_composite_normalizer(tmp_path):
    """Inner CompositeJudge's normalizer.on_episode_start must still fire."""
    norm = PercentileRollingNormalizer(window_size=200)
    norm_spy = MagicMock(wraps=norm)
    norm_spy.bind_keys = norm.bind_keys
    norm_spy.on_episode_start = MagicMock(side_effect=norm.on_episode_start)

    ext = _StubExtractor({"k": "safe"}, {"k": 0.9})
    composer = WeightedSumComposer(weights={"k": 1.0}, full_hit_threshold=0.5)
    inner = CompositeJudge(extractors=[ext], composer=composer, normalizer=norm_spy)
    wrapper = DumpingJudge(inner, [], tmp_path / "d.jsonl", "cfg_a")

    wrapper.on_episode_start(extra_metadata={"task_id": 1, "orig_init_state_idx": 0})
    norm_spy.on_episode_start.assert_called_once()


def test_record_action_forwards_to_inner():
    inner = MagicMock(spec=AlwaysHitJudge())
    wrapper = DumpingJudge(inner, [], "/tmp/unused.jsonl", "cfg_a")
    chunk = torch.zeros(10, 32)
    wrapper.record_action(chunk)
    inner.record_action.assert_called_once_with(chunk)


def test_on_episode_start_filters_inner_kwargs():
    """Inner judge without extra_metadata kwarg must not raise."""
    # AlwaysHitJudge.on_episode_start(self) takes no kwargs
    wrapper = DumpingJudge(AlwaysHitJudge(), [], "/tmp/unused.jsonl", "cfg_a")
    # Must not raise even though inner doesn't accept extra_metadata
    wrapper.on_episode_start(extra_metadata={"task_id": 1, "orig_init_state_idx": 0})


# ------------------------------------------------------------------
# min_required_top_k merging
# ------------------------------------------------------------------


def test_min_top_k_dump_only(tmp_path):
    """Inner=AlwaysHit (no attr) + dump F2(K=5) -> 5."""
    ext = _StubExtractor({"k": "risky"}, {"k": 0.5}, required_top_k=5)
    wrapper = DumpingJudge(AlwaysHitJudge(), [ext], tmp_path / "d.jsonl", "cfg_a")
    assert wrapper.min_required_top_k == 5


def test_min_top_k_inner_only(tmp_path):
    """Inner=CompositeJudge with hint=3 + no dump factors -> 3."""
    ext = _StubExtractor({"k": "safe"}, {"k": 0.5}, required_top_k=3)
    composer = WeightedSumComposer(weights={"k": 1.0}, full_hit_threshold=0.5)
    inner = CompositeJudge(extractors=[ext], composer=composer, normalizer=None)
    wrapper = DumpingJudge(inner, [], tmp_path / "d.jsonl", "cfg_a")
    assert wrapper.min_required_top_k == 3


def test_min_top_k_max_of_inner_and_dump(tmp_path):
    """Inner=CompositeJudge with K=3 + dump F2(K=5) -> max=5."""
    inner_ext = _StubExtractor({"k": "safe"}, {"k": 0.5}, required_top_k=3)
    composer = WeightedSumComposer(weights={"k": 1.0}, full_hit_threshold=0.5)
    inner = CompositeJudge(extractors=[inner_ext], composer=composer, normalizer=None)
    dump_ext = _StubExtractor({"d": "risky"}, {"d": 0.5}, required_top_k=5)
    wrapper = DumpingJudge(inner, [dump_ext], tmp_path / "d.jsonl", "cfg_a")
    assert wrapper.min_required_top_k == 5


def test_min_top_k_zero_when_neither_has_hint(tmp_path):
    """Inner=AlwaysHit (no attr) + no dump factors -> 0."""
    wrapper = DumpingJudge(AlwaysHitJudge(), [], tmp_path / "d.jsonl", "cfg_a")
    assert wrapper.min_required_top_k == 0


# ------------------------------------------------------------------
# __getattr__ fallback (transparent + non-recursive)
# ------------------------------------------------------------------


def test_getattr_passes_through_to_inner():
    class _CustomInner(AlwaysHitJudge):
        custom_attr = "marker"

    wrapper = DumpingJudge(_CustomInner(), [], "/tmp/unused.jsonl", "cfg_a")
    assert wrapper.custom_attr == "marker"


def test_getattr_does_not_recurse_on_internal_attrs():
    """Accessing _inner / _dump_extractors / etc. must not recurse via __getattr__."""
    inner = AlwaysHitJudge()
    wrapper = DumpingJudge(inner, [], "/tmp/unused.jsonl", "cfg_a")
    # If __getattr__ recursed on _inner, this would StackOverflow.
    assert wrapper._inner is inner
    assert wrapper._dump_extractors == []
    assert wrapper._dump_path == "/tmp/unused.jsonl"
    assert wrapper._config_id == "cfg_a"
    assert wrapper._current_extra == {}
