"""Instruction-span masked prompt pooling tests (_CP1BaseKeyBuilder knobs).

Covers: knob-off byte-identical regression, masked-mean value parity,
span slicing with an injected fake tokenizer, span self-check fallback,
collect() caching of prefix_pad_masks / tokenized_prompt, missing-mask hard
error, projection inner threading, and find_instruction_span edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from openpi.cache.components.key_builder import (
    CP1MaxPoolKeyBuilder,
    CP1MeanPoolKeyBuilder,
    find_instruction_span,
)
from openpi.cache.types import CheckpointID

_EMB = 8
_PROMPT_LEN = 10
_REAL = 6  # real prompt tokens; the trailing 4 are padding


class _FakeStage1:
    def __init__(self, prefix, state, mask=None):
        self.prefix_embs = prefix
        self.state = state
        self.prefix_pad_masks = mask


class _FakeTokenizer:
    """encode_fragment stub: the span marker is the id pair [99, 55]."""

    def encode_fragment(self, text):
        return [99, 55]


@pytest.fixture
def stage1():
    torch.manual_seed(0)
    prefix = torch.randn(1, 768 + _PROMPT_LEN, _EMB)
    state = torch.randn(1, 4)
    mask = torch.cat(
        [torch.ones(768), torch.ones(_REAL), torch.zeros(_PROMPT_LEN - _REAL)]
    ).bool()[None]
    return _FakeStage1(prefix, state, mask)


_FIELDS = ["vision_0", "prompt_emb", "robot_state"]


def test_knobs_off_is_byte_identical_legacy(stage1):
    kb = CP1MeanPoolKeyBuilder(enabled_fields=_FIELDS)
    kb.collect(CheckpointID.CP1, stage1=stage1)
    keys = kb.build(CheckpointID.CP1)
    expected = stage1.prefix_embs[0, 768:].mean(0).float()
    assert torch.equal(keys["prompt_emb"], expected)


def test_masked_mean_excludes_padding(stage1):
    kb = CP1MeanPoolKeyBuilder(enabled_fields=_FIELDS, prompt_masked_pool=True)
    kb.collect(CheckpointID.CP1, stage1=stage1)
    keys = kb.build(CheckpointID.CP1)
    expected = stage1.prefix_embs[0, 768:768 + _REAL].mean(0).float()
    assert torch.equal(keys["prompt_emb"], expected)
    # Vision reduction is untouched by the knob.
    assert torch.equal(
        keys["vision_0"], stage1.prefix_embs[0, :256].mean(0).float()
    )


def test_masked_max_pool_variant(stage1):
    kb = CP1MaxPoolKeyBuilder(enabled_fields=_FIELDS, prompt_masked_pool=True)
    kb.collect(CheckpointID.CP1, stage1=stage1)
    keys = kb.build(CheckpointID.CP1)
    expected = stage1.prefix_embs[0, 768:768 + _REAL].max(dim=0).values.float()
    assert torch.equal(keys["prompt_emb"], expected)


def test_instruction_span_slices_at_marker(stage1):
    ids = np.array([2, 10, 11, 99, 55, 12, 0, 0, 0, 0], dtype=np.int64)
    kb = CP1MeanPoolKeyBuilder(
        enabled_fields=_FIELDS,
        prompt_masked_pool=True,
        prompt_instruction_span=True,
        tokenizer_factory=_FakeTokenizer,
    )
    kb.collect(CheckpointID.CP1, stage1=stage1, tokenized_prompt=ids)
    keys = kb.build(CheckpointID.CP1)
    # Marker [99, 55] first occurs at index 3 => instruction tokens = first 3.
    expected = stage1.prefix_embs[0, 768:771].mean(0).float()
    assert torch.equal(keys["prompt_emb"], expected)


def test_span_falls_back_when_marker_absent(stage1, caplog):
    ids = np.array([2, 10, 11, 12, 13, 14, 0, 0, 0, 0], dtype=np.int64)
    kb = CP1MeanPoolKeyBuilder(
        enabled_fields=_FIELDS,
        prompt_masked_pool=True,
        prompt_instruction_span=True,
        tokenizer_factory=_FakeTokenizer,
    )
    kb.collect(CheckpointID.CP1, stage1=stage1, tokenized_prompt=ids)
    with caplog.at_level("WARNING"):
        keys = kb.build(CheckpointID.CP1)
    expected = stage1.prefix_embs[0, 768:768 + _REAL].mean(0).float()
    assert torch.equal(keys["prompt_emb"], expected)
    assert any("marker not found" in r.message for r in caplog.records)


def test_span_falls_back_when_ids_missing(stage1, caplog):
    kb = CP1MeanPoolKeyBuilder(
        enabled_fields=_FIELDS,
        prompt_masked_pool=True,
        prompt_instruction_span=True,
        tokenizer_factory=_FakeTokenizer,
    )
    kb.collect(CheckpointID.CP1, stage1=stage1)  # no tokenized_prompt kwarg
    with caplog.at_level("WARNING"):
        keys = kb.build(CheckpointID.CP1)
    expected = stage1.prefix_embs[0, 768:768 + _REAL].mean(0).float()
    assert torch.equal(keys["prompt_emb"], expected)
    assert any("no tokenized_prompt" in r.message for r in caplog.records)


def test_masked_pool_without_mask_raises(stage1):
    stage1.prefix_pad_masks = None
    kb = CP1MeanPoolKeyBuilder(enabled_fields=_FIELDS, prompt_masked_pool=True)
    kb.collect(CheckpointID.CP1, stage1=stage1)
    with pytest.raises(RuntimeError, match="prefix_pad_masks"):
        kb.build(CheckpointID.CP1)


def test_collect_caches_mask_and_ids(stage1):
    ids = np.arange(_PROMPT_LEN, dtype=np.int64)
    kb = CP1MeanPoolKeyBuilder(enabled_fields=_FIELDS, prompt_masked_pool=True)
    kb.collect(CheckpointID.CP1, stage1=stage1, tokenized_prompt=ids)
    assert "prefix_pad_masks" in kb.cached_data
    assert kb._tokenized_prompt is ids
    # A collect WITHOUT the kwarg REPLACES the ids with None — stale ids from
    # a previous cycle must never cut a new cycle's span (G2 R1 finding 1).
    kb.collect(CheckpointID.CP1, stage1=stage1)
    assert kb._tokenized_prompt is None
    # clear() also drops them.
    kb.collect(CheckpointID.CP1, stage1=stage1, tokenized_prompt=ids)
    kb.clear()
    assert kb._tokenized_prompt is None


def test_stale_ids_never_leak_into_new_cycle_span(stage1, caplog):
    """Marker-bearing step, then clear + a CP1 collect with no ids: the new
    cycle must pool masked-only (fallback + WARN), not reuse the stale span."""
    marker_ids = np.array([2, 10, 11, 99, 55, 12, 0, 0, 0, 0], dtype=np.int64)
    kb = CP1MeanPoolKeyBuilder(
        enabled_fields=_FIELDS,
        prompt_masked_pool=True,
        prompt_instruction_span=True,
        tokenizer_factory=_FakeTokenizer,
    )
    kb.collect(CheckpointID.CP1, stage1=stage1, tokenized_prompt=marker_ids)
    span_key = kb.build(CheckpointID.CP1)["prompt_emb"]
    kb.clear()

    kb.collect(CheckpointID.CP1, stage1=stage1)  # new cycle, ids missing
    with caplog.at_level("WARNING"):
        keys = kb.build(CheckpointID.CP1)
    masked_only = stage1.prefix_embs[0, 768:768 + _REAL].mean(0).float()
    assert torch.equal(keys["prompt_emb"], masked_only)
    assert not torch.equal(keys["prompt_emb"], span_key)
    assert any("no tokenized_prompt" in r.message for r in caplog.records)


def test_projection_inner_threads_knobs(stage1):
    from openpi.cache.components.projection_key_builder import ProjectionKeyBuilder

    inner = CP1MeanPoolKeyBuilder(enabled_fields=_FIELDS, prompt_masked_pool=True)
    kb = ProjectionKeyBuilder(inner, None)  # identity projection
    kb.collect(CheckpointID.CP1, stage1=stage1)
    keys = kb.build(CheckpointID.CP1)
    expected = stage1.prefix_embs[0, 768:768 + _REAL].mean(0).float()
    assert torch.equal(keys["prompt_emb"], expected)


def test_find_instruction_span_edges():
    marker = np.array([99, 55])
    assert find_instruction_span(np.array([99, 55, 1]), marker) == 0
    assert find_instruction_span(np.array([1, 2, 99, 55]), marker) == 2
    assert find_instruction_span(np.array([1, 2, 3]), marker) is None
    assert find_instruction_span(np.array([99]), marker) is None
    assert find_instruction_span(np.array([], dtype=np.int64), marker) is None
    # 2-D [1, L] input is flattened by the canonical reshape(-1).
    assert find_instruction_span(np.array([[1, 99, 55]]), marker) == 1
