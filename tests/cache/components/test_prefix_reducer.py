"""Tests for prefix_reducer.py.

Covers:
  - PrefixReducer Protocol compliance (runtime_checkable isinstance)
  - LLMLayerExtractResult dataclass shape contract
  - PrefixMeanPoolReducer: output dim, output field, masked vs unmasked
    correctness, all-masked degenerate input
  - PerModalityMeanPoolReducer: per-segment fan-out, missing camera omission,
    deterministic emit order, all-masked-segment skip
"""

from __future__ import annotations

import pytest
import torch

from openpi.cache.components.prefix_reducer import (
    LLMLayerExtractResult,
    PerModalityMaxPoolReducer,
    PerModalityMeanPoolReducer,
    PerModalitySpatialPoolReducer,
    PrefixMeanPoolReducer,
    PrefixReducer,
    _masked_max,
    _masked_mean,
)
from openpi.cache.types import PROMPT_EMB, VISION_0, VISION_1, VISION_2

# Fixed Pi0.5 prefix layout shared across tests.
HIDDEN_DIM = 2048
PREFIX_LEN = 968
SEG_OFFSETS = {
    VISION_0: (0, 256),
    VISION_1: (256, 512),
    VISION_2: (512, 768),
    PROMPT_EMB: (768, 968),
}


def _make_result(
    *,
    hidden: torch.Tensor | None = None,
    pad_mask: torch.Tensor | None = None,
    segment_offsets: dict[str, tuple[int, int]] | None = None,
    extract_layer: int = 0,
) -> LLMLayerExtractResult:
    if hidden is None:
        hidden = torch.zeros(PREFIX_LEN, HIDDEN_DIM, dtype=torch.float32)
    if pad_mask is None:
        pad_mask = torch.ones(PREFIX_LEN, dtype=torch.bool)
    if segment_offsets is None:
        segment_offsets = dict(SEG_OFFSETS)
    return LLMLayerExtractResult(
        hidden_states=hidden,
        pad_mask=pad_mask,
        segment_offsets=segment_offsets,
        extract_layer=extract_layer,
    )


# ----------------------------------------------------------------------
# Protocol compliance
# ----------------------------------------------------------------------


def test_prefix_mean_pool_satisfies_protocol():
    assert isinstance(PrefixMeanPoolReducer(), PrefixReducer)


def test_per_modality_mean_pool_satisfies_protocol():
    assert isinstance(PerModalityMeanPoolReducer(), PrefixReducer)


def test_per_modality_max_pool_satisfies_protocol():
    assert isinstance(PerModalityMaxPoolReducer(), PrefixReducer)


def test_per_modality_spatial_pool_satisfies_protocol():
    assert isinstance(PerModalitySpatialPoolReducer(output_tokens=16), PrefixReducer)


# ----------------------------------------------------------------------
# Shape / dim contracts
# ----------------------------------------------------------------------


def test_prefix_mean_pool_output_dims():
    r = PrefixMeanPoolReducer()
    assert r.output_dims == {VISION_0: HIDDEN_DIM}
    assert r.emitted_fields == frozenset({VISION_0})


def test_per_modality_mean_pool_output_dims():
    r = PerModalityMeanPoolReducer()
    expected = {VISION_0: HIDDEN_DIM, VISION_1: HIDDEN_DIM,
                VISION_2: HIDDEN_DIM, PROMPT_EMB: HIDDEN_DIM}
    assert r.output_dims == expected
    assert r.emitted_fields == frozenset(expected.keys())


# ----------------------------------------------------------------------
# _masked_mean correctness
# ----------------------------------------------------------------------


def test_masked_mean_recovers_unmasked_mean_when_all_true():
    h = torch.randn(50, 16)
    m = torch.ones(50, dtype=torch.bool)
    out = _masked_mean(h, m)
    assert torch.allclose(out, h.mean(dim=0), rtol=1e-5)


def test_masked_mean_ignores_padding_positions():
    # Real positions = 1.0, padding = 999.0. Unmasked mean would be huge;
    # masked mean must recover 1.0 exactly.
    h = torch.full((100, 8), 999.0)
    h[:30] = 1.0
    m = torch.zeros(100, dtype=torch.bool)
    m[:30] = True
    out = _masked_mean(h, m)
    assert torch.allclose(out, torch.ones(8))


def test_masked_mean_bf16_input_promotes_to_float32():
    h = torch.randn(20, 4, dtype=torch.bfloat16)
    m = torch.ones(20, dtype=torch.bool)
    out = _masked_mean(h, m)
    assert out.dtype == torch.float32


# ----------------------------------------------------------------------
# PrefixMeanPoolReducer behaviour
# ----------------------------------------------------------------------


def test_prefix_mean_pool_basic():
    hidden = torch.randn(PREFIX_LEN, HIDDEN_DIM)
    r = PrefixMeanPoolReducer()
    out = r.reduce(_make_result(hidden=hidden))
    assert set(out.keys()) == {VISION_0}
    assert out[VISION_0].shape == (HIDDEN_DIM,)
    assert torch.allclose(out[VISION_0], hidden.mean(dim=0), rtol=1e-5)


def test_prefix_mean_pool_respects_pad_mask():
    # Real prefix positions [0:500) hold value 1.0; positions [500:968)
    # are padding holding 999.0. Masked mean must equal 1.0.
    hidden = torch.full((PREFIX_LEN, HIDDEN_DIM), 999.0)
    hidden[:500] = 1.0
    pad_mask = torch.zeros(PREFIX_LEN, dtype=torch.bool)
    pad_mask[:500] = True
    out = PrefixMeanPoolReducer().reduce(_make_result(hidden=hidden, pad_mask=pad_mask))
    assert torch.allclose(out[VISION_0], torch.ones(HIDDEN_DIM))


def test_prefix_mean_pool_returns_empty_when_pad_mask_all_false():
    pad_mask = torch.zeros(PREFIX_LEN, dtype=torch.bool)
    out = PrefixMeanPoolReducer().reduce(_make_result(pad_mask=pad_mask))
    assert out == {}


# ----------------------------------------------------------------------
# PerModalityMeanPoolReducer behaviour
# ----------------------------------------------------------------------


def test_per_modality_mean_pool_emits_all_four_segments():
    hidden = torch.randn(PREFIX_LEN, HIDDEN_DIM)
    out = PerModalityMeanPoolReducer().reduce(_make_result(hidden=hidden))
    assert set(out.keys()) == {VISION_0, VISION_1, VISION_2, PROMPT_EMB}
    for field, (start, end) in SEG_OFFSETS.items():
        assert torch.allclose(out[field], hidden[start:end].mean(dim=0), rtol=1e-5)


def test_per_modality_mean_pool_omits_segment_when_camera_missing():
    # vision_1 segment has all-False pad mask (camera absent).
    pad_mask = torch.ones(PREFIX_LEN, dtype=torch.bool)
    pad_mask[256:512] = False
    out = PerModalityMeanPoolReducer().reduce(_make_result(pad_mask=pad_mask))
    assert VISION_1 not in out
    assert {VISION_0, VISION_2, PROMPT_EMB} <= set(out.keys())


def test_per_modality_mean_pool_omits_segment_missing_from_offsets():
    # Caller did not declare vision_2 offsets at all.
    seg = {k: v for k, v in SEG_OFFSETS.items() if k != VISION_2}
    out = PerModalityMeanPoolReducer().reduce(_make_result(segment_offsets=seg))
    assert VISION_2 not in out


def test_per_modality_mean_pool_lang_padding_isolated_from_vision_pool():
    # Build hidden so vision tokens = 1.0, real lang tokens = 5.0,
    # lang padding = 999.0. Each modality pool must reflect its own segment only.
    hidden = torch.zeros(PREFIX_LEN, HIDDEN_DIM)
    hidden[0:768] = 1.0           # vision
    hidden[768:800] = 5.0          # real lang prompt
    hidden[800:968] = 999.0        # lang padding
    pad_mask = torch.ones(PREFIX_LEN, dtype=torch.bool)
    pad_mask[800:968] = False      # padding masked
    out = PerModalityMeanPoolReducer().reduce(_make_result(hidden=hidden, pad_mask=pad_mask))
    for vfield in (VISION_0, VISION_1, VISION_2):
        assert torch.allclose(out[vfield], torch.ones(HIDDEN_DIM))
    assert torch.allclose(out[PROMPT_EMB], torch.full((HIDDEN_DIM,), 5.0))


def test_per_modality_mean_pool_emit_order_is_deterministic():
    out = PerModalityMeanPoolReducer().reduce(_make_result())
    # Insertion order must match the canonical modality order so artifacts
    # and search results are deterministic across builds.
    assert list(out.keys()) == [VISION_0, VISION_1, VISION_2, PROMPT_EMB]


# ----------------------------------------------------------------------
# _masked_max correctness
# ----------------------------------------------------------------------


def test_masked_max_recovers_unmasked_max_when_all_true():
    h = torch.randn(50, 16)
    m = torch.ones(50, dtype=torch.bool)
    out = _masked_max(h, m)
    assert torch.allclose(out, h.max(dim=0).values, rtol=1e-5)


def test_masked_max_ignores_padding_positions():
    # Real positions hold small values, padding holds huge value; padding
    # must not win the per-dim max.
    h = torch.full((100, 8), 1e9)
    h[:30] = 1.0
    m = torch.zeros(100, dtype=torch.bool)
    m[:30] = True
    out = _masked_max(h, m)
    assert torch.allclose(out, torch.ones(8))


# ----------------------------------------------------------------------
# PerModalityMaxPoolReducer behaviour
# ----------------------------------------------------------------------


def test_per_modality_max_pool_output_dims():
    r = PerModalityMaxPoolReducer()
    expected = {VISION_0: HIDDEN_DIM, VISION_1: HIDDEN_DIM,
                VISION_2: HIDDEN_DIM, PROMPT_EMB: HIDDEN_DIM}
    assert r.output_dims == expected
    assert r.emitted_fields == frozenset(expected.keys())


def test_per_modality_max_pool_matches_segment_max():
    hidden = torch.randn(PREFIX_LEN, HIDDEN_DIM)
    out = PerModalityMaxPoolReducer().reduce(_make_result(hidden=hidden))
    assert set(out.keys()) == {VISION_0, VISION_1, VISION_2, PROMPT_EMB}
    for field, (start, end) in SEG_OFFSETS.items():
        expected = hidden[start:end].max(dim=0).values
        assert torch.allclose(out[field], expected, rtol=1e-5)


def test_per_modality_max_pool_omits_missing_camera():
    pad_mask = torch.ones(PREFIX_LEN, dtype=torch.bool)
    pad_mask[256:512] = False
    out = PerModalityMaxPoolReducer().reduce(_make_result(pad_mask=pad_mask))
    assert VISION_1 not in out
    assert {VISION_0, VISION_2, PROMPT_EMB} <= set(out.keys())


def test_per_modality_max_pool_respects_prompt_padding():
    # Real lang positions hold small values, padding positions hold huge.
    hidden = torch.full((PREFIX_LEN, HIDDEN_DIM), 1e9)
    hidden[:768] = 1.0       # vision
    hidden[768:800] = 2.0    # real lang
    pad_mask = torch.ones(PREFIX_LEN, dtype=torch.bool)
    pad_mask[800:968] = False
    out = PerModalityMaxPoolReducer().reduce(_make_result(hidden=hidden, pad_mask=pad_mask))
    assert torch.allclose(out[PROMPT_EMB], torch.full((HIDDEN_DIM,), 2.0))


# ----------------------------------------------------------------------
# PerModalitySpatialPoolReducer behaviour
# ----------------------------------------------------------------------


def test_spatial_pool_rejects_non_perfect_square():
    with pytest.raises(ValueError, match="perfect square"):
        PerModalitySpatialPoolReducer(output_tokens=10)


def test_spatial_pool_rejects_non_positive():
    with pytest.raises(ValueError, match=">= 1"):
        PerModalitySpatialPoolReducer(output_tokens=0)


def test_spatial_pool_16_output_dims():
    r = PerModalitySpatialPoolReducer(output_tokens=16)
    assert r.output_dims == {
        VISION_0: 16 * HIDDEN_DIM,
        VISION_1: 16 * HIDDEN_DIM,
        VISION_2: 16 * HIDDEN_DIM,
        PROMPT_EMB: HIDDEN_DIM,
    }
    assert r.emitted_fields == frozenset({VISION_0, VISION_1, VISION_2, PROMPT_EMB})


def test_spatial_pool_4_output_dims():
    r = PerModalitySpatialPoolReducer(output_tokens=4)
    assert r.output_dims == {
        VISION_0: 4 * HIDDEN_DIM,
        VISION_1: 4 * HIDDEN_DIM,
        VISION_2: 4 * HIDDEN_DIM,
        PROMPT_EMB: HIDDEN_DIM,
    }


def test_spatial_pool_vision_shape_and_prompt_fallback():
    # A uniform vision segment should produce a uniform pooled vector
    # (constant value preserved through avg pool); prompt falls back to mean.
    hidden = torch.zeros(PREFIX_LEN, HIDDEN_DIM)
    hidden[0:256]   = 3.0     # vision_0 = constant 3
    hidden[256:512] = 7.0     # vision_1 = constant 7
    hidden[512:768] = -2.0    # vision_2 = constant -2
    hidden[768:800] = 5.0     # real lang
    pad_mask = torch.ones(PREFIX_LEN, dtype=torch.bool)
    pad_mask[800:968] = False

    r = PerModalitySpatialPoolReducer(output_tokens=16)
    out = r.reduce(_make_result(hidden=hidden, pad_mask=pad_mask))
    assert out[VISION_0].shape == (16 * HIDDEN_DIM,)
    assert out[VISION_1].shape == (16 * HIDDEN_DIM,)
    assert out[VISION_2].shape == (16 * HIDDEN_DIM,)
    assert out[PROMPT_EMB].shape == (HIDDEN_DIM,)
    assert torch.allclose(out[VISION_0], torch.full((16 * HIDDEN_DIM,), 3.0))
    assert torch.allclose(out[VISION_1], torch.full((16 * HIDDEN_DIM,), 7.0))
    assert torch.allclose(out[VISION_2], torch.full((16 * HIDDEN_DIM,), -2.0))
    assert torch.allclose(out[PROMPT_EMB], torch.full((HIDDEN_DIM,), 5.0))


def test_spatial_pool_omits_missing_camera():
    pad_mask = torch.ones(PREFIX_LEN, dtype=torch.bool)
    pad_mask[256:512] = False
    r = PerModalitySpatialPoolReducer(output_tokens=16)
    out = r.reduce(_make_result(pad_mask=pad_mask))
    assert VISION_1 not in out
    assert {VISION_0, VISION_2, PROMPT_EMB} <= set(out.keys())


def test_spatial_pool_omits_prompt_when_all_padding():
    pad_mask = torch.ones(PREFIX_LEN, dtype=torch.bool)
    pad_mask[768:968] = False   # entire prompt segment is padding
    r = PerModalitySpatialPoolReducer(output_tokens=16)
    out = r.reduce(_make_result(pad_mask=pad_mask))
    assert PROMPT_EMB not in out


def test_spatial_pool_rejects_drifted_vision_segment_length():
    # Vision_0 declared as 200 tokens (not 256) — prefix layout drift.
    bad_offsets = dict(SEG_OFFSETS)
    bad_offsets[VISION_0] = (0, 200)
    r = PerModalitySpatialPoolReducer(output_tokens=16)
    with pytest.raises(ValueError, match="prefix layout drift"):
        r.reduce(_make_result(segment_offsets=bad_offsets))


def test_spatial_pool_emit_order_vision_then_prompt():
    r = PerModalitySpatialPoolReducer(output_tokens=16)
    out = r.reduce(_make_result())
    assert list(out.keys()) == [VISION_0, VISION_1, VISION_2, PROMPT_EMB]


# ----------------------------------------------------------------------
# LLMLayerExtractResult dataclass
# ----------------------------------------------------------------------


def test_extract_result_carries_layer_index():
    r = _make_result(extract_layer=5)
    assert r.extract_layer == 5


def test_extract_result_segment_offsets_independent_per_instance():
    r1 = _make_result()
    r2 = _make_result()
    r1.segment_offsets["foo"] = (0, 1)
    assert "foo" not in r2.segment_offsets


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
