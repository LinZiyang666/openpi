"""Tests for cp1_llm_layer_extract KeyBuilder.

Covers:
  - Constructor argument validation (extract_layer, apply_final_norm)
  - attach_model: layer ref grab, no shared-config mutation, idempotence,
    out-of-range and model-mismatch errors
  - build() preconditions (must call attach_model and collect first)
  - collect / build / clear full cycle with mock paligemma layers
  - Output contract: dtype=float32, device=cpu, contiguous, expected fields
  - extract_layer=0 vs extract_layer=2 produce different outputs
  - robot_state passthrough gated by enabled_fields
  - PerModalityMeanPoolReducer wiring (multi-field output)
  - QueryKeyBuilder runtime_checkable protocol compliance

Note: Interceptor → attach_model auto-hook tests live with Phase 5
(test_llm_layer_key_builder_interceptor.py) and require a real Policy.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from openpi.cache.components.key_builder import QueryKeyBuilder
from openpi.cache.components.llm_layer_key_builder import (
    CP1LLMLayerExtractKeyBuilder,
)
from openpi.cache.components.prefix_reducer import (
    PerModalityMeanPoolReducer,
    PrefixMeanPoolReducer,
)
from openpi.cache.types import (
    PROMPT_EMB,
    ROBOT_STATE,
    VISION_0,
    VISION_1,
    VISION_2,
    CheckpointID,
)

# Test dimensions: tiny enough for fast tests but valid Pi0.5 layout.
HIDDEN = 16
PREFIX_LEN = 968       # Pi0.5: 3*256 + max_token_len(200)
STATE_DIM = 32
DEPTH = 4              # mock model depth (real Pi0.5 uses 18)


# ----------------------------------------------------------------------
# Mock model + layers
# ----------------------------------------------------------------------


class _MockLayer(nn.Module):
    """Minimal stand-in for `GemmaDecoderLayer`.

    Exposes the attribute path `_cast_to_layer_dtype` walks
    (`self_attn.q_proj.weight.dtype`) and a layer-index-dependent linear
    so different `extract_layer` values produce distinct outputs.
    """

    def __init__(self, hidden_size: int, layer_idx: int):
        super().__init__()
        # _cast_to_layer_dtype probes layer.self_attn.q_proj.weight.dtype.
        self.self_attn = SimpleNamespace(q_proj=nn.Linear(1, 1, bias=False))
        self._proj = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            # Identity + small layer-dependent perturbation so each layer
            # produces a different output for the same input.
            w = torch.eye(hidden_size) + (layer_idx + 1) * 0.01
            self._proj.weight.copy_(w)

    def forward(self, hidden_states, **kwargs):
        # Mock layers ignore attention_mask, position_embeddings, etc — we
        # only need the call signature to match what the KeyBuilder passes.
        return (self._proj(hidden_states),)


class _MockRotary(nn.Module):
    def forward(self, hidden, pos_ids):
        bs, length = pos_ids.shape
        half = hidden.shape[-1] // 2
        return torch.zeros(bs, length, half), torch.zeros(bs, length, half)


def _make_mock_model(hidden_size: int = HIDDEN, depth: int = DEPTH):
    config = SimpleNamespace(_attn_implementation="sdpa")
    layers = nn.ModuleList([_MockLayer(hidden_size, i) for i in range(depth)])
    return SimpleNamespace(
        paligemma_with_expert=SimpleNamespace(
            paligemma=SimpleNamespace(
                language_model=SimpleNamespace(
                    layers=layers,
                    rotary_emb=_MockRotary(),
                    config=config,
                )
            )
        )
    )


class _FakeStage1:
    def __init__(self, prefix_embs, prefix_pad_masks, prefix_att_2d_masks_4d,
                 prefix_position_ids, state):
        self.prefix_embs = prefix_embs
        self.prefix_pad_masks = prefix_pad_masks
        self.prefix_att_2d_masks_4d = prefix_att_2d_masks_4d
        self.prefix_position_ids = prefix_position_ids
        self.state = state


def _make_fake_stage1(*, prefix_len: int = PREFIX_LEN,
                       hidden_size: int = HIDDEN,
                       pad_mask: torch.Tensor | None = None,
                       seed: int = 0) -> _FakeStage1:
    g = torch.Generator().manual_seed(seed)
    if pad_mask is None:
        pad_mask = torch.ones(1, prefix_len, dtype=torch.bool)
    return _FakeStage1(
        prefix_embs=torch.randn(1, prefix_len, hidden_size, generator=g),
        prefix_pad_masks=pad_mask,
        prefix_att_2d_masks_4d=torch.zeros(1, 1, prefix_len, prefix_len),
        prefix_position_ids=torch.arange(prefix_len).unsqueeze(0),
        state=torch.randn(1, STATE_DIM, generator=g),
    )


def _make_builder(extract_layer: int = 0,
                  reducer=None,
                  enabled_fields: list[str] | None = None,
                  attach: bool = True,
                  depth: int = DEPTH,
                  hidden_size: int = HIDDEN):
    if reducer is None:
        reducer = PrefixMeanPoolReducer()
    builder = CP1LLMLayerExtractKeyBuilder(
        reducer=reducer,
        extract_layer=extract_layer,
        enabled_fields=enabled_fields,
    )
    if attach:
        builder.attach_model(_make_mock_model(hidden_size, depth))
    return builder


# ----------------------------------------------------------------------
# Constructor validation
# ----------------------------------------------------------------------


def test_extract_layer_negative_raises():
    with pytest.raises(ValueError, match="must be >= 0"):
        CP1LLMLayerExtractKeyBuilder(reducer=PrefixMeanPoolReducer(), extract_layer=-1)


def test_apply_final_norm_true_raises():
    with pytest.raises(NotImplementedError, match="apply_final_norm"):
        CP1LLMLayerExtractKeyBuilder(reducer=PrefixMeanPoolReducer(), apply_final_norm=True)


# ----------------------------------------------------------------------
# attach_model behaviour
# ----------------------------------------------------------------------


def test_attach_model_grabs_layer_refs():
    builder = _make_builder(attach=False)
    model = _make_mock_model()
    builder.attach_model(model)
    expected = model.paligemma_with_expert.paligemma.language_model.layers
    assert builder._layers is expected
    assert builder._depth == DEPTH


def test_attach_model_does_not_mutate_shared_attn_impl():
    model = _make_mock_model()
    cfg = model.paligemma_with_expert.paligemma.language_model.config
    assert cfg._attn_implementation == "sdpa"
    _make_builder(attach=False).attach_model(model)
    # attach_model must NOT mutate the shared model's attention backend: doing
    # so would race concurrent Stage 2 forwards and silently revert them to
    # eager. The layer-N replay runs under the model's existing backend.
    assert cfg._attn_implementation == "sdpa"


def test_attach_model_extract_layer_out_of_range_raises():
    builder = CP1LLMLayerExtractKeyBuilder(
        reducer=PrefixMeanPoolReducer(), extract_layer=DEPTH,
    )
    with pytest.raises(ValueError, match="out of range"):
        builder.attach_model(_make_mock_model(depth=DEPTH))


def test_attach_model_re_attach_same_model_is_noop():
    builder = _make_builder(attach=False)
    model = _make_mock_model()
    builder.attach_model(model)
    builder.attach_model(model)  # idempotent — must not raise
    assert builder._layers is model.paligemma_with_expert.paligemma.language_model.layers


def test_attach_model_different_model_raises():
    builder = _make_builder(attach=False)
    builder.attach_model(_make_mock_model())
    with pytest.raises(RuntimeError, match="different model"):
        builder.attach_model(_make_mock_model())


# ----------------------------------------------------------------------
# build() preconditions
# ----------------------------------------------------------------------


def test_build_without_attach_raises():
    builder = _make_builder(attach=False)
    builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1())
    with pytest.raises(RuntimeError, match="attach_model not called"):
        builder.build(CheckpointID.CP1)


def test_build_without_collect_raises():
    builder = _make_builder()
    with pytest.raises(RuntimeError, match="collect"):
        builder.build(CheckpointID.CP1)


def test_build_unsupported_checkpoint_raises():
    builder = _make_builder()
    builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1())
    with pytest.raises(ValueError, match="Unsupported checkpoint_id"):
        builder.build(CheckpointID.CP2)


# ----------------------------------------------------------------------
# Full collect / build / clear cycle
# ----------------------------------------------------------------------


def test_build_emits_reducer_fields_plus_robot_state():
    builder = _make_builder(reducer=PrefixMeanPoolReducer())
    builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1())
    keys = builder.build(CheckpointID.CP1)
    assert set(keys.keys()) == {VISION_0, ROBOT_STATE}


def test_build_per_modality_emits_four_vision_plus_robot_state():
    builder = _make_builder(reducer=PerModalityMeanPoolReducer())
    builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1())
    keys = builder.build(CheckpointID.CP1)
    assert set(keys.keys()) == {VISION_0, VISION_1, VISION_2, PROMPT_EMB, ROBOT_STATE}


def test_build_robot_state_disabled_when_not_in_enabled_fields():
    builder = _make_builder(enabled_fields=[VISION_0])
    builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1())
    keys = builder.build(CheckpointID.CP1)
    assert ROBOT_STATE not in keys


def test_build_robot_state_enabled_when_explicit():
    builder = _make_builder(enabled_fields=[VISION_0, ROBOT_STATE])
    builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1())
    keys = builder.build(CheckpointID.CP1)
    assert ROBOT_STATE in keys


def test_build_outputs_are_cpu_float32_contiguous():
    builder = _make_builder()
    builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1())
    keys = builder.build(CheckpointID.CP1)
    for field, tensor in keys.items():
        assert tensor.dtype == torch.float32, field
        assert tensor.device.type == "cpu", field
        assert tensor.is_contiguous(), field


def test_build_robot_state_value_is_raw_state():
    builder = _make_builder()
    stage1 = _make_fake_stage1()
    builder.collect(CheckpointID.CP1, stage1=stage1)
    keys = builder.build(CheckpointID.CP1)
    assert torch.allclose(keys[ROBOT_STATE], stage1.state[0].cpu().float())


def test_extract_layer_changes_output():
    """Two builders with extract_layer=0 vs =2 should give different keys."""
    stage1 = _make_fake_stage1()
    b0 = _make_builder(extract_layer=0)
    b2 = _make_builder(extract_layer=2)
    b0.collect(CheckpointID.CP1, stage1=stage1)
    b2.collect(CheckpointID.CP1, stage1=stage1)
    keys0 = b0.build(CheckpointID.CP1)
    keys2 = b2.build(CheckpointID.CP1)
    assert keys0[VISION_0].shape == keys2[VISION_0].shape
    assert not torch.allclose(keys0[VISION_0], keys2[VISION_0])


def test_clear_releases_cache_but_keeps_model_refs():
    builder = _make_builder()
    builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1())
    assert builder.cached_data
    builder.clear()
    assert builder.cached_data == {}
    # Model refs survive clear, so the next collect/build cycle works.
    builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1())
    builder.build(CheckpointID.CP1)


def test_collect_overwrites_previous_call():
    builder = _make_builder()
    builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1(seed=1))
    first_state = builder.cached_data["state"]
    builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1(seed=2))
    second_state = builder.cached_data["state"]
    assert not torch.equal(first_state, second_state)


def test_cp3_works_same_as_cp1():
    """CP3 reuses CP1's key path; behaviour is symmetric."""
    builder = _make_builder()
    stage1 = _make_fake_stage1()
    builder.collect(CheckpointID.CP1, stage1=stage1)
    keys_cp1 = builder.build(CheckpointID.CP1)
    builder.clear()
    builder.collect(CheckpointID.CP1, stage1=stage1)
    keys_cp3 = builder.build(CheckpointID.CP3)
    for f in keys_cp1:
        assert torch.allclose(keys_cp1[f], keys_cp3[f])


# ----------------------------------------------------------------------
# Pad mask propagation through the layer forward
# ----------------------------------------------------------------------


def test_padding_positions_excluded_from_mean_pool():
    """Padded positions must not influence the masked mean output."""
    # First half of prefix is real (random); second half is padding (huge values).
    pad_mask = torch.zeros(1, PREFIX_LEN, dtype=torch.bool)
    pad_mask[:, :500] = True
    s1 = _make_fake_stage1(pad_mask=pad_mask)
    # Stamp huge values into padded positions so a naive (unmasked) mean
    # would be visibly polluted.
    s1.prefix_embs[0, 500:] = 1e3

    builder = _make_builder()
    builder.collect(CheckpointID.CP1, stage1=s1)
    keys = builder.build(CheckpointID.CP1)
    # Mock layer is identity-ish (eye + 0.01); after layer 0 the magnitude
    # at masked positions is still huge. Masked mean should stay bounded.
    assert keys[VISION_0].abs().max() < 100


# ----------------------------------------------------------------------
# Protocol compliance
# ----------------------------------------------------------------------


def test_satisfies_query_key_builder_protocol():
    builder = _make_builder()
    assert isinstance(builder, QueryKeyBuilder)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
