"""Pluggable reducers for LLM layer-N prefix hidden states.

Receives `LLMLayerExtractResult` from the layer-N forward done inside
`CP1LLMLayerExtractKeyBuilder` and compresses the prefix-shaped hidden
state tensor into one or more fixed-dimension key vectors keyed by
`CACHE_QUERY_FIELDS` slot names.

Two-step KeyBuilder pattern (parallel to `cp1_temporal_prune` but for
post-LLM hidden states):
  Step A (extractor, in KeyBuilder):
      Stage1Output -> partial paligemma forward (layers[0..N]) ->
      LLMLayerExtractResult(hidden_states, pad_mask, segment_offsets, extract_layer)
  Step B (this module):
      LLMLayerExtractResult -> dict[str, GPU tensor] -> KeyBuilder handles CPU transfer

Reducers currently shipped:
  - PrefixMeanPoolReducer:        masked mean over the entire prefix -> {vision_0: 2048}
  - PerModalityMeanPoolReducer:       per-segment masked mean -> up to 4 fields, each [2048]
  - PerModalityMaxPoolReducer:    per-segment masked max  -> up to 4 fields, each [2048]
  - PerModalitySpatialPoolReducer: vision segments spatial-avg-pool on the
    16x16 SigLIP grid (output_tokens square), prompt_emb falls back to
    masked mean because the lang segment is variable-length and not square.
    Canonical output_tokens values (aligned with legacy cp1_spatial_pool_*):
    4 -> vision [8192] (2x2 pool), 16 -> vision [32768] (4x4 pool)

Coupling map:
  DEPENDS ON:  nothing (leaf module; only torch + stdlib)
  CONSUMED BY: CP1LLMLayerExtractKeyBuilder.build()
  IF CHANGED:  per-field output_dims must match backend.vector_dims declarations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F

from openpi.cache.types import PROMPT_EMB, VISION_0, VISION_1, VISION_2

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# Gemma 2B (PaliGemma backbone) hidden width. See models/gemma.py:81.
# Layer-N hidden states keep this width since each decoder layer is a
# residual block: in_dim == out_dim == hidden_size.
_GEMMA_2B_WIDTH = 2048

# Expected vision-segment length for spatial pool. SigLIP emits 256 tokens
# per camera (16x16 grid); anything else means the prefix layout drifted.
_VISION_GRID_TOKENS = 256


# ----------------------------------------------------------------------
# LLMLayerExtractResult: Step A output, Step B input
# ----------------------------------------------------------------------


@dataclass
class LLMLayerExtractResult:
    """Output of partial LLM forward (layers[0..extract_layer]).

    All tensors are GPU-resident. Reducers may compute in float32 internally
    but return GPU tensors; CPU transfer is the KeyBuilder's responsibility.

    Lifetime: constructed inside KeyBuilder.build(); consumed once by
    PrefixReducer.reduce(); then discarded along with KeyBuilder._cache on clear().
    """

    hidden_states: torch.Tensor
    """[L, hidden_dim] -- layer-N output for one inference (batch dim already dropped)."""

    pad_mask: torch.Tensor
    """[L] bool -- True at real (non-padding) prefix positions, False at padding."""

    segment_offsets: dict[str, tuple[int, int]]
    """Modality -> (start, end) half-open token range within hidden_states.

    Keys are a subset of {vision_0, vision_1, vision_2, prompt_emb}. After
    layer-N forward each position has been mixed by N+1 rounds of prefix-LM
    attention, so a slice is a contextualized representation of the original
    modality, not its raw token. Slot identity (the start/end offsets)
    remains valid for per-modality pooling.
    """

    extract_layer: int
    """Layer index whose output produced hidden_states (0-indexed). Carried
    through for traceability / artifact metadata."""


# ----------------------------------------------------------------------
# PrefixReducer Protocol
# ----------------------------------------------------------------------


@runtime_checkable
class PrefixReducer(Protocol):
    """Reduce LLM layer-N hidden states to one or more cache key vectors.

    All implementations must:
      - Honour `result.pad_mask` (the lang segment is right-padded; naive
        unmasked mean would dilute keys with padding-token embeddings).
      - Return GPU tensors keyed by a subset of CACHE_QUERY_FIELDS.
        CPU transfer is done by the calling KeyBuilder, not here.
      - Omit a field when its segment is entirely masked off (e.g. a missing
        camera) rather than emitting a zero vector. This matches CLIPKeyBuilder's
        contract (`clip_key_builder.py` "missing image slots are skipped").

    Coupling:
      DEPENDS ON:  LLMLayerExtractResult (same file)
      CONSUMED BY: CP1LLMLayerExtractKeyBuilder.build()
      IF CHANGED:  per-field output_dims must match backend.vector_dims;
                   emitted_fields must stay a subset of CACHE_QUERY_FIELDS
    """

    def reduce(self, result: LLMLayerExtractResult) -> dict[str, torch.Tensor]:
        """Return {field_name: [output_dims[field_name]] GPU tensor}."""
        ...

    @property
    def output_dims(self) -> dict[str, int]:
        """Field name -> dim. Used for backend.vector_dims cross-validation."""
        ...

    @property
    def emitted_fields(self) -> frozenset[str]:
        """Subset of CACHE_QUERY_FIELDS this reducer can emit."""
        ...


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[L, D] + [L] bool -> [D].

    Cast hidden to float32 before summation: layer-N outputs are bf16 and
    naive bf16 sum across ~800 tokens accumulates noticeable rounding.

    The caller must guarantee mask.any() is True; segment-level emit
    decisions live in the reducer, not here.
    """
    h = hidden.float()
    m = mask.to(dtype=h.dtype).unsqueeze(-1)  # [L, 1]
    total = (h * m).sum(dim=0)
    count = m.sum().clamp(min=1.0)
    return total / count


def _masked_max(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[L, D] + [L] bool -> [D].

    Per-dim max over unmasked positions. Padding positions are replaced by
    -inf so they can never win the max; float32 promotion mirrors
    `_masked_mean` to keep numerical behaviour consistent.

    Caller must guarantee `mask.any()` is True.
    """
    h = hidden.float()
    neg_inf = torch.finfo(h.dtype).min
    filled = h.masked_fill(~mask.unsqueeze(-1), neg_inf)
    return filled.max(dim=0).values


def _spatial_avg_pool_vision(hidden: torch.Tensor, pool_size: int) -> torch.Tensor:
    """[256, D] -> [pool_size*pool_size * D].

    Interpret the 256 SigLIP vision tokens as a 16x16 spatial grid (row-major
    exactly as the vision encoder emits them), run adaptive avg pool to
    `[pool_size, pool_size]`, then flatten.

    Vision segments have no intra-segment padding (a camera is either present
    in full or absent in full, decided by the outer `pad_mask.any()` check),
    so no masking is needed inside this helper.
    """
    h = hidden.float()
    D = h.shape[-1]
    grid_side = int(_VISION_GRID_TOKENS**0.5)   # 16
    # [256, D] -> [16, 16, D] -> [1, D, 16, 16] (NCHW for adaptive_avg_pool2d).
    grid = h.reshape(grid_side, grid_side, D).permute(2, 0, 1).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(grid, (pool_size, pool_size))
    # [1, D, pool, pool] -> [pool, pool, D] -> [pool*pool*D]
    return pooled.squeeze(0).permute(1, 2, 0).reshape(-1)


# ----------------------------------------------------------------------
# PrefixMeanPoolReducer
# ----------------------------------------------------------------------


class PrefixMeanPoolReducer:
    """Masked mean over the entire prefix -> single 2048-d key under vision_0.

    Embodies the professor's "drop multi-modal weighting; one fused key" idea:
    one prefix-LM attention round (run by the KeyBuilder before this reducer
    is invoked) has already fused vision + language into every token, so a
    single global pool collapses cleanly to one cache key.

    Output: `{vision_0: [2048]}`. The vision_0 slot is reused as the carrier
    field for the unified key — config validation enforces that no other
    vision/prompt field is enabled when this reducer is active.
    """

    @property
    def output_dims(self) -> dict[str, int]:
        return {VISION_0: _GEMMA_2B_WIDTH}

    @property
    def emitted_fields(self) -> frozenset[str]:
        return frozenset({VISION_0})

    def reduce(self, result: LLMLayerExtractResult) -> dict[str, torch.Tensor]:
        if not bool(result.pad_mask.any()):
            # No real tokens in the prefix at all (degenerate input).
            # Returning an empty dict signals "no key for this step"; the
            # KeyBuilder propagates this to the orchestrator as a forced miss.
            return {}
        return {VISION_0: _masked_mean(result.hidden_states, result.pad_mask)}


# ----------------------------------------------------------------------
# PerModalityMeanPoolReducer
# ----------------------------------------------------------------------


# Order matches CACHE_QUERY_FIELDS slot order; iterating in this order keeps
# the emitted dict deterministic across runs (dict insertion order is part
# of Python's contract since 3.7) which simplifies debugging and serialization.
_MODALITY_FIELDS = (VISION_0, VISION_1, VISION_2, PROMPT_EMB)


class PerModalityMeanPoolReducer:
    """Per-modality masked mean. Skips empty segments.

    Output: up to 4 fields, each `[2048]`. Each field is the masked mean of
    its segment slice of the layer-N hidden state. After one layer of
    prefix-LM attention the per-segment representations are contextualized
    by the other modalities, so this is not a pure single-modality embedding;
    rather it preserves modality grouping for downstream weight ablation.

    A field is omitted (not emitted as a zero vector) when its segment's
    pad_mask is entirely False — this matches the online behaviour where a
    masked-off camera (`img_mask=False`) contributes nothing to attention.
    """

    @property
    def output_dims(self) -> dict[str, int]:
        return {field: _GEMMA_2B_WIDTH for field in _MODALITY_FIELDS}

    @property
    def emitted_fields(self) -> frozenset[str]:
        return frozenset(_MODALITY_FIELDS)

    def reduce(self, result: LLMLayerExtractResult) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for field in _MODALITY_FIELDS:
            offsets = result.segment_offsets.get(field)
            if offsets is None:
                continue
            start, end = offsets
            seg_mask = result.pad_mask[start:end]
            if not bool(seg_mask.any()):
                # Entire segment masked off (e.g. absent camera).
                continue
            seg_hidden = result.hidden_states[start:end]
            out[field] = _masked_mean(seg_hidden, seg_mask)
        return out


# ----------------------------------------------------------------------
# PerModalityMaxPoolReducer
# ----------------------------------------------------------------------


_VISION_ONLY_FIELDS = (VISION_0, VISION_1, VISION_2)


class PerModalityMaxPoolReducer:
    """Per-modality masked max pool. Same emission semantics as
    `PerModalityMeanPoolReducer` (skip segments whose pad_mask is entirely
    False), but the aggregator is `_masked_max` instead of `_masked_mean`.

    Output: up to 4 fields, each `[2048]`. Per-dim max is less smooth than
    mean pool and more sensitive to outlier tokens, which is sometimes
    desirable when the key needs to latch onto salient activations.
    """

    @property
    def output_dims(self) -> dict[str, int]:
        return {field: _GEMMA_2B_WIDTH for field in _MODALITY_FIELDS}

    @property
    def emitted_fields(self) -> frozenset[str]:
        return frozenset(_MODALITY_FIELDS)

    def reduce(self, result: LLMLayerExtractResult) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for field in _MODALITY_FIELDS:
            offsets = result.segment_offsets.get(field)
            if offsets is None:
                continue
            start, end = offsets
            seg_mask = result.pad_mask[start:end]
            if not bool(seg_mask.any()):
                continue
            seg_hidden = result.hidden_states[start:end]
            out[field] = _masked_max(seg_hidden, seg_mask)
        return out


# ----------------------------------------------------------------------
# PerModalitySpatialPoolReducer
# ----------------------------------------------------------------------


class PerModalitySpatialPoolReducer:
    """Vision segments -> spatial avg pool on the 16x16 SigLIP grid; prompt
    segment -> masked mean (it is variable-length and not square-shaped).

    `output_tokens` must be a perfect square (e.g. 16 -> 4x4, 64 -> 8x8).
    Vision output dim = `output_tokens * 2048`; prompt output dim = 2048.

    Skip semantics match `PerModalityMeanPoolReducer`: a vision field whose
    segment pad_mask is entirely False is omitted (not zero-filled), matching
    the online "masked camera" behaviour. The vision-segment length must be
    exactly 256; any other length is treated as a prefix-layout drift and
    raises.
    """

    def __init__(self, output_tokens: int):
        if output_tokens < 1:
            raise ValueError(f"output_tokens must be >= 1, got {output_tokens}")
        pool_size = int(output_tokens**0.5)
        if pool_size**2 != output_tokens:
            raise ValueError(
                f"output_tokens must be a perfect square, got {output_tokens}"
            )
        self._output_tokens = output_tokens
        self._pool_size = pool_size

    @property
    def output_dims(self) -> dict[str, int]:
        vision_dim = self._output_tokens * _GEMMA_2B_WIDTH
        return {
            VISION_0:   vision_dim,
            VISION_1:   vision_dim,
            VISION_2:   vision_dim,
            PROMPT_EMB: _GEMMA_2B_WIDTH,
        }

    @property
    def emitted_fields(self) -> frozenset[str]:
        return frozenset(_MODALITY_FIELDS)

    def reduce(self, result: LLMLayerExtractResult) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}

        # Vision segments: spatial avg pool on the 16x16 grid.
        for field in _VISION_ONLY_FIELDS:
            offsets = result.segment_offsets.get(field)
            if offsets is None:
                continue
            start, end = offsets
            seg_mask = result.pad_mask[start:end]
            if not bool(seg_mask.any()):
                continue
            seg_len = end - start
            if seg_len != _VISION_GRID_TOKENS:
                raise ValueError(
                    f"Vision segment {field} has length {seg_len}, "
                    f"expected {_VISION_GRID_TOKENS} (16x16 SigLIP grid). "
                    f"prefix layout drift — cannot spatial pool."
                )
            out[field] = _spatial_avg_pool_vision(
                result.hidden_states[start:end], self._pool_size
            )

        # Prompt segment: fall back to masked mean (variable-length padding).
        offsets = result.segment_offsets.get(PROMPT_EMB)
        if offsets is not None:
            start, end = offsets
            seg_mask = result.pad_mask[start:end]
            if bool(seg_mask.any()):
                out[PROMPT_EMB] = _masked_mean(
                    result.hidden_states[start:end], seg_mask
                )

        return out
