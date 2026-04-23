"""CP1 KeyBuilder that runs partial PaliGemma forward inside the cache layer.

Two-step pipeline (parallel to `cp1_temporal_prune`):
  Step A (this module): borrow a slice of the live PaliGemma backbone
    (`layers[0..extract_layer]` + `rotary_emb`) and replay the prefill of
    `extract_layer + 1` layers on `Stage1Output.prefix_embs`. No KV cache,
    no gradient. Produces an `LLMLayerExtractResult`.
  Step B (prefix_reducer.py): reduce the layer-N prefix hidden state to
    one or more cache key vectors keyed by CACHE_QUERY_FIELDS slots.

Stage 2 is *not* modified or skipped — this builder spends N+1 layers worth
of FLOPs to compute a richer cache key, and the regular Stage 2 runs
afterwards on a cache miss exactly as before.

Coupling map:
  DEPENDS ON:  PI0Pytorch.paligemma_with_expert.paligemma.language_model
                 (specifically: layers, rotary_emb, config),
               Stage1Output fields (prefix_embs, prefix_pad_masks,
                 prefix_att_2d_masks_4d, prefix_position_ids, state),
               PrefixReducer protocol (prefix_reducer.py),
               key_builder._VISION_OFFSETS / _PROMPT_START
                 (Pi0.5 prefix layout constants — single source of truth).
  CONSUMED BY: CacheOrchestrator.check() via QueryKeyBuilder protocol.
               InferenceInterceptor.__init__ calls attach_model() once.
  IF CHANGED:  backend.vector_dims must follow reducer.output_dims;
               Pi0.5 prefix layout change requires updating
               key_builder._VISION_OFFSETS (shared) and the prompt segment
               end calculation here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from openpi.cache.components.key_builder import _PROMPT_START, _VISION_OFFSETS
from openpi.cache.components.prefix_reducer import (
    LLMLayerExtractResult,
    PrefixReducer,
)
from openpi.cache.types import PROMPT_EMB, ROBOT_STATE, CheckpointID

if TYPE_CHECKING:
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

logger = logging.getLogger(__name__)


# Layer-0 cast in HF Gemma forward: hidden_states promoted to bf16 if the
# layer's q_proj weight is bf16. We mirror that exact rule so partial replay
# matches Stage 2 numerics.
def _cast_to_layer_dtype(hidden: torch.Tensor, layer: torch.nn.Module) -> torch.Tensor:
    target_dtype = layer.self_attn.q_proj.weight.dtype
    if target_dtype == torch.bfloat16 and hidden.dtype != torch.bfloat16:
        return hidden.to(torch.bfloat16)
    return hidden


class CP1LLMLayerExtractKeyBuilder:
    """Two-step CP1 KeyBuilder: layer-N partial forward + pluggable reduction.

    Stateless across episodes (no per-step history). Construction creates the
    builder with a reducer; `attach_model()` must be called exactly once
    (by the InferenceInterceptor) before the first `build()`.

    Online flow per inference cycle:
      collect(CP1, stage1=...)        → cache GPU references
      build(CP1)                      → partial forward → reducer → CPU keys
      clear()                         → release per-cycle cache
    CP3 reuses the same key as CP1 (current orchestrator design); behaviour
    is symmetric.

    Offline flow (artifact build):
      Same triplet of calls, but `stage1` is reconstructed by
      `build_in_memory_cache_artifact._build_fake_stage1_with_masks()` from
      HDF5 fields. The model reference is attached once by the build script.
    """

    def __init__(
        self,
        reducer: PrefixReducer,
        extract_layer: int = 0,
        enabled_fields: list[str] | None = None,
        apply_final_norm: bool = False,
    ) -> None:
        if extract_layer < 0:
            raise ValueError(f"extract_layer must be >= 0, got {extract_layer}")
        if apply_final_norm:
            # Reserved for a future revision; only meaningful at extract_layer
            # == depth-1 anyway. Reject loudly so users do not get silent noop.
            raise NotImplementedError(
                "apply_final_norm=True is reserved; current version always "
                "returns the raw layer output without final RMSNorm."
            )
        self._reducer = reducer
        self._extract_layer = extract_layer
        self._enabled = set(enabled_fields) if enabled_fields is not None else None
        self._apply_final_norm = apply_final_norm

        # Borrowed model references, set by attach_model().
        self._layers: torch.nn.Module | None = None
        self._rotary_emb: torch.nn.Module | None = None
        self._depth: int | None = None

        # Per-cycle cache.
        self._cache: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Model attachment
    # ------------------------------------------------------------------

    def attach_model(self, model: "PI0Pytorch") -> None:
        """Borrow PaliGemma `layers` and `rotary_emb` references.

        Idempotent: re-attaching the same model is a no-op; attaching a
        different model raises. Called once by InferenceInterceptor.__init__.

        Side effect: forces `language_model.config._attn_implementation = "eager"`
        to match `_stage2_llm_backbone()` (which sets the same flag at every
        Stage 2 call). This guarantees layer-0 attention numerics here are
        bit-identical to what Stage 2 produces.
        """
        language_model = model.paligemma_with_expert.paligemma.language_model
        layers = language_model.layers
        depth = len(layers)
        if self._extract_layer >= depth:
            raise ValueError(
                f"extract_layer={self._extract_layer} is out of range; "
                f"model has {depth} layers (valid: 0..{depth - 1})"
            )

        if self._layers is not None and self._layers is not layers:
            raise RuntimeError(
                "attach_model called with a different model than before. "
                "Re-attaching to a fresh model is not supported."
            )

        self._layers = layers
        self._rotary_emb = language_model.rotary_emb
        self._depth = depth

        # Mirror Stage 2's forced eager attention so the layer-N output here
        # matches the layer-N output that Stage 2 would produce.
        language_model.config._attn_implementation = "eager"  # noqa: SLF001
        logger.info(
            "CP1LLMLayerExtractKeyBuilder: attached to model (depth=%d, extract_layer=%d)",
            depth,
            self._extract_layer,
        )

    # ------------------------------------------------------------------
    # QueryKeyBuilder protocol
    # ------------------------------------------------------------------

    def collect(self, checkpoint_id: CheckpointID, **stage_outputs: Any) -> None:
        """Cache GPU tensor references from Stage1Output. No copy, no transfer."""
        self._cache.clear()
        if "stage1" in stage_outputs:
            s1 = stage_outputs["stage1"]
            self._cache["prefix_embs"] = s1.prefix_embs               # [B, L, D]
            self._cache["prefix_pad_masks"] = s1.prefix_pad_masks     # [B, L]
            self._cache["prefix_att_2d_masks_4d"] = s1.prefix_att_2d_masks_4d  # [B, 1, L, L]
            self._cache["prefix_position_ids"] = s1.prefix_position_ids        # [B, L]
            self._cache["state"] = s1.state                            # [B, state_dim]

    def build(self, checkpoint_id: CheckpointID) -> dict[str, torch.Tensor]:
        """Run partial layer forward, reduce, transfer to CPU.

        Raises if `attach_model()` was never called.
        """
        if checkpoint_id not in (CheckpointID.CP1, CheckpointID.CP3):
            raise ValueError(f"Unsupported checkpoint_id: {checkpoint_id}")
        if self._layers is None:
            raise RuntimeError(
                "attach_model not called; cannot run layer-N forward. "
                "InferenceInterceptor should attach the model at construction; "
                "for offline use, call attach_model(model) before build()."
            )
        if "prefix_embs" not in self._cache:
            raise RuntimeError("collect(stage1=...) was not called before build()")

        result = self._extract(self._extract_layer)
        gpu_keys = self._reducer.reduce(result)

        keys: dict[str, torch.Tensor] = {}
        for field, tensor in gpu_keys.items():
            keys[field] = tensor.detach().cpu().to(torch.float32).contiguous()

        if self._enabled is None or ROBOT_STATE in self._enabled:
            state = self._cache["state"][0]                            # [state_dim]
            keys[ROBOT_STATE] = state.detach().cpu().to(torch.float32).contiguous()

        return keys

    @property
    def cached_data(self) -> dict[str, torch.Tensor]:
        return self._cache

    def clear(self) -> None:
        """Release per-cycle GPU references. Model layer refs persist."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Layer forward replay
    # ------------------------------------------------------------------

    def _extract(self, extract_layer: int) -> LLMLayerExtractResult:
        """Run layers[0..extract_layer] on cached prefix and return result."""
        prefix_embs = self._cache["prefix_embs"]                       # [B, L, D]
        attention_mask = self._cache["prefix_att_2d_masks_4d"]         # [B, 1, L, L]
        position_ids = self._cache["prefix_position_ids"]              # [B, L]
        pad_mask = self._cache["prefix_pad_masks"]                     # [B, L]

        # Match HF Gemma forward dtype-cast rule (modeling_gemma.py:506-507).
        layer0 = self._layers[0]
        hidden = _cast_to_layer_dtype(prefix_embs, layer0)

        # Position embeddings are computed once and shared across all layers.
        cos, sin = self._rotary_emb(hidden, position_ids)

        with torch.no_grad():
            for layer_idx in range(extract_layer + 1):
                layer_out = self._layers[layer_idx](
                    hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=None,
                    position_embeddings=(cos, sin),
                    adarms_cond=None,  # paligemma side: use_adarms[0]=False (Pi0.5)
                )
                hidden = layer_out[0]

        # Drop batch dim. B=1 in all online + offline call sites.
        hidden_no_batch = hidden[0]                                    # [L, D]
        pad_mask_no_batch = pad_mask[0]                                # [L]
        prefix_len = hidden_no_batch.shape[0]

        segment_offsets: dict[str, tuple[int, int]] = {
            field: (start, end) for field, start, end in _VISION_OFFSETS
        }
        segment_offsets[PROMPT_EMB] = (_PROMPT_START, prefix_len)

        return LLMLayerExtractResult(
            hidden_states=hidden_no_batch,
            pad_mask=pad_mask_no_batch,
            segment_offsets=segment_offsets,
            extract_layer=extract_layer,
        )
