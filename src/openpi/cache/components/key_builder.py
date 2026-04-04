"""Query key construction from stage outputs.

Data flow: Stage outputs (GPU) -> collect() -> build() -> query_keys dict

Coupling map:
  DEPENDS ON:  Stage1Output/Stage3Output field shapes (models_pytorch/pi0_pytorch.py),
               types.py (ROBOT_STATE, CheckpointID)
  CONSUMED BY: CacheOrchestrator.check(), Gate (via cached_data), Judge (via cached_data)
  FEEDS INTO:  CacheStorage.search() via QuerySpec (must match backend vector_dims)
  IF CHANGED:  Orchestrator QuerySpec construction, Judge threshold calibration
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F

from openpi.cache.types import (
    PROMPT_EMB,
    ROBOT_STATE,
    VISION_0,
    VISION_1,
    VISION_2,
    CheckpointID,
)


@runtime_checkable
class QueryKeyBuilder(Protocol):
    """Build query keys from stage outputs for cache lookup.

    Data flow: Stage outputs (GPU) -> collect() -> build() -> query_keys dict
    Coupling:
      - DEPENDS ON: Stage1Output/Stage3Output field shapes (models_pytorch/pi0_pytorch.py)
      - CONSUMED BY: CacheOrchestrator.check(), Gate (via cached_data), Judge (via cached_data)
      - FEEDS INTO: CacheStorage.search() via QuerySpec (must match backend vector_dims)
      - IF CHANGED: Orchestrator QuerySpec construction, Judge threshold calibration
    """

    def collect(self, checkpoint_id: CheckpointID, **stage_outputs) -> None:
        """Collect and cache raw data from stage outputs.

        Tensors kept on original device (GPU) — no CPU transfer.
        Gate/Judge may read cached_data before build() is called.

        SAFETY: References are valid within a single infer() call.
        The staged path uses max-autotune-no-cudagraphs, so stage outputs
        are regular GPU tensors — not CUDAGraph-managed buffers.
        """
        ...

    def build(self, checkpoint_id: CheckpointID) -> dict[str, torch.Tensor]:
        """Build query key vectors from collected data.

        Returns dict of {field_name: [dim] CPU float32 L2-normalized}.
        Field names from CACHE_QUERY_FIELDS (openpi.cache.types).
        Crossing to storage boundary: tensors are materialized here via
        .cpu().float() — this is the ONLY D2H transfer point.
        """
        ...

    @property
    def cached_data(self) -> dict[str, torch.Tensor]:
        """Expose collected raw tensors (on original device) for Gate/Judge.
        Lifetime: valid from collect() until next collect() or clear().
        """
        ...

    def clear(self) -> None:
        """Release all cached references. Called at end of each inference cycle."""
        ...


class PlaceholderKeyBuilder:
    """Simplest key builder: raw state vector as query key.

    CP1: state [B, 32] -> L2 normalize on GPU -> [32] CPU float32
    CP3: state [B, 32] -> L2 normalize on GPU -> [32] CPU float32
         (same as CP1 for now; concat with action deferred to Step 6)

    Data flow: Stage1Output.state (GPU) -> _cache (GPU ref) -> build() -> CPU normalized

    Coupling:
      - DEPENDS ON: Stage1Output.state shape [B, 32]
      - IF Stage output shapes change: build() output dims change -> backend vector_dims must match

    Tensor lifecycle:
      - _cache: GPU references only, no clone (safe within single infer() call)
      - build() output: CPU float32 (materialized, safe for storage)
    """

    def __init__(self) -> None:
        self._cache: dict[str, torch.Tensor] = {}  # GPU tensors, no copy

    def collect(self, checkpoint_id: CheckpointID, **stage_outputs) -> None:
        self._cache.clear()
        if "stage1" in stage_outputs:
            # Hold reference only — tensor stays on GPU, no clone needed.
            # SAFETY: staged path uses max-autotune-no-cudagraphs, so stage
            # outputs are regular GPU tensors, not CUDAGraph-managed buffers.
            # Reference is valid within a single infer() call.
            self._cache["state"] = stage_outputs["stage1"].state  # [B, 32] GPU
        if "stage3" in stage_outputs:
            self._cache["action_chunk"] = stage_outputs["stage3"].action_chunk  # [B, 50, 32] GPU

    def build(self, checkpoint_id: CheckpointID) -> dict[str, torch.Tensor]:
        state = self._cache["state"][0]  # [32] GPU, drop batch dim

        if checkpoint_id in (CheckpointID.CP1, CheckpointID.CP3):
            # L2 normalize on GPU (~1us kernel), then single D2H transfer.
            # CP3 uses same key as CP1 for now; Step 6 will concat state + action.
            # .contiguous() to satisfy storage_types tensor contract (CPU + contiguous + float32).
            key = F.normalize(state, dim=0)
            return {ROBOT_STATE: key.cpu().float().contiguous()}

        raise ValueError(f"Unsupported checkpoint_id: {checkpoint_id}")

    @property
    def cached_data(self) -> dict[str, torch.Tensor]:
        return self._cache

    def clear(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# Token layout constants for prefix_embs
# ---------------------------------------------------------------------------

# SigLIP: image_resolution=224, patch_size=16 -> (224/16)^2 = 196 tokens per image
_TOKENS_PER_IMAGE = 196
_NUM_IMAGES = 3  # base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb

# Offsets into prefix_embs [B, prefix_len, emb_dim]:
#   [0:196)    = vision_0 (base_0_rgb)
#   [196:392)  = vision_1 (left_wrist_0_rgb)
#   [392:588)  = vision_2 (right_wrist_0_rgb)
#   [588:...)  = prompt_emb (language tokens, padded to max_token_len)
_VISION_OFFSETS = [
    (VISION_0, 0, _TOKENS_PER_IMAGE),
    (VISION_1, _TOKENS_PER_IMAGE, 2 * _TOKENS_PER_IMAGE),
    (VISION_2, 2 * _TOKENS_PER_IMAGE, 3 * _TOKENS_PER_IMAGE),
]
_PROMPT_START = _NUM_IMAGES * _TOKENS_PER_IMAGE  # 588


class FullOriginalKeyBuilder:
    """Raw key builder: slice prefix_embs back into original segments and flatten.

    embed_prefix() concatenates [vision_0 | vision_1 | vision_2 | prompt] into
    prefix_embs. This builder simply reverses that concatenation — slices at
    fixed token boundaries and flattens each segment into a 1-D vector.
    No mean pooling, no L2 normalization. The output matches what the collect
    system's forward hooks capture (multi_modal_projector / embed_tokens output).

    Token layout (from embed_prefix()):
      prefix_embs = [vision_0 (196) | vision_1 (196) | vision_2 (196) | prompt (max_token_len)]

    Output dims (after flatten):
      vision_0:    [196 * emb_dim]   (e.g. 196 * 2048 = 401408)
      vision_1:    [196 * emb_dim]
      vision_2:    [196 * emb_dim]
      prompt_emb:  [max_token_len * emb_dim]  (e.g. 200 * 2048 = 409600)
      robot_state: [action_dim]      (e.g. 32)

    Coupling:
      - DEPENDS ON: Stage1Output fields (prefix_embs, state)
      - DEPENDS ON: SigLIP patch_size=16, image_resolution=224 (196 tokens/image)
      - DEPENDS ON: 3 images in fixed order (IMAGE_KEYS in model.py)
      - IF image count or SigLIP config changes: offsets must be updated
      - IF Stage output shapes change: backend vector_dims must match
    """

    def __init__(self, enabled_fields: list[str] | None = None) -> None:
        self._cache: dict[str, torch.Tensor] = {}
        self._enabled = set(enabled_fields) if enabled_fields is not None else None

    def _is_enabled(self, field: str) -> bool:
        return self._enabled is None or field in self._enabled

    def collect(self, checkpoint_id: CheckpointID, **stage_outputs) -> None:
        self._cache.clear()
        if "stage1" in stage_outputs:
            s1 = stage_outputs["stage1"]
            self._cache["state"] = s1.state                       # [B, action_dim]
            self._cache["prefix_embs"] = s1.prefix_embs           # [B, prefix_len, emb_dim]
        if "stage3" in stage_outputs:
            self._cache["action_chunk"] = stage_outputs["stage3"].action_chunk

    def build(self, checkpoint_id: CheckpointID) -> dict[str, torch.Tensor]:
        if checkpoint_id not in (CheckpointID.CP1, CheckpointID.CP3):
            raise ValueError(f"Unsupported checkpoint_id: {checkpoint_id}")

        keys: dict[str, torch.Tensor] = {}
        prefix = self._cache["prefix_embs"][0]  # [prefix_len, emb_dim]

        # Vision fields: slice at fixed 196-token boundaries, flatten raw.
        for field_name, start, end in _VISION_OFFSETS:
            if not self._is_enabled(field_name):
                continue
            segment = prefix[start:end]  # [196, emb_dim]
            keys[field_name] = segment.reshape(-1).cpu().float().contiguous()

        # Prompt: slice remainder after vision tokens, flatten raw.
        if self._is_enabled(PROMPT_EMB):
            prompt_segment = prefix[_PROMPT_START:]  # [max_token_len, emb_dim]
            keys[PROMPT_EMB] = prompt_segment.reshape(-1).cpu().float().contiguous()

        # Robot state: raw, no normalization.
        if self._is_enabled(ROBOT_STATE):
            state = self._cache["state"][0]  # [action_dim]
            keys[ROBOT_STATE] = state.cpu().float().contiguous()

        return keys

    @property
    def cached_data(self) -> dict[str, torch.Tensor]:
        return self._cache

    def clear(self) -> None:
        self._cache.clear()
