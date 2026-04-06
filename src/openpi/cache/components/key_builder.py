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

# SigLIP: image_resolution=224, patch_size=14 -> (224/14)^2 = 256 tokens per image
_TOKENS_PER_IMAGE = 256
_NUM_IMAGES = 3  # base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb

# Offsets into prefix_embs [B, prefix_len, emb_dim]:
#   [0:256)    = vision_0 (base_0_rgb)
#   [256:512)  = vision_1 (left_wrist_0_rgb)
#   [512:768)  = vision_2 (right_wrist_0_rgb)
#   [768:...)  = prompt_emb (language tokens, variable length)
_VISION_OFFSETS = [
    (VISION_0, 0, _TOKENS_PER_IMAGE),
    (VISION_1, _TOKENS_PER_IMAGE, 2 * _TOKENS_PER_IMAGE),
    (VISION_2, 2 * _TOKENS_PER_IMAGE, 3 * _TOKENS_PER_IMAGE),
]
_PROMPT_START = _NUM_IMAGES * _TOKENS_PER_IMAGE  # 588


# ---------------------------------------------------------------------------
# CP1 experiment key builder helpers (private)
# ---------------------------------------------------------------------------


def _slice_cp1_fields(
    prefix_embs: torch.Tensor,
    state: torch.Tensor,
    enabled: set[str] | None,
) -> dict[str, torch.Tensor]:
    """Slice per-modality raw token sequences from prefix_embs[0] and state[0].

    Returns GPU tensors (not yet reduced or transferred to CPU).
    vision_*: [256, emb_dim], prompt_emb: [num_tokens, emb_dim], robot_state: [state_dim]
    """
    result: dict[str, torch.Tensor] = {}
    prefix = prefix_embs[0]  # [prefix_len, emb_dim], drop batch dim

    for field_name, start, end in _VISION_OFFSETS:
        if enabled is not None and field_name not in enabled:
            continue
        result[field_name] = prefix[start:end]  # [256, emb_dim]

    if enabled is None or PROMPT_EMB in enabled:
        result[PROMPT_EMB] = prefix[_PROMPT_START:]  # [num_prompt_tokens, emb_dim]

    if enabled is None or ROBOT_STATE in enabled:
        result[ROBOT_STATE] = state[0]  # [state_dim]

    return result


def _mean_pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """[num_tokens, emb_dim] -> [emb_dim], mean over token dim."""
    return tokens.mean(dim=0)


def _max_pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """[num_tokens, emb_dim] -> [emb_dim], per-dim max over token dim."""
    return tokens.max(dim=0).values


def _spatial_pool_tokens(
    tokens: torch.Tensor,
    grid_size: int,
    pool_size: int,
) -> torch.Tensor:
    """2D adaptive average pooling on vision patch tokens.

    Args:
        tokens: [grid_size*grid_size, emb_dim] flat SigLIP patch tokens
        grid_size: original grid side (16, since 256 = 16x16)
        pool_size: output grid side (4 for B1, 2 for B2)

    Returns:
        [pool_size*pool_size * emb_dim] flattened 1-D vector
    """
    emb_dim = tokens.shape[1]
    # [grid*grid, emb_dim] -> [1, emb_dim, grid, grid]
    x = tokens.reshape(grid_size, grid_size, emb_dim).permute(2, 0, 1).unsqueeze(0)
    # [1, emb_dim, pool, pool]
    pooled = F.adaptive_avg_pool2d(x, (pool_size, pool_size))
    # -> [pool*pool * emb_dim]
    return pooled.squeeze(0).permute(1, 2, 0).reshape(-1)


def _to_cpu_float32(t: torch.Tensor) -> torch.Tensor:
    """GPU tensor -> CPU float32 contiguous. Single D2H transfer point."""
    return t.cpu().float().contiguous()


# ---------------------------------------------------------------------------
# CP1 experiment key builder base class
# ---------------------------------------------------------------------------


class _CP1BaseKeyBuilder:
    """Shared base for CP1 experiment key builders.

    Subclasses override _reduce_vision() and _reduce_prompt() only.
    robot_state is always output as raw vector (no L2 normalize, no pooling)
    because the backend needs real L2 distance for exp(-d/tau) conversion.
    """

    def __init__(self, enabled_fields: list[str] | None = None) -> None:
        self._cache: dict[str, torch.Tensor] = {}
        self._enabled = set(enabled_fields) if enabled_fields is not None else None

    def collect(self, checkpoint_id: CheckpointID, **stage_outputs) -> None:
        self._cache.clear()
        if "stage1" in stage_outputs:
            s1 = stage_outputs["stage1"]
            self._cache["state"] = s1.state               # [B, state_dim] GPU
            self._cache["prefix_embs"] = s1.prefix_embs   # [B, prefix_len, emb_dim] GPU
        if "stage3" in stage_outputs:
            self._cache["action_chunk"] = stage_outputs["stage3"].action_chunk

    def build(self, checkpoint_id: CheckpointID) -> dict[str, torch.Tensor]:
        if checkpoint_id not in (CheckpointID.CP1, CheckpointID.CP3):
            raise ValueError(f"Unsupported checkpoint_id: {checkpoint_id}")

        raw = _slice_cp1_fields(
            self._cache["prefix_embs"],
            self._cache["state"],
            self._enabled,
        )
        keys: dict[str, torch.Tensor] = {}

        for field_name in (VISION_0, VISION_1, VISION_2):
            if field_name in raw:
                keys[field_name] = _to_cpu_float32(self._reduce_vision(raw[field_name]))

        if PROMPT_EMB in raw:
            keys[PROMPT_EMB] = _to_cpu_float32(self._reduce_prompt(raw[PROMPT_EMB]))

        if ROBOT_STATE in raw:
            keys[ROBOT_STATE] = _to_cpu_float32(raw[ROBOT_STATE])

        return keys

    def _reduce_vision(self, tokens: torch.Tensor) -> torch.Tensor:
        """[256, emb_dim] -> [reduced_dim]. Subclass override."""
        raise NotImplementedError

    def _reduce_prompt(self, tokens: torch.Tensor) -> torch.Tensor:
        """[num_tokens, emb_dim] -> [reduced_dim]. Subclass override."""
        raise NotImplementedError

    @property
    def cached_data(self) -> dict[str, torch.Tensor]:
        return self._cache

    def clear(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# CP1 experiment key builder concrete classes
# ---------------------------------------------------------------------------


class CP1MeanPoolKeyBuilder(_CP1BaseKeyBuilder):
    """Group A: Mean Pool. vision/prompt mean-pooled to [emb_dim].
    Output: vision_*=[2048], prompt_emb=[2048], robot_state=[32].
    """

    def _reduce_vision(self, tokens: torch.Tensor) -> torch.Tensor:
        return _mean_pool_tokens(tokens)

    def _reduce_prompt(self, tokens: torch.Tensor) -> torch.Tensor:
        return _mean_pool_tokens(tokens)


class CP1SpatialPool16KeyBuilder(_CP1BaseKeyBuilder):
    """Group B1: Spatial Pool 4x4 (16x compression).
    Output: vision_*=[16*2048=32768], prompt_emb=[2048] (mean pool), robot_state=[32].
    """

    _GRID_SIZE = 16   # sqrt(256)
    _POOL_SIZE = 4    # 16x16 -> 4x4

    def _reduce_vision(self, tokens: torch.Tensor) -> torch.Tensor:
        return _spatial_pool_tokens(tokens, self._GRID_SIZE, self._POOL_SIZE)

    def _reduce_prompt(self, tokens: torch.Tensor) -> torch.Tensor:
        return _mean_pool_tokens(tokens)


class CP1SpatialPool64KeyBuilder(_CP1BaseKeyBuilder):
    """Group B2: Spatial Pool 2x2 (64x compression).
    Output: vision_*=[4*2048=8192], prompt_emb=[2048] (mean pool), robot_state=[32].
    """

    _GRID_SIZE = 16
    _POOL_SIZE = 2    # 16x16 -> 2x2

    def _reduce_vision(self, tokens: torch.Tensor) -> torch.Tensor:
        return _spatial_pool_tokens(tokens, self._GRID_SIZE, self._POOL_SIZE)

    def _reduce_prompt(self, tokens: torch.Tensor) -> torch.Tensor:
        return _mean_pool_tokens(tokens)


class CP1MaxPoolKeyBuilder(_CP1BaseKeyBuilder):
    """Group C: Max Pool. vision/prompt per-dim max to [emb_dim].
    Output: vision_*=[2048], prompt_emb=[2048], robot_state=[32].
    """

    def _reduce_vision(self, tokens: torch.Tensor) -> torch.Tensor:
        return _max_pool_tokens(tokens)

    def _reduce_prompt(self, tokens: torch.Tensor) -> torch.Tensor:
        return _max_pool_tokens(tokens)


class FullOriginalKeyBuilder:
    """Raw key builder: slice prefix_embs back into original segments and flatten.

    embed_prefix() concatenates [vision_0 | vision_1 | vision_2 | prompt] into
    prefix_embs. This builder simply reverses that concatenation — slices at
    fixed token boundaries and flattens each segment into a 1-D vector.
    No mean pooling, no L2 normalization. The output matches what the collect
    system's forward hooks capture (multi_modal_projector / embed_tokens output).

    Token layout (from embed_prefix()):
      prefix_embs = [vision_0 (256) | vision_1 (256) | vision_2 (256) | prompt (variable)]

    Output dims (after flatten):
      vision_0:    [256 * emb_dim]   (e.g. 256 * 2048 = 524288)
      vision_1:    [256 * emb_dim]
      vision_2:    [256 * emb_dim]
      prompt_emb:  [target_dim from vector_dims]  (zero-padded/truncated to match Qdrant schema)
      robot_state: [action_dim]      (e.g. 32)

    Prompt padding:
      prompt token count varies per input. Qdrant requires fixed-dim vectors.
      The ingest pipeline (exp/qdrant_openpi_common.py::pad_and_flatten_prompt)
      zero-pads to (max_lang_tokens * emb_dim). This builder does the same:
      if vector_dims is provided, prompt_emb is zero-padded or truncated to
      vector_dims[PROMPT_EMB].

    Coupling:
      - DEPENDS ON: Stage1Output fields (prefix_embs, state)
      - DEPENDS ON: SigLIP patch_size=14, image_resolution=224 (256 tokens/image)
      - DEPENDS ON: 3 images in fixed order (IMAGE_KEYS in model.py)
      - IF image count or SigLIP config changes: offsets must be updated
      - IF Stage output shapes change: backend vector_dims must match
    """

    def __init__(
        self,
        enabled_fields: list[str] | None = None,
        vector_dims: dict[str, int] | None = None,
    ) -> None:
        self._cache: dict[str, torch.Tensor] = {}
        self._enabled = set(enabled_fields) if enabled_fields is not None else None
        self._vector_dims = vector_dims or {}

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

        # Prompt: slice remainder after vision tokens, flatten, pad/truncate to target dim.
        # Prompt token count varies per input but Qdrant needs fixed-dim vectors.
        # Zero-pad or truncate to match vector_dims[PROMPT_EMB], mirroring
        # exp/qdrant_openpi_common.py::pad_and_flatten_prompt().
        if self._is_enabled(PROMPT_EMB):
            prompt_segment = prefix[_PROMPT_START:]  # [num_prompt_tokens, emb_dim]
            flat = prompt_segment.reshape(-1).cpu().float().contiguous()
            target_dim = self._vector_dims.get(PROMPT_EMB)
            if target_dim is not None and flat.shape[0] != target_dim:
                if flat.shape[0] < target_dim:
                    flat = torch.nn.functional.pad(flat, (0, target_dim - flat.shape[0]))
                else:
                    flat = flat[:target_dim]
            keys[PROMPT_EMB] = flat

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
