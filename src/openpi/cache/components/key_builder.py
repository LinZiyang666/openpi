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

from openpi.cache.types import ROBOT_STATE, CheckpointID


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
