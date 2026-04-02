"""Shared data models for the cache storage layer.

All types in this module are backend-agnostic: they define what data looks like
at the boundary between CacheOrchestrator and CacheStorage, not how any
particular database stores it.

Tensor contract
---------------
Every torch.Tensor inside CachePayload or CacheEntry must be:
  - CPU
  - contiguous
  - float32

The caller is responsible for converting before constructing these objects.
Tensors returned from search() / fetch_payload() also satisfy these guarantees;
GPU transfer is the Orchestrator's responsibility.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch

from openpi.cache.types import CheckpointID


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class UnsupportedFilterError(Exception):
    """Raised by CacheStorage when QueryFilter contains a field that the
    current backend does not support.  Fail-fast: never silently ignore filters,
    because an ignored filter can change cache semantics (searching the full DB
    instead of a restricted subset).
    """


# ---------------------------------------------------------------------------
# Payload & entry
# ---------------------------------------------------------------------------


@dataclass
class CachePayload:
    """The data stored alongside each cache entry.

    CP-specific field requirements:
      CP1            : action_chunk required; everything else None.
      CP2 full hit   : action_chunk required; intermediates / denoising_num_steps None.
      CP2 warm start : action_chunk + intermediates + denoising_num_steps required.
      CP3            : action_chunk + next_action_chunk required.

    intermediates keys
    ------------------
    Keys are the *float timestep values* from save_timesteps=(0.7, 0.5, 0.3).
    These are Python float literals — no precision issue when comparing against
    the same literals later.  Backends must serialise keys as f"{t:.4f}" strings
    to avoid JSON round-trip drift.

    denoising_num_steps
    -------------------
    Must match the num_steps used during the original inference.  Passed
    directly to run_stage3_from(start_t=t, num_steps=denoising_num_steps).

    task_key
    --------
    Canonical task identifier, e.g. "libero_spatial_pick_up_the_black_bowl".
    NOT the raw language prompt — callers must normalise before storing.
    """

    action_chunk: torch.Tensor                          # [50, 32] CPU float32
    intermediates: Optional[dict[float, torch.Tensor]] = None   # {t: x_t}
    denoising_num_steps: Optional[int] = None
    next_action_chunk: Optional[torch.Tensor] = None   # [50, 32] CPU float32
    task_key: str = ""

    def validate_for_checkpoint(self, checkpoint_id: CheckpointID) -> None:
        """Raise ValueError if CP-specific invariants are violated."""
        if self.action_chunk is None:
            raise ValueError("action_chunk is required for all checkpoints")
        if checkpoint_id == CheckpointID.CP3 and self.next_action_chunk is None:
            raise ValueError("next_action_chunk is required for CP3")
        if self.intermediates is not None and self.denoising_num_steps is None:
            raise ValueError(
                "denoising_num_steps must be set when intermediates is provided"
            )


@dataclass
class CacheEntry:
    """A single unit written to the vector store.

    id semantics
    ------------
    Computed as  stable_hash(checkpoint_id.name + ":" + sorted_concat_of_query_key_bytes).
    This is a *semantic dedup key*: the same observation at the same checkpoint
    always maps to the same id, so repeated writes are idempotent upserts.
    checkpoint_id is part of the hash so CP1/CP2/CP3 entries for the same
    observation do not overwrite each other.
    Task or step metadata is NOT included in the id — filtering is done via
    QueryFilter, not by id uniqueness.

    query_keys
    ----------
    Named query vectors, e.g. {"vision_0": Tensor[1024], "prompt_emb": Tensor[2048]}.
    Keys must be a subset of CACHE_QUERY_FIELDS (openpi.cache.types).
    Each tensor: CPU float32, L2-normalised, shape [dim].
    The backend stores the fields declared in its vector_dims; extra fields in
    query_keys are silently ignored by the backend.
    """

    id: str
    checkpoint_id: CheckpointID
    query_keys: dict[str, torch.Tensor]  # {field: [dim] CPU float32, L2-normalised}
    payload: CachePayload
    timestamp: float = field(default_factory=time.time)

    def validate(self) -> None:
        """Check CP-specific invariants.  Called automatically by CacheStorage.insert()."""
        self.payload.validate_for_checkpoint(self.checkpoint_id)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


@dataclass
class QueryFilter:
    """Per-query constraints that narrow the search scope.

    These are dynamic (differ per call) and therefore cannot live in backend
    config.  Backends that do not support a field must declare so via
    VectorStoreBackend.supported_filters(); CacheStorage will raise
    UnsupportedFilterError rather than silently drop the constraint.

    task_key   : Canonical task identifier — same normalisation as CachePayload.task_key.
    step_range : Restrict to entries whose step_idx is in [min, max] (inclusive).
    """

    task_key: Optional[str] = None
    step_range: Optional[tuple[int, int]] = None


@dataclass
class QuerySpec:
    """Everything a backend needs to execute one search.

    query_keys
    ----------
    Named query vectors, same key conventions as CacheEntry.query_keys.
    The backend uses only the fields it declared in vector_dims; extra fields
    are ignored.  At least one key must match a backend field, otherwise
    CacheStorage raises ValueError before calling the backend.

    Fusion strategy
    ---------------
    How multiple query fields are combined (e.g. RRF, weighted average) is an
    internal backend detail — it is not expressed here.  Configure fusion
    behaviour in the backend's config dataclass.
    """

    query_keys: dict[str, torch.Tensor]          # {field: [dim] CPU float32}
    top_k: int = 10
    checkpoint_id: Optional[CheckpointID] = None
    filters: Optional[QueryFilter] = None


# ---------------------------------------------------------------------------
# Search results (two-phase)
# ---------------------------------------------------------------------------


@dataclass
class SearchResultLite:
    """Lightweight search result — no payload tensors.

    search() returns this type so that the judge can screen candidates by score
    without transferring large action tensors.  Only the winning candidate(s)
    need a subsequent fetch_payload() call.

    score contract
    --------------
    Higher = more similar.  The exact range depends on the backend and query mode:
      - Single-field cosine backend (e.g. Qdrant cosine): ∈ [-1, 1].
      - Multi-field RRF fusion: small positive numbers whose scale is determined
        by the number of prefetch streams and the RRF k parameter.
    SimilarityJudge thresholds must be calibrated to match the backend/mode in use.

    result count
    ------------
    search() returns *at most* top_k results.  Client-side filtering (e.g. for
    backends that don't support server-side QueryFilter) may yield fewer than k
    results; callers must handle that gracefully.
    """

    id: str
    score: float
    checkpoint_id: CheckpointID


@dataclass
class SearchResult:
    """Full search result including payload, returned by fetch_payload() or
    search_and_fetch()."""

    id: str
    score: float
    payload: CachePayload
    checkpoint_id: CheckpointID


# ---------------------------------------------------------------------------
# Write results
# ---------------------------------------------------------------------------


@dataclass
class BatchInsertResult:
    """Outcome of a batch_insert() call.

    batch_insert is best-effort: partial failures do not abort the batch.
    The caller (background write thread) logs failed_ids and discards them —
    a cache miss on those entries is acceptable.
    """

    inserted: int
    failed_ids: list[str]
