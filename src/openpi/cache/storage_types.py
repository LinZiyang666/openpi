"""Shared data models for the cache storage layer.

All types in this module are backend-agnostic: they define what data looks like
at the boundary between CacheOrchestrator and CacheStorage, not how any
particular database stores it.

Coupling map:
  DEPENDS ON:  types.py (CheckpointID)
  CONSUMED BY: KeyBuilder (CachePayload, CacheEntry construction),
               Orchestrator (SearchResultLite, SearchResult, CachePayload,
                             StepRecord, EpisodeRecord),
               SearchStrategy (QuerySpec construction),
               CacheStorage (all types), backends (all types)
  IF CHANGED:  SearchStrategy QuerySpec construction,
               Backend.search() fusion parameter reading,
               Orchestrator write path (CacheEntry, StepRecord, EpisodeRecord)

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
from typing import TYPE_CHECKING, Any, Optional

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
      CP1 FULL_HIT   : action_chunk required; intermediates / denoising_num_steps optional.
      CP1 WARM_START : action_chunk + intermediates + denoising_num_steps required.

    intermediates keys
    ------------------
    Keys are float timestep values.  Online path stores the default
    save_timesteps (0.7, 0.5, 0.3); offline artifact path stores all 9
    intermediates (0.9, 0.8, ..., 0.1).  Judge's warm_tiers.start_t must
    exist in the payload's intermediates keys; Orchestrator validates this
    and downgrades to MISS if the key is absent.
    Backends must serialise keys as f"{t:.4f}" strings to avoid JSON
    round-trip drift.

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
    task_key: str = ""

    def validate_for_checkpoint(self, checkpoint_id: CheckpointID) -> None:
        """Raise ValueError if CP-specific invariants are violated."""
        if self.action_chunk is None:
            raise ValueError("action_chunk is required for all checkpoints")
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
    Each tensor: CPU float32 contiguous, shape [dim].
    Whether to L2-normalise depends on the field's similarity type:
      - cosine fields (vision_0/1/2, prompt_emb): not required (F.cosine_similarity handles it)
      - L2 distance fields (robot_state): must keep raw vector (L2 normalise would destroy distance semantics)
    The backend stores the fields declared in its vector_dims; extra fields in
    query_keys are silently ignored by the backend.
    """

    id: str
    checkpoint_id: CheckpointID
    query_keys: dict[str, torch.Tensor]  # {field: [dim] CPU float32}
    payload: CachePayload
    step_idx: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    # ── Trajectory linked-list fields ──
    prev_ids: list[str] = field(default_factory=list)
    next_ids: list[str] = field(default_factory=list)
    trajectory_id: Optional[str] = None

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

    Data flow: SearchStrategy -> QuerySpec -> CacheStorage.search() -> Backend.search()

    query_keys
    ----------
    Named query vectors, same key conventions as CacheEntry.query_keys.
    The backend uses only the fields it declared in vector_dims; extra fields
    are ignored.  At least one key must match a backend field, otherwise
    CacheStorage raises ValueError before calling the backend.

    Fusion parameters
    -----------------
    fusion_weights is a generic concept (any multi-vector backend may need weighted
    fusion). Backend-specific parameters (e.g. Qdrant's rrf_k, candidate_multiplier)
    are passed via backend_hints — backends read what they recognise, ignore the rest.

    Coupling:
      - CONSTRUCTED BY: SearchStrategy (the only constructor after this change)
      - CONSUMED BY: CacheStorage.search() (validation), Backend.search() (execution)
      - IF CHANGED: SearchStrategy construction logic, Backend.search() parameter reading
    """

    query_keys: dict[str, torch.Tensor]          # {field: [dim] CPU float32}
    top_k: int = 10
    checkpoint_id: Optional[CheckpointID] = None
    filters: Optional[QueryFilter] = None
    fusion_weights: Optional[dict[str, float]] = None     # per-field fusion weights (backend-agnostic)
    backend_hints: Optional[dict[str, Any]] = None        # e.g. {"rrf_k": 60, "candidate_multiplier": 5}

    fusion_method: Optional[str] = None
    # "weighted_rrf" | "weighted_score_sum" | None
    # None: backend falls back to single-field cosine (backward compatible)

    field_similarity: Optional[dict[str, dict[str, Any]]] = None
    # Per-field similarity definition, e.g.:
    # {"vision_0": {"type": "cosine"},
    #  "robot_state": {"type": "l2", "to_similarity": {"type": "exp", "tau": 0.334717}}}

    score_normalization: Optional[dict[str, Any]] = None
    # Only for weighted_score_sum, e.g.:
    # {"type": "percentile",
    #  "fields": {"vision_0": {"p5": 0.82, "p95": 0.99}, ...}}

    # ── Trajectory search fields ──
    trajectory_history: Optional[list[dict[str, torch.Tensor]]] = None
    # Query-side history key sequence, newest-first:
    # [current_step_keys, t-1_keys, t-2_keys, ...]
    # None → falls back to existing single-step search.

    trajectory_weights: Optional[list[float]] = None
    # Per-level weights, newest-first: [w_current, w_t-1, w_t-2, ...]
    # Must be same length as trajectory_history. None → no trajectory fusion.

    # ── Cross-step score memo fields (opt-in; None → uncached path) ──
    search_session_id: Optional[str] = None
    # Per-strategy-per-episode search session id (uuid4 hex). When None, the
    # backend takes the trunk uncached path. Minted by TrajectoryMixin in
    # on_episode_start and registered with the backend by CacheOrchestrator.

    trajectory_query_ids: Optional[list[int]] = None
    # Session-scoped monotonic query identifiers, parallel to
    # trajectory_history (newest-first). Same query tensor keeps the same id
    # across steps, enabling cross-step cosine reuse. None → uncached path.


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


# ---------------------------------------------------------------------------
# Episode-level records (trajectory write path)
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """Per-step staging record for episode-end batch write.

    Accumulated by Orchestrator.buffer_for_write() during an episode,
    then consumed by on_episode_end() to build a linked CacheEntry chain.
    """

    query_keys: dict[str, torch.Tensor]   # CPU float32
    action_chunk: torch.Tensor            # CPU float32, required
    intermediates: Optional[dict[float, torch.Tensor]] = None
    denoising_num_steps: Optional[int] = None


@dataclass
class EpisodeRecord:
    """Entire episode staging record, passed to WritePolicy.should_write().

    Constructed by Orchestrator.on_episode_end() from accumulated StepRecords.
    """

    steps: list[StepRecord]
    task_key: str
    miss_by_checkpoint: dict[CheckpointID, int]  # e.g. {CP1: 3}
    total_steps: int
