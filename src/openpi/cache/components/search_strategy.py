"""SearchStrategy: the single exit point for database search.

Overview
--------
SearchStrategy encapsulates all search parameters (top_k, fusion weights,
step filtering, etc.) and is the ONLY component that constructs QuerySpec
and calls CacheStorage.search().  This decouples search logic from both
Orchestrator (pure orchestration) and Backend (pure KNN execution).

Data flow: SearchContext (from Orchestrator) -> SearchStrategy.search()
             -> QuerySpec (with fusion params + backend_hints) -> CacheStorage.search()
             -> Backend.search() -> list[SearchResultLite]

Coupling map:
  DEPENDS ON:  CacheStorage.search() (Step 3 facade),
               storage_types (QuerySpec, QueryFilter, SearchResultLite)
  CONSUMED BY: CacheOrchestrator.check() — replaces inline QuerySpec construction
  DOES NOT depend on: VectorStoreBackend, KeyBuilder, Gate, Judge
  SHARES:      CacheStorage instance with Orchestrator (Orchestrator uses insert/fetch_payload)
  IF CHANGED:  Orchestrator.check() search call path,
               Judge thresholds may need recalibration (if search semantics change)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

import torch

logger = logging.getLogger(__name__)

from openpi.cache.cache_storage import CacheStorage
from openpi.cache.storage_types import QueryFilter, QuerySpec, SearchResultLite
from openpi.cache.types import CheckpointID


@dataclass
class SearchContext:
    """Runtime context passed from Orchestrator to SearchStrategy.

    Data flow: Orchestrator constructs -> SearchStrategy reads -> used for QuerySpec/filter

    Coupling:
      - CONSTRUCTED BY: CacheOrchestrator.check() (fills from step counter + stage outputs)
      - CONSUMED BY: SearchStrategy.search()
      - IF CHANGED: Orchestrator construction logic, SearchStrategy read logic
    """

    query_keys: dict[str, torch.Tensor]   # from KeyBuilder.build()
    checkpoint_id: CheckpointID
    current_step: int = 0                  # inference cycle count within current task
    task_key: Optional[str] = None         # normalised task identifier (None = no task filter)


@runtime_checkable
class SearchStrategy(Protocol):
    """Encapsulate all search parameters and database interaction.

    Data flow: SearchContext -> build QuerySpec -> CacheStorage.search() -> results

    Coupling:
      - DEPENDS ON: CacheStorage.search()
      - CONSUMED BY: CacheOrchestrator.check()
      - DOES NOT: make hit/miss decisions (that's Judge's job)
      - IF CHANGED: Orchestrator.check() must adapt to new return type or semantics
    """

    def search(self, ctx: SearchContext) -> list[SearchResultLite]:
        """Execute a search against the cache storage.

        Args:
            ctx: Runtime context containing query_keys, checkpoint_id,
                 current_step, and optional task_key. Constructed by Orchestrator.

        Returns:
            Search results sorted by descending score. Judge decides hit/miss.
        """
        ...


class TrajectoryMixin:
    """Shared trajectory buffer logic, mixed into SearchStrategy implementations.

    Provides:
      - History buffer management (query_keys)
      - on_episode_start() lifecycle
      - record_action() broadcast receiver
      - record_query_keys() for gate-skip path
      - _build_trajectory_fields() for QuerySpec construction
    """

    def _init_trajectory(self, trajectory_depth: int, trajectory_weights: Optional[list[float]]) -> None:
        """Call at the end of strategy __init__."""
        self._trajectory_depth = trajectory_depth
        self._trajectory_weights = trajectory_weights
        self._query_history: list[dict[str, torch.Tensor]] = []
        self._action_history: list[Optional[torch.Tensor]] = []
        # Per-strategy search session state (opt-in score memo).
        # Sid is minted in on_episode_start; orchestrator collects it via
        # get_search_session_id() and registers it with the backend.
        self._search_session_id: Optional[str] = None
        self._query_id_counter: int = 0
        self._query_id_history: list[int] = []

    def on_episode_start(self) -> None:
        """Clear history buffers and mint a fresh per-strategy search session.

        Called by Orchestrator at episode start. The minted session id is
        exposed to the orchestrator via `get_search_session_id()` so it can
        be registered with the backend through `storage.open_search_session`.
        """
        self._query_history.clear()
        self._action_history.clear()
        self._query_id_history.clear()
        self._query_id_counter = 0
        self._search_session_id = uuid.uuid4().hex

    def get_search_session_id(self) -> Optional[str]:
        """Return the current per-strategy search session id (or None).

        CacheOrchestrator reads this after broadcasting on_episode_start to
        every strategy and forwards each non-None sid to the backend via
        `storage.open_search_session(sid)`.
        """
        return self._search_session_id

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """Receive Orchestrator-broadcast action. Pure local buffer op."""
        self._action_history.append(action_chunk)

    def record_query_keys(self, query_keys: dict[str, torch.Tensor]) -> None:
        """Buffer current step's query_keys into trajectory history.

        Called by search() internally (normal path) and by Orchestrator
        explicitly on gate skip (to keep trajectory history gap-free).
        Maintains a parallel monotonic query_id list used by the backend
        score memo (key includes the query_id rather than the layer).
        """
        self._query_history.append(query_keys)
        qid = self._query_id_counter
        self._query_id_counter += 1
        self._query_id_history.append(qid)

    def _build_trajectory_fields(self) -> dict[str, Any]:
        """Return trajectory fields for QuerySpec construction.

        Returns empty dict when depth=1 or insufficient history,
        causing QuerySpec fields to stay None (single-step fallback).
        Includes search_session_id + trajectory_query_ids when a session is
        active, enabling cross-step cosine reuse in the backend.
        """
        if self._trajectory_depth <= 1 or not self._trajectory_weights:
            return {}

        actual_depth = min(self._trajectory_depth, len(self._query_history))
        if actual_depth <= 1:
            logger.debug(
                "Trajectory: history=%d < 2, falling back to single-step",
                len(self._query_history),
            )
            return {}

        history_newest_first = list(reversed(self._query_history[-actual_depth:]))
        weights_newest_first = self._trajectory_weights[:actual_depth]

        fields: dict[str, Any] = {
            "trajectory_history": history_newest_first,
            "trajectory_weights": weights_newest_first,
        }
        if self._search_session_id is not None:
            qids_newest_first = list(reversed(self._query_id_history[-actual_depth:]))
            fields["search_session_id"] = self._search_session_id
            fields["trajectory_query_ids"] = qids_newest_first
        return fields


class QdrantWeightedRrfKnnStrategy(TrajectoryMixin):
    """Qdrant-only KNN search strategy for weighted RRF retrieval.

    Data flow: SearchContext -> QueryFilter + QuerySpec(fusion, backend_hints)
              -> CacheStorage.search() -> results

    Coupling:
      - DEPENDS ON: CacheStorage.search() (thread-safe, RLock-protected)
      - HOLDS: all search parameters (top_k, step_filter, fusion_weights, etc.)
      - SHARES: CacheStorage instance with Orchestrator (search vs insert paths)
      - IF CHANGED: search behavior changes, Judge thresholds may need recalibration

    Parameters held (not from config -- injected via constructor):
      - top_k: number of results to return
      - step_filter: "all" (no filter) | "exact" | "window"
      - step_window: window size for "window" mode
      - rrf_k: Qdrant RRF fusion parameter k (passed via backend_hints)
      - fusion_weights: per-field fusion weights used by Qdrant RRF
      - candidate_multiplier: Qdrant prefetch limit (passed via backend_hints)

    This strategy only names a search intent; actual weighted RRF execution
    happens inside QdrantVectorStore.search().
    """

    def __init__(
        self,
        storage: CacheStorage,
        *,
        top_k: int = 1,
        step_filter: str = "all",
        step_window: int = 5,
        rrf_k: int = 60,
        fusion_weights: Optional[dict[str, float]] = None,
        candidate_multiplier: int = 5,
        trajectory_depth: int = 1,
        trajectory_weights: Optional[list[float]] = None,
    ) -> None:
        self._storage = storage
        self._top_k = top_k
        self._step_filter = step_filter
        self._step_window = step_window
        self._rrf_k = rrf_k
        self._fusion_weights = fusion_weights
        self._candidate_multiplier = candidate_multiplier
        self._init_trajectory(trajectory_depth, trajectory_weights)

    def search(self, ctx: SearchContext) -> list[SearchResultLite]:
        """Execute KNN search with configured fusion parameters."""
        self.record_query_keys(ctx.query_keys)

        filters = self._build_filters(ctx)
        spec = QuerySpec(
            query_keys=ctx.query_keys,
            top_k=self._top_k,
            checkpoint_id=ctx.checkpoint_id,
            filters=filters,
            fusion_weights=self._fusion_weights,
            backend_hints={
                "rrf_k": self._rrf_k,
                "candidate_multiplier": self._candidate_multiplier,
            },
            **self._build_trajectory_fields(),
        )
        return self._storage.search(spec)

    def _build_filters(self, ctx: SearchContext) -> Optional[QueryFilter]:
        return _build_step_filters(self._step_filter, self._step_window, ctx)


# ---------------------------------------------------------------------------
# Shared filter builder (used by all strategy implementations)
# ---------------------------------------------------------------------------


def _build_step_filters(
    step_filter: str,
    step_window: int,
    ctx: SearchContext,
) -> Optional[QueryFilter]:
    """Build QueryFilter from step_filter config + runtime context.

    Shared by QdrantWeightedRrfKnnStrategy, WeightedRrfKnnStrategy,
    and WeightedScoreSumKnnStrategy.
    """
    task_filter = QueryFilter(task_key=ctx.task_key) if ctx.task_key else None

    if step_filter == "all":
        return task_filter
    elif step_filter == "exact":
        f = QueryFilter(step_range=(ctx.current_step, ctx.current_step))
        if ctx.task_key:
            f.task_key = ctx.task_key
        return f
    elif step_filter == "window":
        lo = max(0, ctx.current_step - step_window)
        hi = ctx.current_step + step_window
        f = QueryFilter(step_range=(lo, hi))
        if ctx.task_key:
            f.task_key = ctx.task_key
        return f
    else:
        raise ValueError(f"Unknown step_filter: {step_filter}")


# ---------------------------------------------------------------------------
# In-memory experiment strategies
# ---------------------------------------------------------------------------


class WeightedRrfKnnStrategy(TrajectoryMixin):
    """In-memory weighted RRF search strategy.

    Sets fusion_method="weighted_rrf" in QuerySpec.
    InMemoryBackend executes the actual per-field ranking and RRF fusion.
    """

    def __init__(
        self,
        storage: CacheStorage,
        *,
        top_k: int = 1,
        step_filter: str = "all",
        step_window: int = 5,
        fusion_weights: Optional[dict[str, float]] = None,
        rrf_k: int = 60,
        field_similarity: Optional[dict[str, dict[str, Any]]] = None,
        trajectory_depth: int = 1,
        trajectory_weights: Optional[list[float]] = None,
    ) -> None:
        self._storage = storage
        self._top_k = top_k
        self._step_filter = step_filter
        self._step_window = step_window
        self._fusion_weights = fusion_weights
        self._rrf_k = rrf_k
        self._field_similarity = field_similarity
        self._init_trajectory(trajectory_depth, trajectory_weights)

    def search(self, ctx: SearchContext) -> list[SearchResultLite]:
        self.record_query_keys(ctx.query_keys)

        filters = _build_step_filters(self._step_filter, self._step_window, ctx)
        spec = QuerySpec(
            query_keys=ctx.query_keys,
            top_k=self._top_k,
            checkpoint_id=ctx.checkpoint_id,
            filters=filters,
            fusion_weights=self._fusion_weights,
            fusion_method="weighted_rrf",
            field_similarity=self._field_similarity,
            backend_hints={"rrf_k": self._rrf_k},
            **self._build_trajectory_fields(),
        )
        return self._storage.search(spec)


class WeightedScoreSumKnnStrategy(TrajectoryMixin):
    """In-memory weighted score sum search strategy.

    Sets fusion_method="weighted_score_sum" in QuerySpec.
    InMemoryBackend executes similarity computation, normalization, and weighted sum.
    """

    def __init__(
        self,
        storage: CacheStorage,
        *,
        top_k: int = 1,
        step_filter: str = "all",
        step_window: int = 5,
        fusion_weights: Optional[dict[str, float]] = None,
        field_similarity: Optional[dict[str, dict[str, Any]]] = None,
        score_normalization: Optional[dict[str, Any]] = None,
        trajectory_depth: int = 1,
        trajectory_weights: Optional[list[float]] = None,
    ) -> None:
        self._storage = storage
        self._top_k = top_k
        self._step_filter = step_filter
        self._step_window = step_window
        self._fusion_weights = fusion_weights
        self._field_similarity = field_similarity
        self._score_normalization = score_normalization
        self._init_trajectory(trajectory_depth, trajectory_weights)

    def search(self, ctx: SearchContext) -> list[SearchResultLite]:
        self.record_query_keys(ctx.query_keys)

        filters = _build_step_filters(self._step_filter, self._step_window, ctx)
        spec = QuerySpec(
            query_keys=ctx.query_keys,
            top_k=self._top_k,
            checkpoint_id=ctx.checkpoint_id,
            filters=filters,
            fusion_weights=self._fusion_weights,
            fusion_method="weighted_score_sum",
            field_similarity=self._field_similarity,
            score_normalization=self._score_normalization,
            **self._build_trajectory_fields(),
        )
        return self._storage.search(spec)
