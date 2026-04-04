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

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import torch

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


class SimpleKnnStrategy:
    """Standard KNN search with configurable fusion and step filtering.

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
      - rrf_k: RRF fusion parameter k (backend-specific, passed via backend_hints)
      - fusion_weights: per-field fusion weights (from keys config, backend-agnostic)
      - candidate_multiplier: prefetch limit (backend-specific, passed via backend_hints)
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
    ) -> None:
        self._storage = storage
        self._top_k = top_k
        self._step_filter = step_filter
        self._step_window = step_window
        self._rrf_k = rrf_k
        self._fusion_weights = fusion_weights
        self._candidate_multiplier = candidate_multiplier

    def search(self, ctx: SearchContext) -> list[SearchResultLite]:
        """Execute KNN search with configured fusion parameters.

        Flow:
          1. Build QueryFilter from step_filter + ctx.current_step (if applicable)
          2. Construct QuerySpec with fusion_weights + backend_hints
          3. Call self._storage.search(spec)
          4. Return results (Judge decides hit/miss downstream)
        """
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
        )
        return self._storage.search(spec)

    def _build_filters(self, ctx: SearchContext) -> Optional[QueryFilter]:
        """Build QueryFilter based on step_filter mode and runtime context.

        Data flow: step_filter config + ctx.current_step -> QueryFilter or None
        """
        if self._step_filter == "all":
            return None
        elif self._step_filter == "exact":
            return QueryFilter(step_range=(ctx.current_step, ctx.current_step))
        elif self._step_filter == "window":
            lo = max(0, ctx.current_step - self._step_window)
            hi = ctx.current_step + self._step_window
            return QueryFilter(step_range=(lo, hi))
        else:
            raise ValueError(f"Unknown step_filter: {self._step_filter}")
