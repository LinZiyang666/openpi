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

from openpi.cache.cache_storage import CacheStorage
from openpi.cache.storage_types import (
    QueryFilter,
    QuerySpec,
    RetrievalSignals,
    SearchResultLite,
)
from openpi.cache.types import CheckpointID

logger = logging.getLogger(__name__)


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

    # ------------------------------------------------------------------
    # Depth-aware machinery (TRACER M3 / M2). Shared by DynamicDepthKnn-
    # Strategy and DualRetrievalKnnStrategy. These methods assume the
    # subclass __init__ has set the depth attributes (_depth_policy,
    # _allowed_depths) and the fusion attributes (_base_fusion,
    # _fusion_weights, _rrf_k, _field_similarity, _score_normalization,
    # _step_filter, _step_window); the plain fixed-depth strategies never
    # call them. DepthFeatures / _build_step_filters are module-level names
    # defined below and resolved at call time.
    # ------------------------------------------------------------------

    def _action_smoothness(self) -> Optional[float]:
        """L2 norm between the first actions of the two most recent chunks.

        Returns None when fewer than two actions have been recorded (episode
        start) or either recorded chunk is missing.
        """
        hist = self._action_history
        if len(hist) < 2:
            return None
        a_prev, a_prev2 = hist[-1], hist[-2]
        if a_prev is None or a_prev2 is None:
            return None
        # chunk[0] = first action of the [chunk_len, A] chunk (executed-action
        # convention, matching the verdict-factor action channel).
        v1 = a_prev[0].detach().float()
        v2 = a_prev2[0].detach().float()
        return float(torch.linalg.vector_norm(v1 - v2))

    def _select_effective_depth(self, ctx: SearchContext) -> int:
        """Build DepthFeatures, consult the policy, validate, and history-clamp.

        Raises ValueError (explicit, not ``assert`` — survives ``python -O``) if
        the policy returns a depth outside ``self._allowed_depths``.
        """
        features = DepthFeatures(
            current_step=ctx.current_step,
            history_len=len(self._query_history),
            action_smoothness=self._action_smoothness(),
        )
        t_sel = self._depth_policy.select(features)
        if t_sel not in self._allowed_depths:
            raise ValueError(
                f"DepthPolicy returned depth {t_sel} outside allowed_depths "
                f"{sorted(self._allowed_depths)}"
            )
        # history-availability clamp — same as _build_trajectory_fields.
        return min(t_sel, len(self._query_history))

    def _trajectory_fields_for_depth(self, effective_depth: int) -> dict[str, Any]:
        """Build QuerySpec trajectory fields at a runtime depth.

        Mirrors ``_build_trajectory_fields`` but parameterized by the per-step
        effective depth, using un-renormalized prefix weights so a constant
        policy at the full max depth is value-identical to the fixed-depth
        strategy at every history length.
        """
        if effective_depth <= 1 or not self._trajectory_weights:
            return {}
        history_newest_first = list(reversed(self._query_history[-effective_depth:]))
        weights_newest_first = self._trajectory_weights[:effective_depth]
        fields: dict[str, Any] = {
            "trajectory_history": history_newest_first,
            "trajectory_weights": weights_newest_first,
        }
        if self._search_session_id is not None:
            qids_newest_first = list(reversed(self._query_id_history[-effective_depth:]))
            fields["search_session_id"] = self._search_session_id
            fields["trajectory_query_ids"] = qids_newest_first
        return fields

    def _build_spec_at_depth(
        self,
        ctx: SearchContext,
        effective_depth: int,
        *,
        top_k: int,
        extra_filter_outcome: Optional[int] = None,
    ) -> QuerySpec:
        """Construct a QuerySpec at a runtime depth for the configured fusion.

        ``top_k`` is explicit so a dual-retrieval caller can over-fetch (>=2) to
        compute a positive-ambiguity margin without widening the returned list.
        ``extra_filter_outcome`` (+1 / -1) adds a ``QueryFilter.outcome``
        constraint for pool-partitioned (D+/D-) retrieval; None leaves the
        outcome unconstrained so the spec is byte-identical to the fixed-depth
        path.
        """
        filters = _build_step_filters(self._step_filter, self._step_window, ctx)
        if extra_filter_outcome is not None:
            filters = QueryFilter(
                task_key=filters.task_key if filters else None,
                step_range=filters.step_range if filters else None,
                outcome=extra_filter_outcome,
            )
        traj_fields = self._trajectory_fields_for_depth(effective_depth)
        if self._base_fusion == "weighted_rrf":
            return QuerySpec(
                query_keys=ctx.query_keys,
                top_k=top_k,
                checkpoint_id=ctx.checkpoint_id,
                filters=filters,
                fusion_weights=self._fusion_weights,
                fusion_method="weighted_rrf",
                field_similarity=self._field_similarity,
                backend_hints={"rrf_k": self._rrf_k},
                **traj_fields,
            )
        return QuerySpec(
            query_keys=ctx.query_keys,
            top_k=top_k,
            checkpoint_id=ctx.checkpoint_id,
            filters=filters,
            fusion_weights=self._fusion_weights,
            fusion_method="weighted_score_sum",
            field_similarity=self._field_similarity,
            score_normalization=self._score_normalization,
            **traj_fields,
        )


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

    def last_step_features(self):
        """X15 retrieval diagnostics of this strategy's most recent search.

        Read-through to the per-connection storage facade — this strategy holds
        no diagnostics state of its own, which is what keeps the snapshot tied
        to one connection under a pooled backend.
        """
        getter = getattr(self._storage, "last_step_features", None)
        return None if getter is None else getter()


# ---------------------------------------------------------------------------
# Dynamic chain-depth strategy (TRACER Phase 1 / M3)
# ---------------------------------------------------------------------------


@dataclass
class DepthFeatures:
    """Cheap, pre-search, training-free signals for depth selection.

    Populated by DynamicDepthKnnStrategy before each search. Only carries
    signals available without a failure library or a prior search this step;
    the proposal's failure score and positive-ambiguity margin are deferred to
    a later phase.
    """

    current_step: int
    history_len: int
    action_smoothness: Optional[float] = None


@runtime_checkable
class DepthPolicy(Protocol):
    """Choose a trajectory depth for the current step.

    Contract: `select()` MUST return a value in the strategy's configured
    `allowed_depths` set. `DynamicDepthKnnStrategy` raises `ValueError` on any
    out-of-set return before building a QuerySpec (explicit raise, not
    `assert`, so the guard survives `python -O`).
    """

    def select(self, features: DepthFeatures) -> int:
        """Return the chosen trajectory depth (must be in allowed_depths)."""
        ...


class ConstantDepthPolicy:
    """Always return a fixed depth.

    The non-regression default: paired with the full max depth it reproduces
    the existing fixed-depth strategy at every history length.
    """

    def __init__(self, depth: int) -> None:
        self._depth = int(depth)

    def select(self, features: DepthFeatures) -> int:
        return self._depth


class HeuristicDepthPolicy:
    """Deterministic action-smoothness bucketing.

    Smooth motion (small ``||a_{t-1} - a_{t-2}||``) selects a deeper trajectory
    context; abrupt motion selects a shallower one. When smoothness is
    undefined (fewer than two recorded actions) returns ``fallback_depth``.
    Bucket edges are half-open ``[t_i, t_{i+1})`` with ``>=`` resolving ties;
    the map is total and always returns a member of ``allowed_depths``.
    """

    def __init__(
        self,
        allowed_depths: list[int],
        smoothness_thresholds: list[float],
        fallback_depth: int,
    ) -> None:
        depths = sorted({int(d) for d in allowed_depths})
        if not depths:
            raise ValueError("HeuristicDepthPolicy requires a non-empty allowed_depths")
        thr = [float(t) for t in smoothness_thresholds]
        if len(thr) != len(depths) - 1:
            raise ValueError(
                f"smoothness_thresholds length ({len(thr)}) must equal "
                f"len(allowed_depths)-1 ({len(depths) - 1})"
            )
        if any(thr[i] >= thr[i + 1] for i in range(len(thr) - 1)):
            raise ValueError(
                f"smoothness_thresholds must be strictly ascending, got {thr}"
            )
        if int(fallback_depth) not in depths:
            raise ValueError(
                f"fallback_depth {fallback_depth} must be in allowed_depths {depths}"
            )
        # Descending so bucket index 0 (smoothest) maps to the deepest depth.
        self._depths_desc = sorted(depths, reverse=True)
        self._thresholds = thr
        self._fallback = int(fallback_depth)

    def select(self, features: DepthFeatures) -> int:
        s = features.action_smoothness
        if s is None:
            return self._fallback
        idx = sum(1 for t in self._thresholds if s >= t)
        return self._depths_desc[idx]


class DynamicDepthKnnStrategy(TrajectoryMixin):
    """In-memory strategy that picks trajectory depth per step (TRACER M3).

    Wraps a base fusion (``weighted_rrf`` or ``weighted_score_sum``) and
    consults a `DepthPolicy` to choose an effective depth for each search.
    With a `ConstantDepthPolicy` at the full max depth it is value-identical to
    the existing fixed-depth strategy at every history length: the trajectory
    weights are un-renormalized prefixes, matching `TrajectoryMixin`
    semantics (see the Phase-1 plan non-regression contract).

    `trajectory_depth` is the max depth; `trajectory_weights` the max-depth
    (newest-first) weight vector. `_init_trajectory` wires the history and
    search-session machinery shared with the fixed-depth strategies.
    """

    def __init__(
        self,
        storage: CacheStorage,
        *,
        base_fusion: str,
        depth_policy: DepthPolicy,
        allowed_depths: list[int],
        top_k: int = 1,
        step_filter: str = "all",
        step_window: int = 5,
        fusion_weights: Optional[dict[str, float]] = None,
        rrf_k: int = 60,
        field_similarity: Optional[dict[str, dict[str, Any]]] = None,
        score_normalization: Optional[dict[str, Any]] = None,
        trajectory_depth: int = 1,
        trajectory_weights: Optional[list[float]] = None,
    ) -> None:
        if base_fusion not in ("weighted_rrf", "weighted_score_sum"):
            raise ValueError(
                "DynamicDepthKnnStrategy base_fusion must be 'weighted_rrf' or "
                f"'weighted_score_sum', got {base_fusion!r}"
            )
        self._storage = storage
        self._base_fusion = base_fusion
        self._depth_policy = depth_policy
        self._allowed_depths = {int(d) for d in allowed_depths}
        self._top_k = top_k
        self._step_filter = step_filter
        self._step_window = step_window
        self._fusion_weights = fusion_weights
        self._rrf_k = rrf_k
        self._field_similarity = field_similarity
        self._score_normalization = score_normalization
        self._init_trajectory(trajectory_depth, trajectory_weights)

    def search(self, ctx: SearchContext) -> list[SearchResultLite]:
        self.record_query_keys(ctx.query_keys)
        effective_depth = self._select_effective_depth(ctx)
        # Constant policy at the full max depth reproduces the fixed-depth
        # strategy value-for-value (shared TrajectoryMixin depth machinery).
        spec = self._build_spec_at_depth(ctx, effective_depth, top_k=self._top_k)
        return self._storage.search(spec)


# ---------------------------------------------------------------------------
# Failure-aware dual-retrieval strategy (TRACER Phase 3 / M2)
# ---------------------------------------------------------------------------


class DualRetrievalKnnStrategy(TrajectoryMixin):
    """In-memory dual-pool retrieval strategy (TRACER M2 skeleton).

    Searches a positive pool (D+) and, when enabled, a negative pool (D-),
    computing per-query failure-aware signals (Eq 16-20):

      - ``s_pos`` / ``s_neg`` : top-1 similarity in each pool (0.0 if empty).
      - ``margin = s_pos - lambda_ * s_neg``.
      - ``delta_pos = top1 - top2`` similarity in D+ (positive ambiguity).

    The signals are stashed for ``last_retrieval_signals()``, which the
    Orchestrator forwards to the judge (a ``failure_aware_gate``). Depth is
    selected via the shared TrajectoryMixin depth machinery, reusing the
    Phase-1 DepthPolicy; with a ``ConstantDepthPolicy`` at the full max depth
    and ``enable_dual=False`` (single pool, no outcome filter) the returned
    results are value-identical to the fixed-depth ``base_fusion`` strategy —
    the Phase 3 non-regression anchor.

    The positive pool is always fetched with ``top_k >= 2`` so ``delta_pos`` is
    well defined even when the configured ``top_k`` is 1; the returned list is
    sliced back to ``top_k`` to preserve fixed-depth parity (over-fetching does
    not change the top-1). Pool partitioning uses ``QueryFilter.outcome``
    (+1 / -1), which only the in_memory backend advertises; config validation
    gates this strategy to in_memory.
    """

    def __init__(
        self,
        storage: CacheStorage,
        *,
        base_fusion: str,
        depth_policy: DepthPolicy,
        allowed_depths: list[int],
        lambda_: float = 0.0,
        enable_dual: bool = False,
        top_k: int = 1,
        step_filter: str = "all",
        step_window: int = 5,
        fusion_weights: Optional[dict[str, float]] = None,
        rrf_k: int = 60,
        field_similarity: Optional[dict[str, dict[str, Any]]] = None,
        score_normalization: Optional[dict[str, Any]] = None,
        trajectory_depth: int = 1,
        trajectory_weights: Optional[list[float]] = None,
    ) -> None:
        if base_fusion not in ("weighted_rrf", "weighted_score_sum"):
            raise ValueError(
                "DualRetrievalKnnStrategy base_fusion must be 'weighted_rrf' or "
                f"'weighted_score_sum', got {base_fusion!r}"
            )
        self._storage = storage
        self._base_fusion = base_fusion
        self._depth_policy = depth_policy
        self._allowed_depths = {int(d) for d in allowed_depths}
        self._lambda = float(lambda_)
        self._enable_dual = bool(enable_dual)
        self._top_k = top_k
        self._step_filter = step_filter
        self._step_window = step_window
        self._fusion_weights = fusion_weights
        self._rrf_k = rrf_k
        self._field_similarity = field_similarity
        self._score_normalization = score_normalization
        self._last_signals: Optional[RetrievalSignals] = None
        self._init_trajectory(trajectory_depth, trajectory_weights)

    def search(self, ctx: SearchContext) -> list[SearchResultLite]:
        self.record_query_keys(ctx.query_keys)
        effective_depth = self._select_effective_depth(ctx)

        # Positive pool (D+). Over-fetch to >=2 so delta_pos (top1-top2) is well
        # defined regardless of the configured top_k; slice back before return.
        pos_top_k = max(self._top_k, 2)
        pos_outcome = 1 if self._enable_dual else None
        pos_spec = self._build_spec_at_depth(
            ctx, effective_depth, top_k=pos_top_k, extra_filter_outcome=pos_outcome
        )
        pos_full = self._storage.search(pos_spec)
        s_pos = pos_full[0].score if pos_full else 0.0
        delta_pos = (
            pos_full[0].score - pos_full[1].score if len(pos_full) >= 2 else 0.0
        )

        # Negative pool (D-) only when dual retrieval is enabled; s_neg needs
        # only the top-1. Empty / disabled => s_neg = 0 => margin = s_pos.
        if self._enable_dual:
            neg_spec = self._build_spec_at_depth(
                ctx, effective_depth, top_k=1, extra_filter_outcome=-1
            )
            neg = self._storage.search(neg_spec)
            s_neg = neg[0].score if neg else 0.0
        else:
            s_neg = 0.0

        margin = s_pos - self._lambda * s_neg
        self._last_signals = RetrievalSignals(
            s_pos=s_pos,
            s_neg=s_neg,
            margin=margin,
            delta_pos=delta_pos,
            lambda_=self._lambda,
        )
        return pos_full[: self._top_k]

    def last_retrieval_signals(self) -> Optional[RetrievalSignals]:
        """Return the most recent per-query failure-aware signals (or None)."""
        return self._last_signals
