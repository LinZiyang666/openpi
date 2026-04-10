# Trajectory Search Implementation Plan

**Status**: `Plan`
**Date**: 2026-04-07
**Requirements Document**: [trajectory_search_requirements.log.md](trajectory_search_requirements.log.md)

---

## Overview

This plan breaks down the trajectory search requirements into **7 Phases**, ordered by dependencies. Each Phase is independently testable, and merging does not break existing functionality (`trajectory_depth=1` degenerates to single-step search).

**Dependency Graph:**

```
Phase 1 (Data Structures) ──┬──→ Phase 2 (Backend Trajectory Search)
                             ├──→ Phase 3 (SearchStrategy Upgrade) ──→ Phase 5 (Orchestrator Refactor)
                             ├──→ Phase 4 (Gate/Judge Interface Preparation) ──→ Phase 5
                             └──→ Phase 6 (WritePolicy + Config) ──→ Phase 5
Phase 7 (Ingest Script) depends on Phase 1
```

**Phases that can run in parallel**: Phase 2 / 3 / 4 / 6 / 7 all depend only on Phase 1 and can be developed in parallel. Phase 5 is the final integration point.

---

## Phase 1: Data Structure Changes

**Goal**: Modify dataclass definitions in `storage_types.py` and `config.py` to provide base types for subsequent Phases. Pure type changes, no behavior logic changes.

### 1.1 `src/openpi/cache/storage_types.py`

#### CachePayload: Remove `next_action_chunk`

```python
# Remove this field:
# next_action_chunk: Optional[torch.Tensor] = None   # [50, 32] CPU float32

# Remove CP3 validation in validate_for_checkpoint:
# if checkpoint_id == CheckpointID.CP3 and self.next_action_chunk is None:
#     raise ValueError("next_action_chunk is required for CP3")
```

**Impact scope**: Global search for all references to `next_action_chunk` and clean up:
- `storage_types.py` -- field definition + validate_for_checkpoint
- `orchestrator.py` -- if referenced in `schedule_next_action()`
- `backends/in_memory_backend.py` -- if referenced in artifact loading / serialization
- `backends/qdrant_backend.py` -- if referenced in payload serialization
- Test files -- test cases constructing CachePayload

#### CacheEntry: Add Linked List Fields

```python
@dataclass
class CacheEntry:
    id: str
    checkpoint_id: CheckpointID
    query_keys: dict[str, torch.Tensor]
    payload: CachePayload
    step_idx: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    # ── New ──
    prev_ids: list[str] = field(default_factory=list)
    next_ids: list[str] = field(default_factory=list)
    trajectory_id: Optional[str] = None
```

- All new fields have default values; existing construction code does not need modification
- `validate()` method unchanged (new fields have no CP-level constraints to validate)

#### QuerySpec: Add Trajectory Fields

```python
@dataclass
class QuerySpec:
    # ... existing fields unchanged ...

    # ── New ──
    trajectory_history: Optional[list[dict[str, torch.Tensor]]] = None
    # Query-side history key sequence, newest-first: [current step keys, t-1 keys, t-2 keys, ...]
    # When None, degenerates to existing single-step search behavior

    trajectory_weights: Optional[list[float]] = None
    # Per-layer weights, newest-first: [w_current, w_t-1, w_t-2, ...]
    # Must be same length as trajectory_history; when None, no trajectory fusion
```

#### New StepRecord / EpisodeRecord

```python
@dataclass
class StepRecord:
    """Per-step temporary record, used to construct CacheEntry chains at episode end."""
    query_keys: dict[str, torch.Tensor]   # CPU float32
    action_chunk: torch.Tensor            # CPU float32, required (steps without action are not written)

@dataclass
class EpisodeRecord:
    """Entire episode temporary record, passed to WritePolicy to decide whether to write."""
    steps: list[StepRecord]
    task_key: str
    miss_by_checkpoint: dict[CheckpointID, int]  # e.g. {CP1: 3, CP3: 50}
    total_steps: int
```

### 1.2 `src/openpi/cache/config.py`

#### SearchStrategyConfig Add Fields

```python
@dataclass
class SearchStrategyConfig:
    # ... existing fields unchanged ...
    # ── New ──
    trajectory_depth: int = 1
    trajectory_weights: Optional[list[float]] = None
```

#### New WritePolicyConfig

```python
@dataclass
class WritePolicyConfig:
    type: str = "on_any_miss"   # on_any_miss | always | never
```

#### CacheConfig Add Fields

```python
@dataclass
class CacheConfig:
    # ... existing fields unchanged ...
    # ── New ──
    write_policy: WritePolicyConfig = field(default_factory=WritePolicyConfig)
```

#### Registry Update

```python
_CONFIG_TYPES["WritePolicyConfig"] = WritePolicyConfig
```

#### validate_cache_config Add Validation

```python
# Append at the end of validate_cache_config():

# Trajectory validation
for cp_name, cp_cfg in config.checkpoints.items():
    ss = cp_cfg.search_strategy

    # trajectory_depth >= 1
    if ss.trajectory_depth < 1:
        raise ConfigValidationError(
            f"checkpoints.{cp_name}.search_strategy: "
            f"trajectory_depth must be >= 1, got {ss.trajectory_depth}"
        )

    if ss.trajectory_depth > 1:
        if ss.trajectory_weights is None:
            raise ConfigValidationError(
                f"checkpoints.{cp_name}.search_strategy: "
                f"trajectory_weights required when trajectory_depth={ss.trajectory_depth}"
            )
        if len(ss.trajectory_weights) != ss.trajectory_depth:
            raise ConfigValidationError(
                f"checkpoints.{cp_name}.search_strategy: "
                f"trajectory_weights length ({len(ss.trajectory_weights)}) "
                f"!= trajectory_depth ({ss.trajectory_depth})"
            )
        # weights non-negative and sum > 0
        if any(w < 0 for w in ss.trajectory_weights):
            raise ConfigValidationError(
                f"checkpoints.{cp_name}.search_strategy: "
                f"trajectory_weights must be non-negative"
            )
        if sum(ss.trajectory_weights) <= 0:
            raise ConfigValidationError(
                f"checkpoints.{cp_name}.search_strategy: "
                f"trajectory_weights sum must be > 0"
            )

    # Qdrant + trajectory_depth > 1 → fail-fast at config stage
    # backend is a top-level CacheConfig.backend field, not a CheckpointConfig field
    if ss.trajectory_depth > 1 and config.backend.type == "qdrant":
        raise ConfigValidationError(
            f"checkpoints.{cp_name}: trajectory_depth > 1 is not supported "
            f"with Qdrant backend. Use InMemoryBackend or set trajectory_depth=1."
        )

# write_policy.type validation
_VALID_WRITE_POLICY_TYPES = {"on_any_miss", "always", "never"}
if config.write_policy.type not in _VALID_WRITE_POLICY_TYPES:
    raise ConfigValidationError(
        f"write_policy.type '{config.write_policy.type}' unknown, "
        f"valid: {_VALID_WRITE_POLICY_TYPES}"
    )
```

### 1.3 Acceptance Criteria

- `uv run pytest` all pass (pure new fields + defaults, does not break existing tests)
- Global search for `next_action_chunk`; all references cleaned up or updated
- Existing YAML files do not need modification (new fields all have defaults)

---

## Phase 2: InMemoryBackend Trajectory Search

**Goal**: Implement trajectory search in `InMemoryBackend.search()`. Reuse existing per-step fusion (RRF / score_sum), adding cross-step trajectory fusion on top. Pure backend-layer change; no upper-layer components involved.

### 2.1 `src/openpi/cache/backends/in_memory_backend.py`

#### search() Entry Dispatch

The existing `search()` method dispatches by `fusion_method`. Add trajectory search branch:

```python
def search(self, spec: QuerySpec) -> list[SearchResultLite]:
    candidates = self._filter_entries(spec)
    if not candidates:
        return []

    active_fields = self._iter_active_fields(spec)
    if not active_fields:
        return []

    # ── Trajectory search ──
    if (spec.trajectory_history is not None
            and spec.trajectory_weights is not None
            and len(spec.trajectory_weights) > 1):
        return self._search_with_trajectory(candidates, spec, active_fields)

    # ── Existing single-step search (unchanged) ──
    if spec.fusion_method == "weighted_rrf":
        return self._search_weighted_rrf(candidates, spec, active_fields)
    elif spec.fusion_method == "weighted_score_sum":
        return self._search_weighted_score_sum(candidates, spec, active_fields)
    else:
        return self._search_single_field_cosine(candidates, spec)
```

#### New `_search_with_trajectory()`: Two-Pass Recursion

```python
def _search_with_trajectory(
    self,
    candidates: list[CacheEntry],
    spec: QuerySpec,
    active_fields: list[tuple[str, float, dict[str, Any]]],
) -> list[SearchResultLite]:
    """Trajectory search: two-pass recursion + batch scoring in between.

    Fully reuses existing per-step fusion (RRF / score_sum), adding
    cross-step weighted fusion on top.

    Flow:
      Phase A -- First recursion (collection):
        Starting from window-filtered candidates, traverse prev_ids backward
        up to depth layers, collecting the set of entries to score at each layer
        (deduplicated).

      Phase B -- Per-layer batch scoring:
        For each layer's entry set, batch-compute per-step scores using the
        configured fusion_method (RRF / score_sum). RRF's ranking scope =
        the entry set collected at that layer.
        Results stored in level_scores[depth][entry_id] = step_score.

      Phase C -- Second recursion (scoring):
        Traverse the same paths again, look up precomputed scores from
        level_scores, compute weighted sums per trajectory_weights, handle
        branching (take max).

    No normalization: candidates with broken chains naturally score lower;
    this is the correct signal.

    RRF Semantics Note
    -------------------
    When fusion_method="weighted_score_sum", two-pass recursion is mathematically
    equivalent to per-path recursion.
    When fusion_method="weighted_rrf", each layer's RRF ranking scope is "all
    entries reachable from current candidates at that layer" (per-level
    reachable-set RRF score), not an absolute similarity for a single entry.
    This is a reasonable extended definition but differs semantically from
    original single-step RRF; separate testing is needed.
    """
    history = spec.trajectory_history   # newest-first
    weights = spec.trajectory_weights   # newest-first
    max_depth = len(weights) - 1

    # ── Phase A: First recursion, collect entries to score at each layer ──
    # level_entries[idx] = set of entry_ids at that depth level
    level_entries: list[set[str]] = [set() for _ in range(len(weights))]
    for entry in candidates:
        self._collect_trajectory_entries(
            entry_id=entry.id,
            depth=max_depth,
            max_depth=max_depth,
            level_entries=level_entries,
            query_history_len=len(history),
            expected_checkpoint_id=spec.checkpoint_id,
        )

    # ── Phase B: Per-layer batch scoring ──
    # level_scores[idx][entry_id] = per-step score
    level_scores: list[dict[str, float]] = []
    for idx in range(len(weights)):
        entry_ids = level_entries[idx]
        if not entry_ids:
            level_scores.append({})
            continue

        entries_at_level = [self._entries[eid] for eid in entry_ids if eid in self._entries]
        if not entries_at_level:
            level_scores.append({})
            continue

        # Batch score using configured fusion_method (reuse existing logic)
        # active_fields are recomputed inside _batch_step_scores per current layer query_keys
        scores = self._batch_step_scores(entries_at_level, history[idx], spec)
        level_scores.append(scores)

    # ── Phase C: Second recursion, look up precomputed scores for trajectory scoring ──
    scored: list[tuple[CacheEntry, float]] = []
    for entry in candidates:
        path_scores = self._score_trajectory(
            entry_id=entry.id,
            depth=max_depth,
            max_depth=max_depth,
            weights=weights,
            accumulated_sim=0.0,
            level_scores=level_scores,
            query_history_len=len(history),
        )
        traj_score = max(path_scores) if path_scores else 0.0
        scored.append((entry, traj_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [
        SearchResultLite(id=e.id, score=s, checkpoint_id=e.checkpoint_id)
        for e, s in scored[:spec.top_k]
    ]
```

#### New `_collect_trajectory_entries()` (First Recursion)

```python
def _collect_trajectory_entries(
    self,
    entry_id: str,
    depth: int,
    max_depth: int,
    level_entries: list[set[str]],
    query_history_len: int,
    expected_checkpoint_id: CheckpointID | None = None,
) -> None:
    """First recursion: collect entry ids to score at each layer.

    checkpoint_id consistency check: skip prev entries with mismatched
    checkpoint_id, preventing bad data or pruning/merging from contaminating
    trajectory scores with cross-checkpoint pointers.
    """
    entry = self._entries.get(entry_id)
    if entry is None:
        return

    # Skip entries with mismatched checkpoint (defensive check)
    if expected_checkpoint_id is not None and entry.checkpoint_id != expected_checkpoint_id:
        return

    idx = max_depth - depth
    if idx >= query_history_len:
        return

    level_entries[idx].add(entry_id)

    if depth == 0 or not entry.prev_ids:
        return

    for prev_id in entry.prev_ids:
        self._collect_trajectory_entries(
            prev_id, depth - 1, max_depth,
            level_entries, query_history_len,
            expected_checkpoint_id,
        )
```

#### New `_batch_step_scores()` (Per-Layer Batch Scoring)

```python
def _batch_step_scores(
    self,
    entries: list[CacheEntry],
    query_keys: dict[str, torch.Tensor],
    spec: QuerySpec,
) -> dict[str, float]:
    """Batch-compute per-step scores for a layer's entry set.

    Reuses existing per-step fusion logic (RRF / score_sum / single cosine);
    ranking scope = the passed entries set.

    active_fields are recomputed per current layer's query_keys (historical
    queries may lack certain fields), ensuring consistency with existing
    single-step search semantics.

    Returns {entry_id: step_score}.
    """
    if not entries:
        return {}

    # Construct a temporary QuerySpec, only replacing query_keys with current layer's query
    temp_spec = QuerySpec(
        query_keys=query_keys,
        top_k=len(entries),  # Return all, no truncation
        checkpoint_id=spec.checkpoint_id,
        fusion_weights=spec.fusion_weights,
        fusion_method=spec.fusion_method,
        field_similarity=spec.field_similarity,
        score_normalization=spec.score_normalization,
        backend_hints=spec.backend_hints,
        # trajectory fields not passed, uses single-step logic
    )

    # Recompute active_fields per current layer's query_keys
    # Historical queries may lack certain fields; cannot reuse outer active_fields
    level_active_fields = self._iter_active_fields(temp_spec)
    if not level_active_fields:
        return {}

    # Reuse existing single-step search methods
    if spec.fusion_method == "weighted_rrf":
        results = self._search_weighted_rrf(entries, temp_spec, level_active_fields)
    elif spec.fusion_method == "weighted_score_sum":
        results = self._search_weighted_score_sum(entries, temp_spec, level_active_fields)
    else:
        results = self._search_single_field_cosine(entries, temp_spec)

    return {r.id: r.score for r in results}
```

#### New `_score_trajectory()` (Second Recursion)

```python
def _score_trajectory(
    self,
    entry_id: str,
    depth: int,
    max_depth: int,
    weights: list[float],
    accumulated_sim: float,
    level_scores: list[dict[str, float]],
    query_history_len: int,
) -> list[float]:
    """Second recursion: look up precomputed scores and compute weighted sum.

    Index mapping: idx = max_depth - depth
    Early termination when idx >= query_history_len (consistent with first pass
    _collect), avoiding meaningless extra branches when history is insufficient.
    Returns list of non-normalized trajectory_sim for all complete paths.
    """
    entry = self._entries.get(entry_id)
    if entry is None:
        return [accumulated_sim]

    idx = max_depth - depth
    if idx >= query_history_len:
        return [accumulated_sim]
    if idx >= len(level_scores):
        return [accumulated_sim]

    step_score = level_scores[idx].get(entry_id, 0.0)
    accumulated_sim += weights[idx] * step_score

    if depth == 0 or not entry.prev_ids:
        return [accumulated_sim]

    all_paths = []
    for prev_id in entry.prev_ids:
        all_paths.extend(self._score_trajectory(
            prev_id, depth - 1, max_depth,
            weights, accumulated_sim, level_scores,
            query_history_len,
        ))
    return all_paths
```

#### QdrantBackend Guard

```python
# src/openpi/cache/backends/qdrant_backend.py
# Prepend at the beginning of search():

if (spec.trajectory_history is not None
        and spec.trajectory_weights is not None
        and len(spec.trajectory_weights) > 1):
    raise NotImplementedError(
        "Trajectory search (trajectory_depth > 1) is not supported in QdrantBackend. "
        "Use InMemoryBackend for trajectory search, or set trajectory_depth=1."
    )
```

#### VectorStoreBackend ABC

```python
# src/openpi/cache/backend_base.py
# Append to search() docstring:

"""
...
Trajectory search
-----------------
If spec.trajectory_history is not None and len(spec.trajectory_weights) > 1,
the backend should compute trajectory-aware similarity for all filtered
candidates in a single pass — no two-phase "initial search + rerank".
Backends that do not support this MUST raise NotImplementedError.
The recursive traversal follows entry.prev_ids to compute cross-step
similarity fusion (see trajectory_search_requirements.log.md section 2.3).
"""
```

### 2.2 Acceptance Criteria

- Unit tests: construct CacheEntry chains with `prev_ids`/`next_ids` and verify:
  - `trajectory_history=None` -> behavior identical to pre-modification
  - `trajectory_depth=1, weights=[1.0]` -> results identical to single-step search
  - `trajectory_depth=3` -> correct backtracking, correct weighting, correct ranking
  - Broken chain (`prev_ids=[]`) -> graceful termination, no errors
  - `query_history` length insufficient -> graceful termination, score equals weighted sum using only available history, no extra branches
  - **Rank reversal test**: construct A with highest single-step score but poor trajectory, B with second-highest single-step but good trajectory -> after trajectory search, B ranks first
  - **Chain break penalty test**: same query, complete chain candidate scores > broken chain candidate score (broken chains are naturally penalized; no normalization)
  - **Historical query missing a field**: certain layer query_keys has only vision_0 without robot_state -> no error, scores using remaining fields
  - **Checkpoint consistency**: prev_ids point to entries with different checkpoints -> skipped, do not participate in scoring
  - **RRF trajectory search**: `fusion_method="weighted_rrf"` + `trajectory_depth=3` -> correct ranking (per-level reachable-set RRF semantics)
  - **score_sum vs RRF semantic difference**: same dataset, compare trajectory scores of both fusion_methods, verify score_sum two-pass recursion is equivalent to per-path recursion
- QdrantBackend test: `trajectory_depth > 1` raises `NotImplementedError`

---

## Phase 3: SearchStrategy Upgrade

**Goal**: Add history buffer maintenance and `trajectory_history`/`trajectory_weights` population in QuerySpec for all existing SearchStrategy implementation classes.

### 3.1 Extract Common Mixin: `TrajectoryMixin`

To avoid repeating the same buffer logic in three strategy classes, extract a mixin class:

```python
# src/openpi/cache/components/search_strategy.py new addition

class TrajectoryMixin:
    """Common trajectory buffer logic, mixed into each SearchStrategy implementation class.

    Provides:
      - history buffer management (query_keys + action_chunk)
      - on_episode_start() lifecycle
      - record_action() broadcast reception
      - _build_trajectory_fields() for constructing QuerySpec trajectory fields
    """

    def _init_trajectory(self, trajectory_depth: int, trajectory_weights: Optional[list[float]]) -> None:
        """Call at the end of strategy class __init__."""
        self._trajectory_depth = trajectory_depth
        self._trajectory_weights = trajectory_weights
        self._query_history: list[dict[str, torch.Tensor]] = []
        self._action_history: list[Optional[torch.Tensor]] = []

    def on_episode_start(self) -> None:
        """Clear history buffer. Called by Orchestrator at episode start."""
        self._query_history.clear()
        self._action_history.clear()

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """Receive Orchestrator-broadcast action.

        Implementation constraint: pure local buffer operation (append to list).
        Prohibited from calling back to Backend / CacheStorage / Orchestrator
        or acquiring any external locks.
        """
        self._action_history.append(action_chunk)

    def record_query_keys(self, query_keys: dict[str, torch.Tensor]) -> None:
        """Store current step's query_keys into trajectory history.

        Call timing:
          - Automatically called inside search() (normal search path)
          - Explicitly called by Orchestrator on gate skip (ensures trajectory history completeness)

        Implementation constraint: pure local buffer operation; prohibited from calling back to external components.
        """
        self._query_history.append(query_keys)

    def _build_trajectory_fields(self) -> dict[str, Any]:
        """Return trajectory fields to fill into QuerySpec.

        When depth=1 or history is empty, returns empty dict (QuerySpec fields stay None, degenerates to single-step).
        """
        if self._trajectory_depth <= 1 or not self._trajectory_weights:
            return {}

        # query_history is appended oldest-first; needs reversal to newest-first
        actual_depth = min(self._trajectory_depth, len(self._query_history))
        if actual_depth <= 1:
            return {}

        history_newest_first = list(reversed(self._query_history[-actual_depth:]))
        weights_newest_first = self._trajectory_weights[:actual_depth]

        return {
            "trajectory_history": history_newest_first,
            "trajectory_weights": weights_newest_first,
        }
```

### 3.2 Modifications to Each Strategy Class

Using `WeightedScoreSumKnnStrategy` as an example (the other two follow the same pattern):

```python
class WeightedScoreSumKnnStrategy(TrajectoryMixin):
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
        # ── New ──
        trajectory_depth: int = 1,
        trajectory_weights: Optional[list[float]] = None,
    ) -> None:
        # ... existing initialization unchanged ...
        self._init_trajectory(trajectory_depth, trajectory_weights)

    def search(self, ctx: SearchContext) -> list[SearchResultLite]:
        # ── New: record current step query_keys ──
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
            # ── New: fill in trajectory fields ──
            **self._build_trajectory_fields(),
        )
        return self._storage.search(spec)
```

**Apply the same modifications to `WeightedRrfKnnStrategy` and `QdrantWeightedRrfKnnStrategy`**:
1. Inherit `TrajectoryMixin`
2. `__init__` adds `trajectory_depth` / `trajectory_weights` parameters; call `_init_trajectory()` at the end
3. `search()` calls `self.record_query_keys(ctx.query_keys)` at the start
4. Expand trajectory fields with `**self._build_trajectory_fields()` when constructing `QuerySpec`

### 3.3 `_build_search_strategy()` Factory Function Update

```python
# In src/openpi/cache/config.py _build_search_strategy(),
# pass new parameters when constructing each strategy instance:

def _build_search_strategy(cfg: SearchStrategyConfig, storage, fusion_weights):
    common_kwargs = {
        # ... existing parameters ...
        "trajectory_depth": cfg.trajectory_depth,
        "trajectory_weights": cfg.trajectory_weights,
    }
    if cfg.type == "weighted_score_sum_knn":
        return WeightedScoreSumKnnStrategy(storage, **common_kwargs, ...)
    elif cfg.type == "weighted_rrf_knn":
        return WeightedRrfKnnStrategy(storage, **common_kwargs, ...)
    # ...
```

### 3.4 Acceptance Criteria

- Unit tests:
  - `trajectory_depth=1` (default) -> `_build_trajectory_fields()` returns empty dict -> QuerySpec has no trajectory fields -> behavior completely unchanged
  - `trajectory_depth=3` -> first two steps `_build_trajectory_fields()` returns empty dict (insufficient history) -> from the third step onward returns correct history and weights
  - `on_episode_start()` -> buffer cleared
  - `record_action()` -> action correctly appended
- Integration test: combined testing with Phase 2's Backend for full trajectory search

---

## Phase 4: Gate / Judge Interface Preparation

**Goal**: Add `on_episode_start()` and `record_action()` methods (no-op implementations) to Gate and Judge concrete implementation classes, with detailed comments explaining future extension approaches. **Do not modify Protocol definitions**.

### 4.1 `src/openpi/cache/components/gate.py`

```python
class AlwaysSearchGate:
    """Gate that always permits cache search."""

    def __call__(self, checkpoint_id: CheckpointID, cached_data: dict[str, torch.Tensor]) -> bool:
        return True

    # ── New: trajectory-aware preparation interfaces ──

    def on_episode_start(self) -> None:
        """Clear internal history buffer. Called by Orchestrator at episode start.

        Current implementation: no-op. AlwaysSearchGate does not maintain historical state.

        Future extension: trajectory-aware gate can clear cached_data history buffer here,
        for detecting temporal consistency (e.g., only trigger search when observation
        change across consecutive steps exceeds a threshold).
        When extending, clear self._cached_data_history in this method.
        """
        pass

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """Receive Orchestrator-broadcast action.

        Implementation constraint: pure local buffer operation; prohibited from calling
        back to Backend / CacheStorage / Orchestrator or acquiring any external locks.

        Current implementation: no-op. AlwaysSearchGate does not use action data.

        Future extension: trajectory-aware gate can store action history for
        action-conditioned gating (e.g., force cache miss when action drift is detected).
        """
        pass
```

### 4.2 `src/openpi/cache/components/judge.py`

Apply the same treatment to `AlwaysHitJudge` and `ThresholdJudge`:

```python
class ThresholdJudge:
    # ... existing __init__ and __call__ unchanged ...

    def on_episode_start(self) -> None:
        """Clear internal history buffer. Called by Orchestrator at episode start.

        Current implementation: no-op. ThresholdJudge is based on single-step
        threshold judgment and does not maintain historical state.

        Future extension: trajectory-aware judge can clear history here for
        temporal consistency verification (e.g., judge as miss when hit scores
        show a declining trend across consecutive steps).
        """
        pass

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """Receive Orchestrator-broadcast action.

        Implementation constraint: pure local buffer operation; prohibited from calling
        back to Backend / CacheStorage / Orchestrator or acquiring any external locks.

        Current implementation: no-op. ThresholdJudge does not use action data.

        Future extension: trajectory-aware judge can store action history for
        action consistency verification (e.g., judge as miss when candidate entry's
        historical actions deviate significantly from query-side historical actions).
        """
        pass
```

### 4.3 Acceptance Criteria

- All existing tests pass (purely new methods added; no existing behavior changed)
- Every prepared method has a detailed docstring: current behavior, implementation constraints, future extension approach

---

## Phase 5: Orchestrator Refactor

**Goal**: Refactor `CacheOrchestrator`, adding action broadcast, episode lifecycle management, and changing the write flow from per-step insert to unified write at episode end.

**Dependencies**: Phase 3 (SearchStrategy has `on_episode_start()`/`record_action()`), Phase 4 (Gate/Judge same), Phase 6 (WritePolicy).

### 5.1 `src/openpi/cache/orchestrator.py`

#### `__init__` New Parameters

```python
class CacheOrchestrator:
    def __init__(
        self,
        storage: CacheStorage,
        key_builder: QueryKeyBuilder,
        gates: dict[CheckpointID, GateFunction],
        judges: dict[CheckpointID, SimilarityJudge],
        search_strategies: dict[CheckpointID, SearchStrategy],
        timer: Optional[SystemTimer] = None,
        # ── New ──
        write_policy: Optional[WritePolicy] = None,
    ) -> None:
        # ... existing initialization ...
        self._write_policy = write_policy

        # ── New: episode-level buffer ──
        self._episode_steps: list[StepRecord] = []
        self._miss_by_checkpoint: dict[CheckpointID, int] = {}
        self._current_task_key: str = ""
```

#### `on_task_begin()` Extension

```python
def on_task_begin(self, task_key: str = "") -> None:
    self._step_counter = 0
    self._current_task_key = task_key
    self._reset_episode_buffer()
    # Notify all components
    self._broadcast_episode_start()
```

#### `on_episode_start()` Extension

```python
def on_episode_start(self, task_key: str = "", episode_id: str = "") -> None:
    self._step_counter = 0
    if task_key:
        self._current_task_key = task_key
    self._current_episode_id = episode_id
    self._reset_episode_buffer()
    self._broadcast_episode_start()
```

#### New `_broadcast_episode_start()`

```python
def _broadcast_episode_start(self) -> None:
    """Notify all components that an episode has started, clearing their history buffers."""
    for strategy in self._search_strategies.values():
        if hasattr(strategy, 'on_episode_start'):
            strategy.on_episode_start()
    for gate in self._gates.values():
        if hasattr(gate, 'on_episode_start'):
            gate.on_episode_start()
    for judge in self._judges.values():
        if hasattr(judge, 'on_episode_start'):
            judge.on_episode_start()
```

#### New `broadcast_action()`

```python
def broadcast_action(self, action_chunk: torch.Tensor) -> None:
    """Broadcast action to all components.

    Called by the external caller after obtaining an action (whether from cache hit
    or model inference). Must be called after check() returns (all locks released;
    no deadlock risk).
    """
    for strategy in self._search_strategies.values():
        if hasattr(strategy, 'record_action'):
            strategy.record_action(action_chunk)
    for gate in self._gates.values():
        if hasattr(gate, 'record_action'):
            gate.record_action(action_chunk)
    for judge in self._judges.values():
        if hasattr(judge, 'record_action'):
            judge.record_action(action_chunk)
```

#### `check()` Modification: Inference Pipeline Adjustment + Miss Recording

**Warning: Key change: `build()` moved up before the gate skip decision.**

Existing order: `collect() -> gate() -> [return on skip] -> build() -> search() -> judge()`
Changed to: `collect() -> gate() -> build() (always executed) -> [record_query_keys + return on skip] -> search() -> judge()`

This way, when gate skips there are still `query_keys` available, which can be recorded in SearchStrategy history, ensuring trajectory search has no gaps. `build()` is just CPU reduction; its cost is far less than search. Write buffering does not happen in check(); the Interceptor calls `buffer_for_write()` after the action is produced.

```python
def check(self, checkpoint_id: CheckpointID, **stage_outputs) -> CheckResult:
    # ... collect ...
    # ... gate ...

    # ── Change: build() always executes, no longer skipped by gate skip ──
    with self._timer.measure(f"{prefix}_build"):
        query_keys = self._key_builder.build(checkpoint_id)

    if not gate_pass:
        # On gate skip: record query_keys to SearchStrategy history (trajectory search completeness)
        # Do not call buffer_for_write() (action not yet produced; Interceptor calls after action is produced)
        strategy = self._search_strategies.get(checkpoint_id)
        if strategy and hasattr(strategy, 'record_query_keys'):
            strategy.record_query_keys(query_keys)
        self._miss_by_checkpoint[checkpoint_id] = self._miss_by_checkpoint.get(checkpoint_id, 0) + 1
        if checkpoint_id == CheckpointID.CP1:
            self._step_counter += 1
        return CheckResult(hit_type=HitType.MISS, query_keys=query_keys)
        # ^ Returns query_keys for Interceptor to later call buffer_for_write(query_keys, action)

    # ── search: pass task_key when constructing SearchContext ──
    context = SearchContext(
        checkpoint_id=checkpoint_id,
        query_keys=query_keys,
        task_key=self._current_task_key,  # <- New: ensures task filter takes effect
    )
    # ... strategy.build_query_spec(context) -> spec -> storage.search(spec) -> results ...
    # ... judge(results) -> hit_type ...

    if hit_type == HitType.MISS:
        self._miss_by_checkpoint[checkpoint_id] = self._miss_by_checkpoint.get(checkpoint_id, 0) + 1

    result.query_keys = query_keys  # Filled on all paths, including FULL_HIT
    return result
```

#### CheckResult Dataclass Change

```python
@dataclass
class CheckResult:
    hit_type: HitType
    # ... existing fields unchanged ...
    # ── New ──
    query_keys: Optional[dict[str, torch.Tensor]] = None
    # Filled on all return paths (gate skip / miss / FULL_HIT).
    # For Interceptor to uniformly call buffer_for_write(query_keys, action).
```

#### New `buffer_for_write()`

```python
def buffer_for_write(
    self,
    query_keys: dict[str, torch.Tensor],
    action_chunk: torch.Tensor,
) -> None:
    """Store current step data to episode buffer; unified write at episode end.

    Called by Interceptor after action is produced (whether from cache hit or model inference).
    Not called in check() -- action not yet produced at that point.

    Both parameters are required:
      - query_keys: obtained from check()'s returned CheckResult.query_keys
      - action_chunk: obtained from cache hit payload or model inference result
    """
    self._episode_steps.append(StepRecord(
        query_keys=query_keys,
        action_chunk=action_chunk,
    ))
```

#### New `on_episode_end()`

```python
def on_episode_end(self) -> None:
    """Called when episode ends. Decides whether to write buffered data to cache based on WritePolicy.

    Write flow:
      1. Construct EpisodeRecord
      2. Call WritePolicy.should_write() to decide
      3. If writing: construct buffered StepRecord list into a linked CacheEntry chain
      4. batch_insert in one call
    """
    if not self._episode_steps:
        return

    if self._write_policy is None:
        return

    record = EpisodeRecord(
        steps=self._episode_steps,
        task_key=self._current_task_key,
        miss_by_checkpoint=dict(self._miss_by_checkpoint),
        total_steps=len(self._episode_steps),
    )

    if not self._write_policy.should_write(record):
        self._reset_episode_buffer()
        return

    # Construct CacheEntry chain
    entries = self._build_entry_chain(record)
    if entries:
        self._storage.batch_insert(entries)

    self._reset_episode_buffer()
```

#### New `_build_entry_chain()`

```python
def _build_entry_chain(self, record: EpisodeRecord) -> list[CacheEntry]:
    """Convert EpisodeRecord into a linked list of CacheEntries."""
    import uuid

    trajectory_id = str(uuid.uuid4())
    entries: list[CacheEntry] = []

    # First pass: create all entries (don't set prev_ids / next_ids yet)
    for step_idx, step in enumerate(record.steps):
        entry_id = f"{trajectory_id}:{step_idx}"  # unique id, not using _stable_hash
        entry = CacheEntry(
            id=entry_id,
            checkpoint_id=CheckpointID.CP1,
            query_keys=step.query_keys,
            payload=CachePayload(
                action_chunk=step.action_chunk,
                task_key=record.task_key,
            ),
            step_idx=step_idx,
            trajectory_id=trajectory_id,
        )
        entries.append(entry)

    # Second pass: link the chain
    for i in range(len(entries)):
        if i > 0:
            entries[i].prev_ids = [entries[i - 1].id]
        if i < len(entries) - 1:
            entries[i].next_ids = [entries[i + 1].id]

    return entries
```

#### New `_reset_episode_buffer()`

```python
def _reset_episode_buffer(self) -> None:
    self._episode_steps.clear()
    self._miss_by_checkpoint.clear()
```

#### Remove Old `write()` / `write_with_keys()`

**Delete directly**, do not preserve old per-step write path. Rationale:
- Phase 5b will migrate all callers from the interceptor to the new path
- Preserving old interface would produce entries without linked lists; mixing them in degrades trajectory search effectiveness
- There are no external callers that need compatibility (write path is entirely inside the interceptor)

Also delete auxiliary logic only used by the old `write()` (e.g., `_stable_hash()` if no longer needed).

### 5.2 `on_task_begin()` Signature Change

Existing `on_task_begin()` has no parameters. Add optional `task_key` parameter:

```python
def on_task_begin(self, task_key: str = "") -> None:
```

Callers that don't pass `task_key` see unchanged behavior (`task_key=""`).

### 5.3 Acceptance Criteria

- Unit tests:
  - `on_episode_start()` -> all components' `on_episode_start()` called
  - `broadcast_action()` -> all components' `record_action()` called
  - `buffer_for_write()` + `on_episode_end()` -> correctly constructs CacheEntry chain + batch_insert
  - WritePolicy `never` -> no write
  - WritePolicy `on_any_miss`, no miss -> no write
  - WritePolicy `on_any_miss`, has miss -> writes
  - Generated CacheEntry chain's `prev_ids`/`next_ids` are correct
- Existing `write()` / `write_with_keys()` tests **migrated** to `buffer_for_write()` + `on_episode_end()` new path

### 5.4 Phase 5b: Interceptor + serve_policy Integration

**Goal**: Migrate Interceptor and serve_policy.py to the new episode-level write path, replacing the old per-step `_bg_write()`.

#### `src/openpi/cache/interceptor.py` Modifications

1. **`on_episode_start(experiment, task, episode_id)`**:
   - Call `orchestrator.on_episode_start(task_key=task, episode_id=episode_id)`
   - task parameter needs normalization (following `CachePayload.task_key` convention)

2. **CP1 cache hit early-return path**:
   - `check()` returns `FULL_HIT`, `CheckResult.query_keys` is not None
   - Call `orchestrator.broadcast_action(cached_action_chunk)`
   - Call `orchestrator.buffer_for_write(result.query_keys, cached_action_chunk)`
   - Return cached action, skip subsequent Stage 2/3

3. **Normal inference path (CP1 miss)**:
   - After Stage 3 inference completes, obtain `action_chunk`
   - Call `orchestrator.broadcast_action(action_chunk)`
   - Call `orchestrator.buffer_for_write(result.query_keys, action_chunk)`
   - **Delete or bypass** old `_bg_write()` per-step write logic

4. **`on_episode_end(success)`**:
   - Call `orchestrator.on_episode_end()`
   - WritePolicy judges inside Orchestrator whether to write

#### `scripts/serve_policy.py` Modifications

- Pass `write_policy=components["write_policy"]` when constructing `CacheOrchestrator`

#### 5b Acceptance Criteria

- End-to-end integration tests:
  - CP1 miss -> normal inference -> `buffer_for_write()` -> `on_episode_end()` -> batch_insert forms chain
  - CP1 hit early-return -> still `broadcast_action()` + `buffer_for_write()`
  - `on_episode_start(task_key=...)` results in non-empty payload.task_key on write
- Old `_bg_write()` path no longer called

---

## Phase 6: WritePolicy Implementation + Config Factory

**Goal**: Implement WritePolicy Protocol and three concrete strategies, plus the factory function in config.py.

### 6.1 New `src/openpi/cache/components/write_policy.py`

```python
"""Write policy: pluggable switch to decide whether to write trajectory at episode end.

Coupling map:
  DEPENDS ON:  storage_types.py (EpisodeRecord)
  CONSUMED BY: CacheOrchestrator.on_episode_end()
  IF CHANGED:  Orchestrator write decision logic
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from openpi.cache.storage_types import EpisodeRecord
from openpi.cache.types import CheckpointID


@runtime_checkable
class WritePolicy(Protocol):
    """Decides whether to write the episode trajectory to cache at episode end."""

    def should_write(self, episode_record: EpisodeRecord) -> bool: ...


class OnAnyMissWritePolicy:
    """Write if the episode had any cache miss or gate skip.

    Rationale: episodes with zero misses are fully covered by existing cache;
    only episodes with novel steps benefit from being stored.
    """

    def should_write(self, episode_record: EpisodeRecord) -> bool:
        return episode_record.miss_by_checkpoint.get(CheckpointID.CP1, 0) > 0


class AlwaysWritePolicy:
    """Always write, regardless of cache hit/miss ratio."""

    def should_write(self, episode_record: EpisodeRecord) -> bool:
        return True


class NeverWritePolicy:
    """Never write (read-only mode)."""

    def should_write(self, episode_record: EpisodeRecord) -> bool:
        return False
```

### 6.2 `src/openpi/cache/config.py` Factory Function

```python
def _build_write_policy(cfg: WritePolicyConfig):
    """Build WritePolicy from config."""
    from openpi.cache.components.write_policy import (
        AlwaysWritePolicy,
        NeverWritePolicy,
        OnAnyMissWritePolicy,
    )
    if cfg.type == "on_any_miss":
        return OnAnyMissWritePolicy()
    elif cfg.type == "always":
        return AlwaysWritePolicy()
    elif cfg.type == "never":
        return NeverWritePolicy()
    else:
        raise ConfigValidationError(f"Unknown write_policy.type '{cfg.type}'")
```

#### `build_cache_components()` Return Dict Addition

```python
def build_cache_components(config: CacheConfig) -> dict[str, Any]:
    # ... existing logic ...
    return {
        # ... existing keys ...
        "write_policy": _build_write_policy(config.write_policy),
    }
```

#### `build_per_connection_components()` Sync

```python
def build_per_connection_components(...) -> dict[str, Any]:
    # ... existing logic ...
    return {
        # ... existing keys ...
        "write_policy": _build_write_policy(config.write_policy),
    }
```

### 6.3 Acceptance Criteria

- `OnAnyMissWritePolicy`: `miss_by_checkpoint={CP1: 1}` -> True, `miss_by_checkpoint={}` -> False
- `AlwaysWritePolicy`: always True
- `NeverWritePolicy`: always False
- YAML with unknown `write_policy.type` -> `ConfigValidationError`
- `build_cache_components()` returned dict contains `write_policy` key

---

## Phase 7: Ingest Script Update

**Goal**: Modify the offline ingest script to populate `prev_ids` / `next_ids` / `trajectory_id` when constructing CacheEntries.

### 7.1 Locate Ingest Script

Search the `exp/` directory for ingest scripts (e.g., `ingest_*.py` or `build_artifact.py`), find where CacheEntries are constructed and `batch_insert()` is called.

### 7.2 Modification Logic

```python
# Pseudocode: build linked list by episode in the ingest script

for episode_id, episode_steps in grouped_by_episode.items():
    entries = []
    for step_idx, step_data in enumerate(sorted(episode_steps, key=lambda s: s.step_idx)):
        entry = CacheEntry(
            id=f"{episode_id}:{step_idx}",
            checkpoint_id=CheckpointID.CP1,
            query_keys=step_data.query_keys,
            payload=CachePayload(
                action_chunk=step_data.action_chunk,
                task_key=step_data.task_key,
            ),
            step_idx=step_idx,
            trajectory_id=episode_id,         # <- New
        )
        entries.append(entry)

    # Link the chain
    for i in range(len(entries)):
        if i > 0:
            entries[i].prev_ids = [entries[i - 1].id]
        if i < len(entries) - 1:
            entries[i].next_ids = [entries[i + 1].id]

    backend.batch_insert(entries)
```

### 7.3 Artifact Loading Compatibility

`InMemoryBackend.load_artifact()` loads entries from HDF5/pickle. Need to confirm:
- New fields `prev_ids` / `next_ids` / `trajectory_id` use default values when absent in old artifacts (`[]` / `[]` / `None`)
- New artifact format needs to include serialization of these three fields

### 7.4 Acceptance Criteria

- After ingest, artifact entries within the same episode have correct mutual `prev_ids`/`next_ids` pointers
- All entries' `trajectory_id` equals the corresponding `episode_id`
- Old artifacts load with new fields at default values, no errors

---

## File Change Summary

| File | Phase | Change Type |
|------|-------|-------------|
| `src/openpi/cache/storage_types.py` | 1 | Modify: CachePayload remove field, CacheEntry add fields, QuerySpec add fields, new StepRecord/EpisodeRecord |
| `src/openpi/cache/config.py` | 1, 6 | Modify: SearchStrategyConfig add fields, new WritePolicyConfig, CacheConfig add fields, registry, validation, factory |
| `src/openpi/cache/backends/in_memory_backend.py` | 2 | Modify: search() add trajectory re-ranking, 3 new private methods |
| `src/openpi/cache/backends/qdrant_backend.py` | 2 | Modify: search() add NotImplementedError guard |
| `src/openpi/cache/backend_base.py` | 2 | Modify: docstring append trajectory search description |
| `src/openpi/cache/components/search_strategy.py` | 3 | Modify: new TrajectoryMixin, three strategy classes inherit mixin and pass parameters |
| `src/openpi/cache/components/gate.py` | 4 | Modify: AlwaysSearchGate add on_episode_start() + record_action() |
| `src/openpi/cache/components/judge.py` | 4 | Modify: AlwaysHitJudge + ThresholdJudge add on_episode_start() + record_action() |
| `src/openpi/cache/components/write_policy.py` | 6 | **New**: WritePolicy Protocol + 3 implementation classes |
| `src/openpi/cache/orchestrator.py` | 5a | Modify: remove old write/write_with_keys, new write_policy parameter, episode buffer, broadcast, on_episode_end, buffer_for_write, `check()` pipeline adjustment (build moved up) |
| `src/openpi/cache/interceptor.py` | 5b | Modify: episode lifecycle forwarding, action broadcast, buffer_for_write replaces old _bg_write |
| `scripts/serve_policy.py` | 5b | Modify: pass write_policy when constructing Orchestrator |
| `exp/` ingest script | 7 | Modify: populate linked list fields when constructing CacheEntry |

**New files**: 1 (`write_policy.py`)
**Modified files**: 12

---

## Risks and Notes

1. **Impact scope of `next_action_chunk` removal**: Phase 1 requires a global search for all references to this field (including tests, documentation, serialization logic); missed references will cause runtime AttributeError
2. **Old artifact compatibility**: `InMemoryBackend.load_artifact()` must use `getattr` to fill in missing `prev_ids=[]` / `next_ids=[]` / `trajectory_id=None` for old entries
3. **Recursion depth**: `_collect_trajectory_entries()` and `_score_trajectory()` recursion depth = `trajectory_depth`. LIBERO episodes typically have hundreds of steps but trajectory_depth is generally no more than 10; will not trigger Python recursion limit
4. **Performance**: Trajectory search uses a two-pass recursion + per-layer batch scoring design. First recursion collects per-layer entry sets (cost ~= linked list traversal), `_batch_step_scores()` batch-scores per layer reusing existing fusion methods, second recursion only looks up and sums. depth is typically 2-5; InMemoryBackend already does full traversal anyway; additional overhead is acceptable
5. **Inference pipeline change**: In `check()`, `build()` is moved up before gate skip (see Phase 5a); gate skip no longer skips `build()`, adding minor CPU overhead but ensuring trajectory completeness
6. **CP3 linked list replacement path not implemented**: Removing `next_action_chunk` does not mean CP3 has been replaced by linked lists. Future CP3 enablement requires a new `fetch_entry_metadata(id)` API. Not implemented in this round
7. **Entry ID semantic change**: Trajectory entries use `trajectory_id:step_idx` instead of `_stable_hash`; idempotent upsert no longer supported. If semantic deduplication is needed, a `semantic_key` field can be added
8. **Judge threshold needs recalibration after trajectory enablement**: `_batch_step_scores()` reuses the configured fusion_method, but the final trajectory score is `sum(weights[i] * step_score_i)`, with different scale from single-step search. After enabling trajectory, ThresholdJudge thresholds must be recalibrated. Note: do not reuse single-step `weighted_score_sum` percentile calibration data -- trajectory score distributions are different

---

## Code Plan Review Questions and Suggestions (2026-04-07)

> Reviewer: Codex
>
> Conclusion: The overall direction of the current plan is sound, but implementing directly from the current version is not recommended. The issues below need to be clarified or incorporated into plan revisions first; otherwise, trajectory chains may be corrupted, runtime may not actually switch to episode-level writes, and trajectory rerank score semantics may be inconsistent with existing strategies.

### P0 Blocking Issues

1. **Entry ID scheme conflicts with trajectory linked list semantics**
   - Current Phase 5 `_build_entry_chain()` still uses `_stable_hash(CheckpointID.CP1, step.query_keys)` as `CacheEntry.id`.
   - Existing `_stable_hash` semantics are "same checkpoint + same query key yields same id," used for idempotent upsert; this suits single-step cache but not trajectory node ids.
   - Risk: When the same observation appears in multiple episodes or multiple times in the same episode, it writes to the same id, causing `prev_ids` / `next_ids` pointers to be overwritten or cross-contaminated; the trajectory is no longer a real episode chain.
   - Need to confirm: Should trajectory entries be "semantically deduplicated" or "each occurrence is an independent node"?
   - Suggestion: Change trajectory node `id` to a unique id, e.g., `trajectory_id:step_idx` / UUID / episode filename + step; if semantic deduplication is still needed, add a `semantic_key` field instead of reusing `id`.

2. **Missing `InferenceInterceptor` and `serve_policy.py` runtime integration**
   - The current file change summary does not include `src/openpi/cache/interceptor.py` and `scripts/serve_policy.py`.
   - But action production, CP1 hit early-return, post-inference writes, and episode lifecycle forwarding all happen in `InferenceInterceptor`.
   - Risk: Even if Orchestrator adds `buffer_for_write()` / `broadcast_action()` / `on_episode_end()`, at runtime these new interfaces won't be called; the old background per-step `write_with_keys()` path will still be used.
   - Suggestion: Phase 5 should include explicit Interceptor changes:
     - `on_episode_start(experiment, task, episode_id)` passes `task` / `episode_id` to Orchestrator.
     - `on_episode_end(success)` calls `orchestrator.on_episode_end()`.
     - CP1 cache hit early-return also calls `broadcast_action()` and `buffer_for_write()` as needed.
     - After Stage3 normal inference, call `broadcast_action(stage3.action_chunk[0])` and `buffer_for_write(cp1_query_keys, action_chunk)`, replacing or bypassing old `_bg_write()`.
     - `serve_policy.py` passes `components["write_policy"]` when constructing `CacheOrchestrator`.

3. **`task_key` has no real data flow**
   - Plan changes `on_task_begin(task_key="")`, but the current WebSocket lifecycle task info comes from `on_episode_start(experiment, task, episode_id)`.
   - Existing `SearchContext` already has `task_key`, but Orchestrator doesn't pass it in during construction, so task filtering beyond `step_filter` doesn't actually work.
   - Risk: `EpisodeRecord.task_key` is empty; payload's `task_key` written at runtime is empty; search doesn't filter by task either, increasing cross-task false matching risk.
   - Suggestion: Use `on_episode_start(..., task, episode_id)` as the episode-level task source; clarify task normalization rules; Orchestrator should pass `task_key=self._current_task_key` when constructing `SearchContext`.

4. **Trajectory rerank candidate pool too small**
   - Phase 2 plans to first use existing `search()` returning `top_k` results, then trajectory-rerank these `top_k`.
   - Risk: When `top_k=1`, trajectory search has no opportunity to pull up candidates that are "slightly worse on current step but very consistent historically," equivalent to only re-scoring the single-step top-1.
   - Requirements doc Q7 says the existing `candidate_multiplier` can be used, but the plan doesn't wire it into InMemoryBackend's initial candidate pool.
   - Suggestion: In trajectory mode, initial candidate count should use `top_k * candidate_multiplier` or add a new `trajectory_candidate_k`; InMemory's `WeightedRrfKnnStrategy` / `WeightedScoreSumKnnStrategy` also need to receive and pass `candidate_multiplier`.

5. **`_compute_step_similarity()` doesn't truly reuse existing per-step fusion semantics**
   - Phase 2 pseudocode computes field similarity weighted average for a single entry.
   - But existing `weighted_score_sum` does cosine to `[0,1]` conversion, L2 `exp(-d/tau)` conversion, optional percentile normalization; existing `weighted_rrf` is rank fusion over a candidate set, not a single-entry weighted average.
   - Risk: Trajectory rerank's step score is semantically inconsistent with initial search score; Judge threshold and ranking behavior become hard to explain.
   - Suggestion: First clarify whether trajectory search only supports `weighted_score_sum`. If `weighted_rrf` support is desired, the candidate set and rank computation rules for each historical layer need to be defined; cannot simply substitute single-entry cosine for RRF.

6. **Trajectory score scale not normalized; threshold is unstable**
   - Current recursion accumulates `sum(weights[i] * step_sim_i)`.
   - Risk: Insufficient history, broken chains, different depths or different weight sums change the score upper bound; `ThresholdJudge`'s threshold cannot be directly reused and is difficult to calibrate.
   - Suggestion: Normalize by actual participating `sum(active_weights)`, or explicitly state in the plan that trajectory score is a non-normalized cumulative score and Judge threshold must be recalibrated.

7. **Gate skip does not record query history**
   - Plan records `_record_query_keys(ctx.query_keys)` at the beginning of `SearchStrategy.search()`.
   - But when Gate returns False, Orchestrator currently doesn't build keys and doesn't call SearchStrategy.
   - Risk: Requirements state "query keys generated at each step should be stored temporarily," but gate-skipped steps are missing, causing query trajectory history to be incomplete.
   - Suggestion: If gate-skipped steps should participate in history, Orchestrator needs to still build and record query keys after gate skip, or add an explicit `SearchStrategy.record_query_keys()` called by Orchestrator; if gate-skip steps are not recorded, this behavior needs to be declared in both the requirements and plan.

### P1 Design Gaps

8. **Incomplete replacement path for `next_action_chunk` deletion**
   - Plan deletes `CachePayload.next_action_chunk` and believes the next-step action can be obtained via `next_ids[0] -> fetch entry -> payload.action_chunk`.
   - But current `CacheStorage.fetch_payload()` only fetches payload by id; search result `SearchResultLite` doesn't carry `next_ids`; Qdrant payload currently doesn't serialize `next_ids`.
   - Risk: CP3 or future predictive skip cannot obtain the next-step entry id from the winner.
   - Suggestion: If CP3 is still suspended/disabled in this phase, explicitly state "deleting `next_action_chunk` does not mean CP3 has been replaced by linked lists"; if the replacement path is to be implemented, a new `fetch_entry()` or at least `fetch_entry_metadata()` API is needed, and linked list fields must be serialized in both InMemory/Qdrant backends.

9. **Insufficient Qdrant / artifact linked list field serialization**
   - Phase 2 only requires Qdrant to raise `NotImplementedError` when trajectory depth > 1, but Phase 5/7 may still write entries with linked list fields through Qdrant.
   - Current Qdrant `_to_point()` payload only writes `checkpoint_id`, `timestamp`, `task_key`, and payload tensors; does not include `step_idx`, `prev_ids`, `next_ids`, `trajectory_id`.
   - InMemory pickle -- old artifact unpickled `CacheEntry` objects may also lack new attributes; dataclass defaults alone may not automatically fill in old objects.
   - Suggestion: Even if Qdrant doesn't support trajectory search yet, clarify whether linked list field persistence is supported; `InMemoryBackend.load_artifact()` should use `getattr` to fill in `prev_ids=[]` / `next_ids=[]` / `trajectory_id=None` for old entries.

10. **`StepRecord.action_chunk: Optional` conflicts with `CachePayload` validation**
    - Phase 1 defines `StepRecord.action_chunk: Optional[torch.Tensor] = None`, but `CachePayload.validate_for_checkpoint()` requires all checkpoints to have `action_chunk`.
    - Risk: `buffer_for_write()` receives None, then `on_episode_end()` constructs CP1 `CachePayload(action_chunk=None)`, and `batch_insert()` validation fails.
    - Suggestion: If CP1 trajectory writing must have action, change `StepRecord.action_chunk` to non-Optional and fail-fast in `buffer_for_write()`; if action absence is allowed, synchronously adjust payload validation and cache hit behavior.

11. **Phase 5's `_episode_had_miss` semantics need checkpoint scoping**
    - Plan sets `_episode_had_miss=True` directly on gate skip / judge miss.
    - Risk: CP3 is currently a skeleton or may have long-term misses, causing `on_any_miss` to almost always write, even when CP1 has full hits.
    - Suggestion: `had_miss` should at least be checkpoint-differentiated, e.g., only count CP1 misses, or change `EpisodeRecord` to `miss_by_checkpoint: dict[CheckpointID, int]`; `OnAnyMissWritePolicy` should explicitly specify which checkpoints to use.

12. **SearchStrategy history buffer and Orchestrator write buffer may duplicate and go out of sync**
    - SearchStrategy maintains `_query_history` / `_action_history`; Orchestrator also maintains `_episode_steps`.
    - Risk: On CP1 hit early-return, gate skip, CP3 check, or old write path retention, these two buffers may differ in length or content; `record_action()` stores action in SearchStrategy, but the current search doesn't use it.
    - Suggestion: Clarify the responsibility boundary of both buffers: SearchStrategy buffer only for query trajectory search; Orchestrator buffer only for writing. Cover CP1 hit early-return, gate skip, and normal miss paths in tests for buffer length consistency.

13. **Preserving old `write()` / `write_with_keys()` will long-term produce entries without linked lists**
    - Plan preserves old path for compatibility, but trajectory search can only treat old-path-written data as chain breaks.
    - Risk: If runtime doesn't fully migrate to `buffer_for_write()`, the database will accumulate large amounts of `prev_ids=[]` single-node entries; trajectory search benefit is unclear and hard to debug.
    - Suggestion: Short-term preserve old API, but mark as legacy single-step write in docstring/log; if trajectory_depth > 1 and old write path is still called, issue a warning.

### P2 Testing and Acceptance Suggestions

14. **Add end-to-end tests, not just Orchestrator tests**
    - Current Phase 5 acceptance focuses on Orchestrator unit tests, but the real data flow crosses `InferenceInterceptor`.
    - Suggest adding tests: simulate one CP1 miss with normal inference then `buffer_for_write()`, episode_end followed by batch_insert forming a chain; simulate CP1 hit early-return still broadcasting action; verify `on_episode_start(task=...)` results in non-empty payload.task_key on write.

15. **Add trajectory rerank rank-reversal test**
    - Need to construct a scenario: single-step top-1 is A, but depth=3 trajectory score makes B win.
    - This test prevents "only reranking already-returned top_k" from making trajectory search ineffective.

16. **Add score normalization test**
    - For the same query, verify score scale meets expectations across three situations: insufficient history, complete history, broken chain.
    - If using active weight normalization, test that the same effective match produces comparable scores across different available history lengths.

17. **Supplement configuration validation**
    - `trajectory_depth >= 1`.
    - Whether negative values or all-zero `trajectory_weights` are allowed needs clarification; suggest requiring non-negative with sum > 0.
    - When backend.type == `qdrant` and any checkpoint has `trajectory_depth > 1`, should it error at config stage or QdrantBackend raises `NotImplementedError` at runtime? Suggest config-stage fail-fast.

### Suggested Plan Revision Order

1. First revise Phase 1: determine trajectory entry id semantics, whether `StepRecord.action_chunk` is Optional, score normalization rules, config validation rules.
2. Then revise Phase 2: clarify `weighted_score_sum` / `weighted_rrf` trajectory per-step scoring semantics, and incorporate initial candidate pool expansion mechanism.
3. Then revise Phase 5: split into `Orchestrator core` and `InferenceInterceptor / serve_policy integration` sub-phases.
4. Finally revise Phase 7: clarify that `exp/build_in_memory_cache_artifact.py` is the current InMemory artifact script; whether Qdrant ingest is in scope for this round needs separate statement.

---

## Review Responses (2026-04-07)

### P0 Blocking Issues Responses

#### #1 Entry ID Scheme Conflicts with Trajectory Linked List Semantics

**Acknowledged, this is a real bug.**

`_stable_hash(CP1, query_keys)` is a semantic dedup key -- when the same observation appears in different episodes the id is the same, and upsert will overwrite the previous entry's `prev_ids`/`next_ids`, causing linked list cross-contamination.

**Modification plan**:
- Change trajectory node id to `f"{trajectory_id}:{step_idx}"`, ensuring global uniqueness for every node in every write
- If semantic dedup is needed in the future ("has this observation been seen before"), add a new `semantic_key: Optional[str]` field (computed using the existing `_stable_hash`), without reusing `id`
- Offline ingest similarly changes to use `trajectory_id:step_idx` as id
- **Affected Phases**: Phase 1 (CacheEntry optionally adds `semantic_key`), Phase 5 (`_build_entry_chain()` changes id generation), Phase 7 (ingest changes id generation)

#### #2 Missing InferenceInterceptor and serve_policy.py

**Acknowledged.**

The plan only modified Orchestrator internals, but the actual call chain is in `interceptor.py` and `serve_policy.py`. Without modifying these two files, the new interfaces will not be invoked and will still take the old background per-step `write_with_keys()` path.

**Modification plan**: Split Phase 5 into two sub-phases:
- **Phase 5a -- Orchestrator core**: Existing Phase 5 content unchanged
- **Phase 5b -- Interceptor + serve_policy integration**:
  - `interceptor.py`:
    - `on_episode_start()` passes task/episode_id to Orchestrator
    - Before CP1 hit early-return, call `broadcast_action()` + `buffer_for_write()`
    - After Stage3 normal inference, call `broadcast_action(action_chunk)` + `buffer_for_write(cp1_query_keys, action_chunk)`
    - `on_episode_end()` calls `orchestrator.on_episode_end()`
    - The above replaces or bypasses the old `_bg_write()` path
  - `serve_policy.py`: Pass `components["write_policy"]` when constructing Orchestrator
- **File change summary additions**: `src/openpi/cache/interceptor.py`, `scripts/serve_policy.py`

#### #3 task_key Has No Real Data Flow

**Acknowledged.**

The existing Orchestrator constructs SearchContext with an empty `task_key`, which is a known gap.

**Modification plan**:
- `on_episode_start(task_key: str = "", episode_id: str = "")` receives task info, stored in `self._current_task_key` and `self._current_episode_id`
- Orchestrator passes `task_key=self._current_task_key` when constructing SearchContext
- Task normalize rule: follow the existing `CachePayload.task_key` convention (canonical task identifier, not raw prompt)
- Completed in Phase 5a

#### #4 Trajectory Rerank Candidate Pool Too Small

**Rejected. The issue stems from the plan's flawed design, not the requirement itself.**

The reviewer's concern is based on the plan's two-stage design (first single-step search truncates to top_k, then trajectory rerank on the truncated results). But this two-stage design itself is the plan's deviation from the requirements document -- the requirements document never mandated two stages.

InMemoryBackend already does full traversal of all filtered entries; there is no ANN index truncation issue. The candidate pool is the full set of filtered entries; no `candidate_multiplier` is needed.

**Modification plan**: Phase 2 has been revised to a two-pass recursion design:
1. First recursion follows `prev_ids` to collect the set of entries needing scoring at each level (deduplicated)
2. Per-level batch scoring: `_batch_step_scores()` reuses the configured fusion_method (RRF / score_sum / single cosine), with ranking scope = that level's entry set
3. Second recursion looks up pre-computed scores and computes weighted sum using `trajectory_weights`

See Phase 2 main text for details.

#### #5 `_compute_step_similarity()` Does Not Truly Reuse Existing Per-Step Fusion Semantics

**Acknowledged, fully resolved in Phase 2.**

The new design no longer implements per-step similarity calculation on its own, but instead **fully reuses** existing per-step fusion methods through `_batch_step_scores()`:
- Constructs a temporary QuerySpec (only replacing `query_keys` with the current level's historical query, not passing trajectory fields so it takes the single-step logic path)
- Dispatches to existing `_search_weighted_rrf()` / `_search_weighted_score_sum()` / `_search_single_field_cosine()` based on the configured `fusion_method`
- RRF ranking scope = the entry set collected at that level (the first recursion of the two-pass approach has already collected per-level entry sets, providing the candidate set ranking basis for RRF)

Therefore there is no semantic inconsistency of "trajectory search uses score sum while single-step uses RRF" -- **every level's per-step score in trajectory search uses exactly the same fusion semantics as single-step search**.

#### #6 Trajectory Score Scale Not Normalized

**Rejected, no normalization will be done.**

- **Insufficient query-side history** (first few steps of episode): All candidates have the same `active_weights_sum` (same `query_history`); normalization does not change ranking
- **Broken chain** (a candidate has `prev_ids=[]`): A broken chain means incomplete trajectory information; scoring lower than a complete chain is **the correct signal**; normalization would erase this useful signal
- **ThresholdJudge threshold**: At deployment time `trajectory_depth` is fixed; the threshold is calibrated to the score scale at that depth; cross-depth comparability is not needed

Trajectory score uses non-normalized weighted accumulated score `sum(weights[i] * step_sim_i)`.

#### #7 Gate Skip Does Not Record Query History

**Acknowledged, inference pipeline modification needed.**

In the existing `check()` flow, gate skip occurs **before** `key_builder.build()` (`orchestrator.py:166-169`); when gate skips, `query_keys` are not built. But writing a complete trajectory requires `query_keys` at every step; otherwise the linked list has gaps.

**Modification plan**: When gate skips, **still call `build()`** to ensure `query_keys` exist at every step.

**Inference pipeline adjustment** (Phase 5a, `orchestrator.py`'s `check()` method):

Existing order:
```
collect() -> gate() -> [skip: return directly] -> build() -> search() -> judge()
```

Changed to:
```
collect() -> gate() -> build() (executed regardless of gate result) -> [skip: buffer + return] -> search() -> judge()
```

- `build()` is moved from **after** the gate skip check to **before** it (moved up a few lines)
- The gate skip `return` is moved down to after `build()`
- On gate skip: `check()` calls `strategy.record_query_keys(query_keys)` (trajectory search history completeness); `buffer_for_write()` is not called in `check()` but by the Interceptor after the action is produced
- `gate()` uses `cached_data` (GPU), which does not depend on `query_keys`; the order change has no side effects
- `build()` is just CPU dimensionality reduction, much cheaper than search; gate skip still saves the search overhead

This way:
- Query history is complete at every step; trajectory search has no gaps
- Write buffer is complete at every step; linked lists written at episode end have no gaps
- Gate's `cached_data` history is accumulated by gate itself in `__call__` (gate is called before `build()`)

### P1 Design Gaps Responses

#### #8 Incomplete Replacement Path After Removing next_action_chunk

**Partially acknowledged.**

The reviewer is correct: the current `fetch_payload()` returns a payload without `next_ids`, and after removing `next_action_chunk`, CP3 indeed has no replacement path. However, the requirements document explicitly states "the specific logic for CP3 forward rolling is not in scope for this design."

**Modification plan**: Add to the plan's risk list:
- Removing `next_action_chunk` eliminates redundancy; it does not claim CP3 has been replaced by linked lists
- CP3 is currently in skeleton/disabled state; removal does not affect runtime
- When CP3 is enabled in the future, a new `fetch_entry_metadata(id) -> CacheEntry` (without payload tensor) or returning `next_ids` in `SearchResultLite` will be needed
- Not implemented in this round

#### #9 Qdrant / Artifact Linked List Field Serialization

**Acknowledged.**

Old pickle artifacts will not automatically acquire new dataclass fields when unpickled.

**Modification plan**:
- Add `getattr` fallback in `InMemoryBackend.load_artifact()`:

```python
for entry in loaded_entries:
    entry.prev_ids = getattr(entry, 'prev_ids', [])
    entry.next_ids = getattr(entry, 'next_ids', [])
    entry.trajectory_id = getattr(entry, 'trajectory_id', None)
```

- Qdrant payload serialization is not in scope for this round (Qdrant does not support trajectory search; linked list fields do not need to be persisted); explicitly stated in the plan
- The new artifact build script (Phase 7) naturally includes these fields

#### #10 StepRecord.action_chunk: Optional Conflicts with CachePayload Validation

**Acknowledged.**

`CachePayload.validate_for_checkpoint()` requires `action_chunk is not None`, but `StepRecord` allowing None would cause `_build_entry_chain()` to construct an illegal payload.

**Modification plan**:
- Change `StepRecord.action_chunk` to **non-Optional**: `action_chunk: torch.Tensor`
- Change `buffer_for_write()` signature to `buffer_for_write(query_keys, action_chunk)` where both are required parameters
- If `action_chunk` has not been produced yet at call time, do not call `buffer_for_write()` (wait until action is produced)

#### #11 `_episode_had_miss` Semantics Need Checkpoint-Specific Scoping

**Acknowledged.**

CP3 is a skeleton that always misses, which would cause `on_any_miss` to always write.

**Modification plan**: Change `EpisodeRecord` to:

```python
@dataclass
class EpisodeRecord:
    steps: list[StepRecord]
    task_key: str
    miss_by_checkpoint: dict[CheckpointID, int]  # e.g. {CP1: 3, CP3: 50}
    total_steps: int
```

`OnAnyMissWritePolicy` defaults to checking only whether CP1's miss count > 0. Configurable in the future to specify which checkpoints to monitor.

#### #12 SearchStrategy Buffer and Orchestrator Buffer Are Redundant

**Rejected, this is by design, not a bug.**

The two buffers have different responsibilities and record at different times:

- **SearchStrategy `_query_history`**: Used for trajectory search. Recorded inside `check()` -- during normal search, `record_query_keys()` is automatically called in `search()`; on gate skip, Orchestrator explicitly calls `record_query_keys()`. **Recorded every step** (because `build()` has been moved up before gate skip)
- **Orchestrator `_episode_steps`**: Used for episode-end writes. **Not recorded in `check()`**; the Interceptor calls `buffer_for_write(query_keys, action_chunk)` after the action is produced. **Recorded every step** (Interceptor produces an action whether from cache hit or model inference)

**Buffers will not be merged.** Test coverage includes:
- CP1 hit early-return: SearchStrategy records query_keys in `search()`; Interceptor calls `buffer_for_write()` after getting cache action
- Gate skip: SearchStrategy has query_keys explicitly recorded by Orchestrator; Interceptor calls `buffer_for_write()` after getting model inference action
- Normal miss: Same as gate skip

#### #13 Preserving Old write() Will Long-Term Produce Entries Without Linked Lists

**Acknowledged. Old API will be directly removed, not preserved.**

Phase 5b already modifies the interceptor to migrate all call sites to the new path (`buffer_for_write()` + `on_episode_end()`); there is no need to preserve an old interface that produces entries without linked lists.

**Modification plan**:
- **Remove** `Orchestrator.write()` and `Orchestrator.write_with_keys()` in Phase 5a
- In Phase 5b, interceptor fully switches to `buffer_for_write()` + `broadcast_action()` + `on_episode_end()` new path
- All test cases using the old `write()` are migrated to the new path
- Clean up `_stable_hash()` if only the old `write()` was using it (need to confirm whether `_build_entry_chain()` still needs it)

### P2 Testing and Acceptance Suggestions Responses

#### #14 Add End-to-End Tests

**Acknowledged.**

Phase 5b acceptance criteria adds end-to-end integration tests:
- Simulate CP1 miss -> normal inference -> `buffer_for_write()` -> `on_episode_end()` -> batch_insert forms chain
- Simulate CP1 hit early-return -> still `broadcast_action()`
- Verify `on_episode_start(task_key=...)` results in non-empty payload.task_key on write

#### #15 Add Trajectory Rerank Rank-Reversal Test

**Acknowledged, excellent test suggestion.**

Phase 2 acceptance criteria adds: Construct a scenario where A is single-step top-1 but has poor trajectory, and B is single-step top-2 but has a good trajectory; verify that after rerank, B ranks first.

#### #16 Add Score Normalization Test

**Acknowledged.**

Phase 2 acceptance criteria adds: Verify that under non-normalized scores, complete chain > broken chain; when history is insufficient, only compare ranking correctness without requiring same scale across different history lengths.

#### #17 Supplement Configuration Validation

**Acknowledged.**

Phase 1's `validate_cache_config()` adds:
- `trajectory_depth >= 1`
- All values in `trajectory_weights` are non-negative with sum > 0
- When `backend.type == "qdrant"` and any checkpoint has `trajectory_depth > 1`, **error at config stage** (fail-fast), do not wait until runtime

### Suggested Plan Revision Order

**Acknowledged that the reviewer's suggested revision order is reasonable.**
