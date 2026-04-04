# Custom Cache Components Guide

How to write your own Gate, Judge, and SearchStrategy for the openpi cache system.

> Prerequisites: read [cache_system_architecture.md](cache_system_architecture.md) for the CP1/CP2/CP3 design. This guide covers the **component extension** layer, not the cache system's overall architecture.

---

## 1. System Overview

The cache pipeline runs inside `CacheOrchestrator.check()` at each checkpoint:

```
Stage outputs
  -> KeyBuilder.collect()        # extract raw tensors (GPU)
  -> Gate(checkpoint_id, cached_data) -> bool
  -> KeyBuilder.build()          # CPU float32 L2-normalised query keys
  -> SearchStrategy.search(ctx)  # construct QuerySpec, call CacheStorage.search()
  -> Judge(results, checkpoint_id, cached_data) -> (HitType, winner_id)
  -> fetch_payload(winner_id)    # only on FULL_HIT
  -> CheckResult
```

Each component is a **Protocol** (structural typing). You don't need to inherit from anything -- just implement the right method signatures and pass the `isinstance` check.

### Per-checkpoint dispatch

Gates, Judges, and SearchStrategies are stored as `dict[CheckpointID, Component]` in the Orchestrator. CP1 and CP3 can use **different** instances. KeyBuilder is shared across checkpoints.

Each checkpoint can be **individually disabled** via `enabled: false` in the YAML config:

```yaml
checkpoints:
  cp1:
    enabled: true    # CP1 cache check active
    ...
  cp3:
    enabled: false   # CP3 cache check entirely skipped (always MISS)
```

When a checkpoint is disabled, `build_cache_components()` does not create its Gate/Judge/SearchStrategy, and `Orchestrator.check()` returns MISS immediately without touching any component.

---

## 2. GateFunction

### What it does

Decides whether to perform a cache lookup at all. If it returns `False`, the Orchestrator skips search/judge entirely and returns `MISS`. This is the cheapest exit point in the pipeline.

### Protocol

```python
from openpi.cache.types import CheckpointID

class GateFunction(Protocol):
    def __call__(
        self,
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
    ) -> bool: ...
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `checkpoint_id` | `CheckpointID` | `CP1` or `CP3`. Tells you which checkpoint is being evaluated. |
| `cached_data` | `dict[str, Tensor]` | Raw tensors from `KeyBuilder.cached_data`. These are on the **original device** (GPU). Keys are internal names like `"state"`, `"action_chunk"` -- not the query field names. |

### Return value

`True` = proceed with cache search. `False` = skip, return MISS immediately.

### Example: State-Change Gate

Only search the cache when the robot state has changed significantly since the last check:

```python
import torch

class StateChangeGate:
    """Skip cache search if state hasn't changed much since last check."""

    def __init__(self, threshold: float = 0.01) -> None:
        self._threshold = threshold
        self._last_state: torch.Tensor | None = None

    def __call__(
        self,
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
    ) -> bool:
        state = cached_data.get("state")
        if state is None:
            return True  # no state available, search anyway

        if self._last_state is not None:
            diff = (state - self._last_state).norm().item()
            if diff < self._threshold:
                return False  # state barely changed, skip search

        self._last_state = state.clone()
        return True
```

### Constraints and pitfalls

1. **No side effects on the storage layer.** Gate must not call `CacheStorage.search()` or any backend method. It is a pure predicate.
2. **`cached_data` tensors are GPU references.** Do not call `.cpu()` inside the gate -- that would insert an unwanted D2H transfer in the hot path. Use in-place GPU operations if you need to compute on them.
3. **Called before `build()`.** At gate time, `query_keys` (CPU normalised vectors) don't exist yet. You only have `cached_data` (raw GPU tensors). This is intentional -- if the gate says no, we skip the `build()` D2H transfer entirely.
4. **Stateful gates must handle `on_task_begin()`**. The Orchestrator resets its `_step_counter` on task begin, but your gate's internal state (like `_last_state` above) won't be reset automatically. If you need reset semantics, consider clearing state when `checkpoint_id == CP1` and `_last_state` is stale, or coordinate with the Interceptor's task lifecycle.

### Registration

To make your gate available via YAML config:

1. In `config.py`, add your type string to the `gate.type` validation list.
2. Add a branch in `_build_gate()`:
   ```python
   elif cfg.type == "state_change":
       return StateChangeGate(threshold=cfg.threshold)
   ```
3. Add any new config fields to `GateConfig` dataclass.

---

## 3. SimilarityJudge

### What it does

Receives the search results (sorted by descending score) and decides whether the top result constitutes a cache hit. The Judge does **not** fetch payloads or interact with storage -- it's pure judgment logic.

### Protocol

```python
from openpi.cache.components.judge import HitType
from openpi.cache.storage_types import SearchResultLite

class SimilarityJudge(Protocol):
    def __call__(
        self,
        results: list[SearchResultLite],
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
    ) -> tuple[HitType, Optional[str]]: ...
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `results` | `list[SearchResultLite]` | Search results sorted by descending score. May be empty (no entries in DB). Each has `.id`, `.score`, `.checkpoint_id`. |
| `checkpoint_id` | `CheckpointID` | Which checkpoint. Useful for per-CP thresholds. |
| `cached_data` | `dict[str, Tensor]` | Same as Gate's `cached_data`. Available for re-scoring or secondary checks. |

### Return value

`(HitType, winner_id)`:
- `(HitType.FULL_HIT, "entry_id_string")` -- Orchestrator will call `fetch_payload(winner_id)`.
- `(HitType.MISS, None)` -- no cache hit, proceed with normal inference.

### Score semantics

**Crucial:** the score range depends on the backend and search mode.

| Mode | Score range | Example |
|------|-------------|---------|
| Single-field cosine (InMemoryBackend, Qdrant cosine) | `[-1, 1]` | `0.98` = very similar |
| Multi-field RRF fusion (Qdrant) | Small positive numbers | `0.016` = typical top-1 |

Your threshold must match the backend/mode in use. If you switch from single-field cosine to RRF fusion, **all thresholds must be recalibrated**.

### Example: Top-K Margin Judge

Hit only when the top result's score is significantly better than the second:

```python
class MarginJudge:
    """Hit only if top-1 score exceeds a threshold AND has sufficient margin over top-2."""

    def __init__(self, threshold: float = 0.95, margin: float = 0.05) -> None:
        self._threshold = threshold
        self._margin = margin

    def __call__(
        self,
        results: list[SearchResultLite],
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
    ) -> tuple[HitType, Optional[str]]:
        if not results:
            return HitType.MISS, None

        top = results[0]
        if top.score < self._threshold:
            return HitType.MISS, None

        # Check margin over second result (if exists).
        if len(results) >= 2:
            second = results[1]
            if top.score - second.score < self._margin:
                return HitType.MISS, None  # ambiguous, don't trust the match

        return HitType.FULL_HIT, top.id
```

### Constraints and pitfalls

1. **No storage access.** The Judge must not call `fetch_payload()` or any CacheStorage method. Fetching the payload is the Orchestrator's responsibility after the Judge returns `FULL_HIT`. This is intentional: if the Judge says MISS, no expensive payload deserialization happens.
2. **`results` can be empty.** Always handle the `len(results) == 0` case first.
3. **`results` may have fewer than `top_k` entries.** Client-side filtering or a small DB can yield fewer results.
4. **Return exactly one `winner_id`.** Even if multiple results exceed the threshold, return only the top one. Multi-hit semantics are not supported.
5. **Threshold recalibration.** When the KeyBuilder, backend, or fusion weights change, the score distribution shifts. Always re-examine your thresholds after such changes.

### Registration

Same pattern as Gate:

1. Add type string to the `judge.type` validation list in `validate_cache_config()`.
2. Add a branch in `_build_judge()`.
3. Add any new fields to `JudgeConfig`.

---

## 4. SearchStrategy

### What it does

The **single exit point** for all database searches. SearchStrategy is the only component that constructs `QuerySpec` and calls `CacheStorage.search()`. This decouples search parameters from the Orchestrator.

### Protocol

```python
from openpi.cache.components.search_strategy import SearchContext
from openpi.cache.storage_types import SearchResultLite

class SearchStrategy(Protocol):
    def search(self, ctx: SearchContext) -> list[SearchResultLite]: ...
```

### SearchContext (input)

Constructed by the Orchestrator, read-only for the strategy.

```python
@dataclass
class SearchContext:
    query_keys: dict[str, torch.Tensor]   # CPU float32 L2-normalised, from KeyBuilder.build()
    checkpoint_id: CheckpointID
    current_step: int = 0                  # inference cycles since task begin
    task_key: Optional[str] = None         # normalised task identifier
```

| Field | Source | Usage |
|-------|--------|-------|
| `query_keys` | `KeyBuilder.build()` | Named vectors for search. Keys must be in `CACHE_QUERY_FIELDS`. |
| `checkpoint_id` | Orchestrator | Which checkpoint. Forward to `QuerySpec.checkpoint_id`. |
| `current_step` | Orchestrator's `_step_counter` | For step-based filtering. Incremented after each CP1 check. Reset by `on_task_begin()`. |
| `task_key` | Currently always `None` | For future task-based filtering. |

### QuerySpec (output to CacheStorage)

The strategy constructs this and passes it to `CacheStorage.search()`:

```python
@dataclass
class QuerySpec:
    query_keys: dict[str, torch.Tensor]          # required
    top_k: int = 10
    checkpoint_id: Optional[CheckpointID] = None
    filters: Optional[QueryFilter] = None         # task_key, step_range
    fusion_weights: Optional[dict[str, float]] = None     # per-field weights
    backend_hints: Optional[dict[str, Any]] = None        # backend-specific params
```

### QueryFilter

```python
@dataclass
class QueryFilter:
    task_key: Optional[str] = None
    step_range: Optional[tuple[int, int]] = None  # [min, max] inclusive
```

**Backend support for filters is not universal.** Each backend declares what it supports via `supported_filters()`. CacheStorage raises `UnsupportedFilterError` if you use a filter the backend doesn't support. Check before using:

| Filter | InMemoryBackend | Qdrant |
|--------|----------------|--------|
| `checkpoint_id` | Yes | Yes |
| `task_key` | No | Yes |
| `step_range` | No | Yes (but write path doesn't persist `step_idx` yet) |

**Current limitation:** `step_range` filter on Qdrant is blocked by config validation (`qdrant` + `step_filter != "all"` raises `ConfigValidationError`), because the write path does not yet persist `step_idx` into point payloads. This will be lifted when the write path is extended.

### backend_hints

A `dict[str, Any]` pass-through channel for backend-specific parameters. The backend reads what it recognises, ignores the rest. Currently used hints:

| Key | Used by | Default | Description |
|-----|---------|---------|-------------|
| `rrf_k` | Qdrant | 60 | RRF fusion parameter k. Higher k = more weight to lower-ranked results. |
| `candidate_multiplier` | Qdrant | 5 | Prefetch limit = `top_k * candidate_multiplier`. |

InMemoryBackend ignores all hints. If you add a new backend, define your own hint keys and document them.

### Example: Weighted Multi-Strategy

Use different top_k for CP1 vs CP3 based on the checkpoint:

```python
class AdaptiveKnnStrategy:
    """Use larger top_k for CP3 to increase recall for action scheduling."""

    def __init__(
        self,
        storage: CacheStorage,
        cp1_top_k: int = 1,
        cp3_top_k: int = 5,
        fusion_weights: Optional[dict[str, float]] = None,
    ) -> None:
        self._storage = storage
        self._cp1_top_k = cp1_top_k
        self._cp3_top_k = cp3_top_k
        self._fusion_weights = fusion_weights

    def search(self, ctx: SearchContext) -> list[SearchResultLite]:
        top_k = self._cp3_top_k if ctx.checkpoint_id == CheckpointID.CP3 else self._cp1_top_k
        spec = QuerySpec(
            query_keys=ctx.query_keys,
            top_k=top_k,
            checkpoint_id=ctx.checkpoint_id,
            fusion_weights=self._fusion_weights,
        )
        return self._storage.search(spec)
```

### Constraints and pitfalls

1. **Must call `self._storage.search(spec)`, not the backend directly.** CacheStorage provides thread safety (RLock), dimension validation, and filter support checking. Calling the backend directly bypasses all of these.
2. **`query_keys` are already CPU float32 L2-normalised.** Don't re-normalise or transfer to GPU.
3. **Return results as-is from `CacheStorage.search()`.** The Judge expects `list[SearchResultLite]` sorted by descending score. Don't re-sort or filter -- that's the Judge's job.
4. **CacheStorage instance is shared with Orchestrator.** The Orchestrator uses `storage.insert()` and `storage.fetch_payload()`. Your strategy uses `storage.search()`. They share the same RLock, so concurrent access is safe, but don't hold long-lived references to results across calls.
5. **Don't make hit/miss decisions.** That's the Judge's responsibility. The strategy returns candidates; the judge evaluates them.

### Registration

1. Add type string to the `search_strategy.type` validation list in `validate_cache_config()`.
2. Add a branch in `_build_search_strategy()`. Note the function receives `storage` and `fusion_weights` as arguments.
3. Add any new fields to `SearchStrategyConfig`.

---

## 5. Wiring It All Together

### Option A: YAML config (recommended)

Add your component type to `config.py`, then configure in `cache.yaml`:

```yaml
checkpoints:
  cp1:
    gate:
      type: state_change
      threshold: 0.01
    judge:
      type: margin
      threshold: 0.95
      margin: 0.05
    search_strategy:
      type: adaptive_knn
      cp1_top_k: 1
      cp3_top_k: 5
```

For each new type, you need to:
1. Add a config dataclass (e.g. extend `GateConfig` fields or create a parallel one).
2. Add validation in `validate_cache_config()`.
3. Add a factory branch in `_build_gate()` / `_build_judge()` / `_build_search_strategy()`.

### Option B: Direct construction in tests

Tests can bypass config entirely and construct components directly:

```python
from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.types import CheckpointID

backend = InMemoryBackend({"robot_state": 32})
storage = CacheStorage(backend)

gate = StateChangeGate(threshold=0.01)
judge = MarginJudge(threshold=0.95, margin=0.05)
strategy = AdaptiveKnnStrategy(storage, cp1_top_k=1, cp3_top_k=5)

orch = CacheOrchestrator(
    storage=storage,
    key_builder=PlaceholderKeyBuilder(),
    gates={CheckpointID.CP1: gate, CheckpointID.CP3: gate},
    judges={CheckpointID.CP1: judge, CheckpointID.CP3: judge},
    search_strategies={CheckpointID.CP1: strategy, CheckpointID.CP3: strategy},
)
```

### Component isolation rules

| Rule | Reason |
|------|--------|
| Components must NOT import `config.py` | Config is a factory layer, not a dependency. Components receive values via constructors. |
| Gate/Judge must NOT call CacheStorage | Gate decides "should we search?", Judge decides "is this a hit?". Neither does IO. |
| Only SearchStrategy calls `storage.search()` | Single point of query construction. |
| Orchestrator calls `storage.insert()` and `storage.fetch_payload()` | Write and payload retrieval are not delegated to any pluggable component. |

---

## 6. Testing your component

### Gate

```python
def test_my_gate_returns_false_when_state_unchanged():
    gate = StateChangeGate(threshold=0.01)
    state = torch.randn(1, 32)
    cached_data = {"state": state}

    # First call: no previous state, should return True.
    assert gate(CheckpointID.CP1, cached_data) is True

    # Second call with same state: should return False.
    assert gate(CheckpointID.CP1, cached_data) is False
```

### Judge

```python
def test_my_judge_misses_on_empty_results():
    judge = MarginJudge(threshold=0.95, margin=0.05)
    hit_type, winner = judge([], CheckpointID.CP1, {})
    assert hit_type == HitType.MISS

def test_my_judge_hits_with_sufficient_margin():
    results = [
        SearchResultLite(id="a", score=0.99, checkpoint_id=CheckpointID.CP1),
        SearchResultLite(id="b", score=0.80, checkpoint_id=CheckpointID.CP1),
    ]
    judge = MarginJudge(threshold=0.95, margin=0.05)
    hit_type, winner = judge(results, CheckpointID.CP1, {})
    assert hit_type == HitType.FULL_HIT
    assert winner == "a"
```

### SearchStrategy

For strategies that use filter modes unsupported by InMemoryBackend, mock `storage.search()`:

```python
from unittest.mock import MagicMock

def test_my_strategy_builds_correct_spec():
    storage = MagicMock(spec=CacheStorage)
    storage.search.return_value = []

    strategy = AdaptiveKnnStrategy(storage, cp1_top_k=3, cp3_top_k=10)
    ctx = SearchContext(
        query_keys={"robot_state": torch.randn(32)},
        checkpoint_id=CheckpointID.CP3,
    )
    strategy.search(ctx)

    spec = storage.search.call_args[0][0]
    assert spec.top_k == 10  # CP3 path
```
