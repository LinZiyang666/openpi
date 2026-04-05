# OpenPI Inference Cache System — Complete Tutorial

> **Replaces**: `adding_vector_store_backend.md`, `custom_cache_components.md`
> For deep design rationale and checkpoint theory, see [cache_system_architecture.md](cache_system_architecture.md).

This is a self-contained guide for developers who want to understand, configure, or extend the multi-level inference cache. No prior reading required, but the architecture doc provides deeper context.

---

## Table of Contents

1. [Glossary](#1-glossary)
2. [Architecture Overview](#2-architecture-overview)
3. [Data Flow](#3-data-flow)
4. [Component: KeyBuilder](#4-component-keybuilder)
5. [Component: Gate](#5-component-gate)
6. [Component: Judge](#6-component-judge)
7. [Component: SearchStrategy](#7-component-searchstrategy)
8. [Storage Layer](#8-storage-layer)
9. [Adding a Custom Vector DB Backend](#9-adding-a-custom-vector-db-backend)
10. [YAML Config System](#10-yaml-config-system)
11. [Orchestrator](#11-orchestrator)
12. [Interceptor](#12-interceptor)
13. [Component Isolation Rules](#13-component-isolation-rules)
14. [Testing Patterns](#14-testing-patterns)
15. [Current Validation Status](#15-current-validation-status)

---

## 1. Glossary

| Term | Definition |
|------|-----------|
| **CheckpointID** | Enum (`CP1`, `CP2`, `CP3`). Three positions in the inference pipeline where cache checks occur. Defined in `types.py`. |
| **CP1** | After Stage 1 (vision + tokenisation). A hit skips Stage 2 + Stage 3. |
| **CP2** | After Stage 2 (LLM backbone). **Suspended** — not implemented. |
| **CP3** | After Stage 3 (flow matching). Schedules a cached action for the *next* inference cycle. |
| **CACHE_QUERY_FIELDS** | The five canonical field names: `vision_0`, `vision_1`, `vision_2`, `prompt_emb`, `robot_state`. Defined as a `frozenset` in `types.py`. |
| **CacheEntry** | A single unit written to the vector store. Contains `id`, `checkpoint_id`, `query_keys` dict, `CachePayload`, `timestamp`. |
| **CachePayload** | The stored action data: `action_chunk` [50, 32] (always required), `next_action_chunk` (CP3 only), `intermediates` (CP2 only), `task_key`. |
| **QuerySpec** | Everything a backend needs to execute one search: `query_keys`, `top_k`, `filters`, `fusion_weights`, `backend_hints`. Constructed **only** by SearchStrategy. |
| **QueryFilter** | Dynamic per-query constraints: `task_key` (exact match), `step_range` (inclusive range on step index). |
| **SearchResultLite** | Lightweight search result (no payload tensors): `id`, `score`, `checkpoint_id`. Phase 1 of two-phase search. |
| **CheckResult** | Output of `Orchestrator.check()`: `hit_type`, optional `payload`, `score`, `entry_id`. |
| **HitType** | Enum: `MISS`, `FULL_HIT`. |
| **KeyBuilder** | Extracts stage outputs (GPU) into query key vectors (CPU float32). The only D2H transfer point. |
| **Gate** | Boolean predicate: should we search at all? Called before D2H transfer to avoid unnecessary work. |
| **Judge** | Evaluates search results to decide hit/miss based on score thresholds. |
| **SearchStrategy** | The single exit point for database queries. Constructs `QuerySpec` and calls `CacheStorage.search()`. |
| **CacheStorage** | Thread-safe facade over `VectorStoreBackend`. Validates dimensions and filters, serialises calls with `RLock`. |
| **VectorStoreBackend** | ABC that all vector database implementations extend. |
| **Orchestrator** | Assembles components, runs the check/write pipeline: gate → build → search → judge → fetch. |
| **Interceptor** | `InferenceInterceptor` — a `BasePolicy` drop-in that wraps the real policy and integrates cache checks into inference. |
| **Two-phase search** | Phase 1: `search()` returns `SearchResultLite` (no payload). Phase 2: `fetch_payload()` called only for the winner. Avoids bulk tensor transfer. |
| **RRF** | Reciprocal Rank Fusion — multi-field search combining results from multiple named vectors with weighted ranks. Used by Qdrant backend. |
| **backend_hints** | Pass-through dict in `QuerySpec` for backend-specific parameters (e.g., `rrf_k`, `candidate_multiplier`). Ignored by backends that don't understand them. |
| **Stable hash** | `SHA256(checkpoint_id.name + sorted_query_key_bytes)[:32]`. Same observation → same ID. Enables idempotent upserts. |
| **Idempotent insert** | Reinserting the same ID overwrites the previous entry. No duplicates, no errors. |

---

## 2. Architecture Overview

```
InferenceInterceptor          ← BasePolicy drop-in, wraps real policy
  └─ CacheOrchestrator        ← Assembles components, runs check/write
       ├─ QueryKeyBuilder      ← GPU → CPU query vectors
       ├─ GateFunction         ← Should we search?
       ├─ SearchStrategy       ← Builds QuerySpec, calls storage
       ├─ SimilarityJudge      ← Hit or miss?
       └─ CacheStorage         ← Thread-safe facade
            └─ VectorStoreBackend ABC
                 ├─ InMemoryBackend      (dev/test)
                 └─ QdrantVectorStore    (production)
```

**Layer responsibilities:**

| Layer | File | Role |
|-------|------|------|
| Interceptor | `cache/interceptor.py` | Wraps policy, inserts CP1/CP3 checks into inference loop |
| Orchestrator | `cache/orchestrator.py` | Coordinates gate → build → search → judge → fetch pipeline |
| Components | `cache/components/*.py` | Pluggable protocols: KeyBuilder, Gate, Judge, SearchStrategy |
| Storage | `cache/cache_storage.py` | Thread safety, dimension validation, filter checking |
| Backend | `cache/backend_base.py` + `cache/backends/` | Vector DB implementations |
| Config | `cache/config.py` | YAML → dataclass → factory → component instances |

For checkpoint design rationale (why CP1/CP2/CP3, compute savings per stage), see [cache_system_architecture.md](cache_system_architecture.md) Sections 2–3.

---

## 3. Data Flow

### 3.1 Check Pipeline

When `orchestrator.check(checkpoint_id, **stage_outputs)` is called:

```
Step 1: key_builder.collect(checkpoint_id, stage1=...)
        └─ Hold GPU tensor references (no copy)

Step 2: gate(checkpoint_id, key_builder.cached_data)
        └─ If False → return CheckResult(MISS) immediately

Step 3: key_builder.build(checkpoint_id)
        └─ GPU → CPU float32, L2-normalise (impl-dependent)
        └─ THIS IS THE ONLY D2H TRANSFER POINT

Step 4: Construct SearchContext(query_keys, checkpoint_id, current_step, task_key)

Step 5: search_strategy.search(ctx)
        └─ Build QueryFilter (step filtering)
        └─ Build QuerySpec (fusion_weights, backend_hints)
        └─ CacheStorage.search(spec) → Backend.search(spec)
        └─ Returns list[SearchResultLite] sorted descending by score

Step 6: judge(results, checkpoint_id, cached_data)
        └─ Returns (HitType, winner_id)
        └─ If MISS → return CheckResult(MISS)

Step 7: storage.fetch_payload(winner_id)
        └─ Returns CachePayload (CPU tensors)

Step 8: Increment step counter (CP1 only)

Step 9: Return CheckResult(FULL_HIT, payload, score, entry_id)
```

### 3.2 Write Pipeline

When `orchestrator.write(checkpoint_id, payload, **stage_outputs)` is called:

```
Step 1: key_builder.collect(checkpoint_id, stage1=...)
Step 2: query_keys = key_builder.build(checkpoint_id)
Step 3: entry_id = _stable_hash(checkpoint_id, query_keys)
Step 4: entry = CacheEntry(id, checkpoint_id, query_keys, payload)
Step 5: storage.insert(entry)  →  backend.insert(entry)
```

In the Interceptor, CP1 writes happen in a **background thread** (`ThreadPoolExecutor(max_workers=1)`). The main thread materialises tensors to CPU via `build()`, then submits `write_with_keys()` to avoid blocking inference.

---

## 4. Component: KeyBuilder

**Source**: `src/openpi/cache/components/key_builder.py`
**Pipeline position**: Steps 1 and 3 of check; Steps 1–2 of write

### Protocol

```python
class QueryKeyBuilder(Protocol):
    def collect(self, checkpoint_id: CheckpointID, **stage_outputs) -> None:
        """Cache GPU tensor references from stage outputs. No copy."""

    def build(self, checkpoint_id: CheckpointID) -> dict[str, torch.Tensor]:
        """Build query vectors. Returns {field_name: [dim] CPU float32}.
        This is the ONLY D2H transfer point."""

    @property
    def cached_data(self) -> dict[str, torch.Tensor]:
        """Raw GPU tensors for Gate/Judge to read (before build)."""

    def clear(self) -> None:
        """Release cached references. Called at end of each cycle."""
```

### Input / Output

| Method | Input | Output | Device |
|--------|-------|--------|--------|
| `collect()` | `stage1: Stage1Output` (`.state [B, 32]`, `.prefix_embs [B, prefix_len, emb_dim]`), optionally `stage3: Stage3Output` (`.action_chunk [B, 50, 32]`) | None (stores refs) | GPU |
| `build()` | (reads internal cache) | `dict[str, Tensor]` — field → `[dim]` | CPU float32 |
| `cached_data` | — | `dict[str, Tensor]` — raw GPU refs | GPU |

### Existing Implementations

**PlaceholderKeyBuilder** — Simplest implementation. Uses only `stage1.state`:
- `collect()`: stores `state [B, 32]` GPU reference
- `build()`: `F.normalize(state[0], dim=0)` → `{robot_state: [32] L2-normalized CPU float32}`
- Only supports `robot_state` field
- CP1 and CP3 produce the same key (action concat deferred to Step 6)

**FullOriginalKeyBuilder** — Multi-modal key builder. Splits `prefix_embs` back into original segments:
- Token layout: `prefix_embs = [vision_0 (256 tokens) | vision_1 (256) | vision_2 (256) | prompt (variable)]`
- Each segment is **flattened raw** (`reshape(-1)`) — no mean-pooling, no L2-normalisation
- Prompt segment is zero-padded or truncated to match `vector_dims[PROMPT_EMB]`
- Robot state output raw (no normalisation)
- Constructor: `FullOriginalKeyBuilder(enabled_fields=["vision_0", "robot_state", ...], vector_dims={...})`
- Output dims example: `{vision_0: 524288, vision_1: 524288, vision_2: 524288, prompt_emb: 409600, robot_state: 32}`

### Registering a New KeyBuilder

1. Create a class implementing the 4-method protocol
2. Add your type string to `config.py` validation (line ~341): `if config.key_builder.type not in ("placeholder", "full_original", "your_type")`
3. Add a factory branch in `_build_key_builder()` (line ~498)
4. Add cross-validation rules if your builder only supports certain fields

---

## 5. Component: Gate

**Source**: `src/openpi/cache/components/gate.py`
**Pipeline position**: Step 2 of check — called **before** D2H transfer

### Protocol

```python
class GateFunction(Protocol):
    def __call__(
        self,
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],  # GPU tensors from KeyBuilder
    ) -> bool:
        """Return True to proceed with search, False to skip (return MISS)."""
```

### Input / Output

| Parameter | Type | Description |
|-----------|------|-------------|
| `checkpoint_id` | `CheckpointID` | CP1 or CP3 |
| `cached_data` | `dict[str, Tensor]` | Raw GPU tensors from `KeyBuilder.cached_data` (read-only) |
| **Returns** | `bool` | `True` = search, `False` = skip |

### Existing Implementation

**AlwaysSearchGate** — Always returns `True`. No dependencies, no state. Suitable for initial development.

### Example: StateChangeGate (sketch)

```python
class StateChangeGate:
    def __init__(self, threshold: float = 0.01):
        self._prev_state: Optional[torch.Tensor] = None
        self._threshold = threshold

    def __call__(self, checkpoint_id, cached_data):
        state = cached_data.get("state")
        if state is None or self._prev_state is None:
            self._prev_state = state
            return True
        diff = (state - self._prev_state).norm().item()
        self._prev_state = state
        return diff > self._threshold  # skip search if state barely changed
```

### Constraints

- **No storage access** — Gate is a predicate, not a query
- **Do not call `.cpu()`** on `cached_data` tensors — they are GPU, avoid unnecessary transfers
- **Stateful gates** must manage their own reset (Orchestrator does not reset gate state)

### Registration

1. Add type string to gate validation in `config.py` (line ~353)
2. Add factory branch in `_build_gate()` (line ~512)
3. Add config fields to `GateConfig` dataclass if your gate has parameters

---

## 6. Component: Judge

**Source**: `src/openpi/cache/components/judge.py`
**Pipeline position**: Step 6 of check — after search results returned

### Protocol

```python
class SimilarityJudge(Protocol):
    def __call__(
        self,
        results: list[SearchResultLite],       # sorted descending by score
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],  # GPU tensors from KeyBuilder
    ) -> tuple[HitType, Optional[str]]:
        """Returns (hit_type, winner_id). winner_id is None for MISS."""
```

### Input / Output

| Parameter | Type | Description |
|-----------|------|-------------|
| `results` | `list[SearchResultLite]` | Sorted descending by score. May be empty. |
| `checkpoint_id` | `CheckpointID` | CP1 or CP3 |
| `cached_data` | `dict[str, Tensor]` | Raw GPU tensors (for future re-scoring judges) |
| **Returns** | `(HitType, Optional[str])` | `(FULL_HIT, entry_id)` or `(MISS, None)` |

### Score Semantics

**Thresholds must be calibrated to match the backend + key_builder combination:**

| Mode | Score Range | Example |
|------|-------------|---------|
| Single-field cosine (InMemoryBackend) | [-1, 1] | 0.98 = very similar |
| Multi-field RRF (Qdrant) | Small positive numbers | Scale depends on `rrf_k` and prefetch count |

### Existing Implementations

**ThresholdJudge** — Per-checkpoint threshold comparison:
```python
ThresholdJudge(cp1_threshold=0.98, cp3_threshold=0.95)
```
Logic: if `results[0].score >= threshold` → `FULL_HIT`, else `MISS`.

**AlwaysHitJudge** — Returns `FULL_HIT` for top-1 result (if any). Useful for testing and calibration.

### Constraints

- Handle empty results (return MISS)
- No storage access
- Return exactly one `winner_id` (the entry whose payload will be fetched)
- Recalibrate thresholds when switching backends, key builders, or fusion strategies

### Registration

1. Add type string to judge validation in `config.py`
2. Add factory branch in `_build_judge()` (line ~522)
3. Add config fields to `JudgeConfig` dataclass

---

## 7. Component: SearchStrategy

**Source**: `src/openpi/cache/components/search_strategy.py`
**Pipeline position**: Step 5 of check — the **ONLY** place that constructs `QuerySpec` and calls `CacheStorage.search()`

### Protocol

```python
@dataclass
class SearchContext:
    query_keys: dict[str, torch.Tensor]   # from KeyBuilder.build(), CPU float32
    checkpoint_id: CheckpointID
    current_step: int = 0                 # inference cycle count within task
    task_key: Optional[str] = None        # normalised task ID

class SearchStrategy(Protocol):
    def search(self, ctx: SearchContext) -> list[SearchResultLite]:
        """Execute search. Returns results sorted descending by score."""
```

### Input / Output

| Parameter | Type | Description |
|-----------|------|-------------|
| `ctx.query_keys` | `dict[str, Tensor]` | CPU float32 vectors from KeyBuilder |
| `ctx.checkpoint_id` | `CheckpointID` | CP1 or CP3 |
| `ctx.current_step` | `int` | Step counter for time-based filtering |
| `ctx.task_key` | `Optional[str]` | Task filter (None = no filter) |
| **Returns** | `list[SearchResultLite]` | Sorted descending by score |

### QuerySpec Fields

The SearchStrategy builds a `QuerySpec` to pass to `CacheStorage.search()`:

```python
QuerySpec(
    query_keys=ctx.query_keys,          # vectors to search
    top_k=1,                            # max results
    checkpoint_id=ctx.checkpoint_id,    # for backend filtering
    filters=QueryFilter(                # dynamic constraints
        task_key="pick_up_cup",
        step_range=(0, 10),
    ),
    fusion_weights={"vision_0": 1.0, "robot_state": 2.0},  # per-field RRF weights
    backend_hints={"rrf_k": 60, "candidate_multiplier": 5},  # backend-specific
)
```

### Step Filter Modes

| Mode | QueryFilter | Description |
|------|-------------|-------------|
| `"all"` | `None` | No step filtering |
| `"exact"` | `step_range=(step, step)` | Only entries at exact same step |
| `"window"` | `step_range=(step - window, step + window)` | Entries within a time window |

### Existing Implementation: SimpleKnnStrategy

```python
SimpleKnnStrategy(
    storage=cache_storage,          # receives storage reference
    top_k=1,
    step_filter="all",              # "all" | "exact" | "window"
    step_window=5,
    rrf_k=60,                       # Qdrant RRF fusion param
    fusion_weights={"vision_0": 1.0, ...},
    candidate_multiplier=5,         # prefetch = top_k * multiplier
)
```

### Constraints

- **Must call `self._storage.search(spec)`** — never call backend directly
- **Do not make hit/miss decisions** — that's the Judge's job
- Do not re-sort results (storage returns them already sorted)
- `backend_hints` are pass-through; backends that don't understand them will ignore them

### Registration

1. Add type string to search_strategy validation in `config.py`
2. Add factory branch in `_build_search_strategy()` (line ~536)
3. Note: the factory passes `storage` and `fusion_weights` to the constructor

---

## 8. Storage Layer

**Source**: `src/openpi/cache/cache_storage.py`

`CacheStorage` is a thread-safe facade over any `VectorStoreBackend`. Application code (Orchestrator, SearchStrategy) uses `CacheStorage`, never backends directly.

### Responsibilities

1. **Thread safety**: `RLock` serialises all backend calls
2. **Dimension validation**: checks that `query_keys` shapes match `backend.vector_dims`
3. **Entry validation**: calls `entry.validate()` before insert (CP-specific invariants)
4. **Filter checking**: raises `UnsupportedFilterError` if backend doesn't support a requested filter
5. **Two-phase search**: `search()` returns lightweight results; `fetch_payload()` fetches tensors only for winners

### Key Methods

| Method | Description |
|--------|-------------|
| `search(spec: QuerySpec) → list[SearchResultLite]` | Phase 1: lightweight vector search |
| `fetch_payload(id: str) → CachePayload` | Phase 2: fetch full tensors for winner |
| `insert(entry: CacheEntry)` | Validate + upsert one entry |
| `batch_insert(entries) → BatchInsertResult` | Bulk insert (partial failures tolerated) |
| `count() → int` | Number of stored entries |
| `close()` | Release resources |

---

## 9. Adding a Custom Vector DB Backend

### 9.1 The VectorStoreBackend ABC

**Source**: `src/openpi/cache/backend_base.py`

```python
class VectorStoreBackend(ABC):

    # --- Metadata ---
    @property
    @abstractmethod
    def vector_dims(self) -> dict[str, int]:
        """Field names → embedding dimensions.
        Example: {"vision_0": 1024, "robot_state": 32}"""

    @abstractmethod
    def supported_filters(self) -> frozenset[str]:
        """Filter fields you support: e.g., frozenset({"checkpoint_id", "task_key"})"""

    # --- Core CRUD ---
    @abstractmethod
    def insert(self, entry: CacheEntry) -> None:
        """Upsert one entry. Same id → overwrite."""

    @abstractmethod
    def search(self, spec: QuerySpec) -> list[SearchResultLite]:
        """Return at most spec.top_k results, descending by score."""

    @abstractmethod
    def fetch_payload(self, id: str) -> CachePayload:
        """Fetch full payload. Raises KeyError if not found."""

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete by id. Non-existent ids silently ignored."""

    @abstractmethod
    def count(self) -> int:
        """Number of stored entries."""

    # --- Optional overrides ---
    def batch_insert(self, entries: list[CacheEntry]) -> BatchInsertResult:
        """Default: sequential inserts. Override for native bulk API."""

    def flush(self) -> None:
        """Persist buffered writes. No-op for remote stores."""

    def close(self) -> None:
        """Release resources. Calls flush() first."""
```

### 9.2 Data Types at the Boundary

**CacheEntry** (what you receive in `insert()`):
```python
entry.id: str                              # stable hash, use as primary key
entry.checkpoint_id: CheckpointID          # CP1 or CP3
entry.query_keys: dict[str, torch.Tensor]  # {field: [dim] CPU float32}
entry.payload: CachePayload                # action tensors to serialise
entry.timestamp: float                     # time.time()
```

**QuerySpec** (what you receive in `search()`):
```python
spec.query_keys: dict[str, torch.Tensor]          # vectors to match
spec.top_k: int                                    # max results
spec.checkpoint_id: Optional[CheckpointID]         # filter by checkpoint
spec.filters: Optional[QueryFilter]                # task_key, step_range
spec.fusion_weights: Optional[dict[str, float]]    # per-field weights for RRF
spec.backend_hints: Optional[dict[str, Any]]       # pass-through params
```

**Score contract**: `SearchResultLite.score` must be cosine similarity ∈ [-1, 1]. Higher = more similar. Convert your backend's native distance metric.

| Backend distance | Conversion |
|------------------|------------|
| Cosine similarity | Already correct |
| Cosine distance | `score = 1 - distance` |
| L2 distance | `score = 1 - distance² / 2` (for L2-normalised vectors) |
| Inner product | Already correct (for L2-normalised vectors) |

### 9.3 Implementation Skeleton (FAISS Example)

```python
# src/openpi/cache/backends/faiss_backend.py

import faiss
import torch
from openpi.cache.backend_base import VectorStoreBackend
from openpi.cache.storage_types import (
    CacheEntry, CachePayload, QuerySpec, SearchResultLite, BatchInsertResult,
)
from openpi.cache.types import CheckpointID

class FaissVectorStore(VectorStoreBackend):

    def __init__(self, vector_dims: dict[str, int]):
        self._dims = vector_dims
        # One FAISS index per field
        self._indexes: dict[str, faiss.IndexFlatIP] = {
            field: faiss.IndexFlatIP(dim) for field, dim in vector_dims.items()
        }
        self._entries: dict[str, CacheEntry] = {}  # id → entry
        self._id_to_idx: dict[str, int] = {}       # id → FAISS row index

    @property
    def vector_dims(self) -> dict[str, int]:
        return dict(self._dims)

    def supported_filters(self) -> frozenset[str]:
        return frozenset({"checkpoint_id"})  # declare what you support

    def insert(self, entry: CacheEntry) -> None:
        # Store entry and add vectors to FAISS indexes
        self._entries[entry.id] = entry
        idx = len(self._id_to_idx)
        self._id_to_idx[entry.id] = idx
        for field, vec in entry.query_keys.items():
            if field in self._indexes:
                self._indexes[field].add(vec.unsqueeze(0).numpy())

    def search(self, spec: QuerySpec) -> list[SearchResultLite]:
        # Search each field, combine scores
        # ... (implement scoring logic)
        pass

    def fetch_payload(self, id: str) -> CachePayload:
        if id not in self._entries:
            raise KeyError(f"Entry {id} not found")
        return self._entries[id].payload

    def delete(self, ids: list[str]) -> None:
        for id in ids:
            self._entries.pop(id, None)

    def count(self) -> int:
        return len(self._entries)
```

### 9.4 Method-by-Method Guidance

**`insert()`**: Extract vectors from `entry.query_keys` for fields in your `vector_dims`. Serialise `entry.payload` (CachePayload contains torch tensors — use `torch.save()` to bytes, then base64 or blob storage). Store `entry.checkpoint_id` and `entry.timestamp` as payload metadata.

**`search()`**: Read `spec.query_keys`, compute similarity for fields you support. If multi-field: implement fusion (RRF or weighted average). Apply `spec.filters` if supported. Convert scores to cosine similarity range. Return sorted descending, at most `spec.top_k`.

**`fetch_payload()`**: Deserialise and return `CachePayload`. Must reconstruct torch tensors on CPU.

**`supported_filters()`**: Only declare fields you genuinely handle. `CacheStorage` will reject queries with unsupported filters before calling you.

### 9.5 Wiring into the Config System

1. Add backend type to `_build_backend()` in `config.py` (line ~476):
```python
elif cfg.type == "faiss":
    from openpi.cache.backends.faiss_backend import FaissVectorStore
    return FaissVectorStore(vector_dims=cfg.vector_dims)
```

2. Add config dataclass if needed (like `QdrantConfig`):
```python
@dataclass
class FaissConfig:
    nprobe: int = 16
```

3. Add validation in `validate_cache_config()` for the new type

4. Update `cache.yaml`:
```yaml
backend:
  type: faiss
  vector_dims:
    robot_state: 32
```

### 9.6 Existing Backends

**InMemoryBackend** (`backends/in_memory_backend.py`, ~97 lines):
- Python dict storage, brute-force cosine similarity
- Single-field cosine only (ignores `fusion_weights` / `backend_hints`)
- Supported filters: `{"checkpoint_id"}` only
- Score range: [-1, 1]
- O(n) search — suitable for < 10k entries
- Use for: unit tests, integration tests, development

**QdrantVectorStore** (`backends/qdrant_backend.py`, ~455 lines):
- Named vectors: each field in `vector_dims` is a separate Qdrant named vector
- **Chunking**: Qdrant limits 65,535 dims per vector. Large fields (e.g., vision_0 at 524,288 dims) are split into chunks `field__chunk_000`, `field__chunk_001`, etc.
- **RRF fusion**: multiple chunks/fields combined via server-side Reciprocal Rank Fusion
- Tensor serialisation: `torch.save()` → base64 strings stored as Qdrant payload
- Supported filters: `{"checkpoint_id", "task_key", "step_range"}`
- Requires external Qdrant server

### 9.7 Testing Checklist

1. Insert one entry, count() returns 1
2. Insert then search with identical vector → score close to 1.0
3. Insert then fetch_payload → tensors match original
4. Search with top_k > count → returns all entries (no error)
5. Delete entry, count decreases, fetch_payload raises KeyError
6. Duplicate insert (same id) → count stays 1, payload updated
7. Search with supported filter → results filtered correctly

---

## 10. YAML Config System

### 10.1 Loading Pipeline

```
cache.yaml
  → _substitute_env_vars()     # ${VAR} and ${VAR:-default}
  → yaml.safe_load()           # Python dict
  → _dict_to_dataclass()       # Recursive conversion to CacheConfig
  → validate_cache_config()    # 7 cross-validation checks
  → build_cache_components()   # Factory → component instances
```

Entry point: `load_cache_config(path) → CacheConfig`
Factory: `build_cache_components(config) → dict` with keys: `timer`, `storage`, `key_builder`, `gates`, `judges`, `search_strategies`

### 10.2 Environment Variable Substitution

| Syntax | Behaviour |
|--------|-----------|
| `${QDRANT_URL}` | Read env var; error if unset |
| `${QDRANT_URL:-http://localhost:6333}` | Read env var; use default if unset |

Example from `cache.yaml`:
```yaml
url: ${QDRANT_URL:-http://155.98.36.13:6333}
```

### 10.3 Full Annotated Config Reference

```yaml
# Root switch — set false to disable entire cache system
enabled: true

# Timing system
timer:
  enabled: true           # Enable per-component timing probes
  buffer_size: 10000      # Circular buffer size for timing samples
  output_csv_dir: null    # null = terminal only; path = write CSV files

# Query vector field config
# enabled: whether this field participates in key building and search
# weight: fusion weight for SearchStrategy (higher = more important in RRF)
keys:
  vision_0:    { enabled: true,  weight: 1.0 }   # base_0_rgb [256 * emb_dim]
  vision_1:    { enabled: true,  weight: 1.0 }   # left_wrist_0_rgb
  vision_2:    { enabled: true,  weight: 1.0 }   # right_wrist_0_rgb
  prompt_emb:  { enabled: true,  weight: 1.0 }   # language tokens [max_lang * emb_dim]
  robot_state: { enabled: true,  weight: 1.0 }   # raw state vector [action_dim]

# Key builder type
key_builder:
  type: full_original     # "full_original" = raw flatten (multi-modal)
                          # "placeholder" = robot_state only (testing)

# Per-checkpoint config (CP1 and CP3 independently)
checkpoints:
  _defaults: &cp_defaults          # YAML anchor for shared defaults
    gate:
      type: always_search          # Only type available currently
    search_strategy:
      type: simple_knn
      top_k: 1                     # Max results to return
      step_filter: all             # "all" | "exact" | "window"
      step_window: 5               # Window size (only for step_filter=window)
      rrf_k: 60                    # Qdrant RRF param (in_memory ignores)
      candidate_multiplier: 5      # Qdrant prefetch = top_k * multiplier

  cp1:
    <<: *cp_defaults               # Inherit defaults
    enabled: true                  # Set false to disable CP1 entirely
    judge:
      type: always_hit             # "always_hit" | "threshold"
      # threshold: 0.98            # Only for type=threshold

  cp3:
    <<: *cp_defaults
    enabled: true
    judge:
      type: always_hit

# Vector store backend
backend:
  type: qdrant                     # "in_memory" | "qdrant"
  vector_dims:                     # Must match collection schema
    vision_0: 524288               # 256 * 2048
    vision_1: 524288
    vision_2: 524288
    prompt_emb: 409600             # 200 * 2048
    robot_state: 32
  qdrant:                          # Only used when type=qdrant
    url: ${QDRANT_URL:-http://155.98.36.13:6333}
    collection_name: openpi_steps_named
    prefer_grpc: false
    grpc_port: 6334
    request_timeout: 30
```

### 10.4 Cross-Validation Rules

`validate_cache_config()` checks these invariants:

| # | Rule | Error if violated |
|---|------|-------------------|
| 1 | Enabled keys must exist in `backend.vector_dims` | Missing field in vector_dims |
| 2 | `vector_dims` keys must be subset of `CACHE_QUERY_FIELDS` | Unknown field name |
| 3 | Checkpoint names must be `cp1` or `cp3` | Invalid checkpoint name |
| 4 | `key_builder.type` must be valid | Unknown key_builder.type |
| 5 | Gate/Judge/SearchStrategy types must be valid | Unknown component type |
| 6 | `PlaceholderKeyBuilder` only supports `robot_state` | Unsupported fields for placeholder |
| 7 | `step_filter` must be `all`, `exact`, or `window` | Invalid step_filter |

### 10.5 Instantiation Order

`build_cache_components()` creates instances in this order (order matters — SearchStrategy needs `storage`):

```
1. Timer
2. Backend → CacheStorage(backend)
3. KeyBuilder (receives enabled_fields, vector_dims)
4. Per-checkpoint: Gate, Judge, SearchStrategy(storage, fusion_weights)
```

### 10.6 CLI Usage

```bash
# Serve with cache enabled
uv run scripts/serve_policy.py --env aloha_sim --cache_config cache.yaml

# --cache_config overrides --cache and --timing_csv_dir flags
```

---

## 11. Orchestrator

**Source**: `src/openpi/cache/orchestrator.py`

### Constructor

```python
CacheOrchestrator(
    storage: CacheStorage,
    key_builder: QueryKeyBuilder,
    gates: dict[CheckpointID, GateFunction],          # per-checkpoint dispatch
    judges: dict[CheckpointID, SimilarityJudge],
    search_strategies: dict[CheckpointID, SearchStrategy],
    timer: Optional[SystemTimer] = None,
)
```

**Per-checkpoint dispatch**: Gates, Judges, and SearchStrategies are `dict[CheckpointID, Component]`. CP1 and CP3 can use different instances. If a checkpoint is not in the dict (disabled in YAML), `check()` returns MISS gracefully.

**KeyBuilder is shared** across checkpoints (same data extraction logic).

### Step Counter

- `_step_counter`: inference cycle count within a task
- Incremented after each CP1 check (not CP3)
- Reset by `on_task_begin()` / `on_episode_start()`
- Passed to SearchStrategy via `SearchContext.current_step`

### Stable Hash

`_stable_hash(checkpoint_id, query_keys)` → `SHA256(checkpoint_id.name + sorted_concat_of_key_bytes)[:32]`. Same observation at same checkpoint → same ID → idempotent upserts.

### CP3 Stubs (Step 6)

- `schedule_next_action(action)` — empty stub, `pass`
- `should_skip_inference()` — always returns `None`
- These will be implemented in Step 6 with a `DeferredWriter` pattern

---

## 12. Interceptor

**Source**: `src/openpi/cache/interceptor.py`

`InferenceInterceptor` wraps a `Policy` as a `BasePolicy` drop-in. The original policy's stage methods are compiled with `torch.compile(mode="max-autotune-no-cudagraphs")`.

### Inference Flow

```
infer(obs)
  ├─ Input transforms
  ├─ CP3 consume: should_skip_inference() → if action, return early [STUB]
  ├─ Stage 1 (vision + tokenisation)
  ├─ CP1 check → if FULL_HIT: return cached action, skip stages 2–3
  ├─ Stage 2 (LLM backbone)
  ├─ Stage 3 (flow matching)
  ├─ CP3 check → infrastructure validation [always MISS in Step 4]
  ├─ CP1 write → background thread (ThreadPoolExecutor, max_workers=1)
  └─ Output transforms → return actions
```

### Background Write

The main thread materialises tensors to CPU via `key_builder.build()`, then submits `write_with_keys()` to a background thread to avoid blocking inference.

### Task Lifecycle

| Method | When called | Effect |
|--------|-------------|--------|
| `on_task_begin()` | Client connection opens | Reset step counter |
| `on_episode_start()` | Episode starts | Reset step counter |
| `on_episode_end()` | Episode ends | (reserved) |
| `on_task_end()` | Client disconnects | Print timing summary |

---

## 13. Component Isolation Rules

| Rule | Reason |
|------|--------|
| Components must **NOT** import `config.py` | Config is the factory layer; components are pure logic |
| Gate / Judge must **NOT** call `CacheStorage` | Gate: predicate only. Judge: evaluator only. Neither does IO. |
| Only `SearchStrategy` calls `storage.search()` | Single query construction point — ensures QuerySpec is built in one place |
| Only `Orchestrator` calls `storage.insert()` / `storage.fetch_payload()` | Write and payload retrieval are not delegated to components |

---

## 14. Testing Patterns

### Gate Test

```python
from openpi.cache.components.gate import AlwaysSearchGate
from openpi.cache.types import CheckpointID

gate = AlwaysSearchGate()
assert gate(CheckpointID.CP1, {}) is True
```

### Judge Test

```python
from openpi.cache.components.judge import ThresholdJudge, HitType
from openpi.cache.storage_types import SearchResultLite
from openpi.cache.types import CheckpointID

judge = ThresholdJudge(cp1_threshold=0.98)
results = [SearchResultLite(id="abc", score=0.99, checkpoint_id=CheckpointID.CP1)]
hit_type, winner = judge(results, CheckpointID.CP1, {})
assert hit_type == HitType.FULL_HIT
assert winner == "abc"
```

### SearchStrategy Test (with InMemoryBackend)

```python
from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.search_strategy import SimpleKnnStrategy, SearchContext

backend = InMemoryBackend(vector_dims={"robot_state": 32})
storage = CacheStorage(backend)
strategy = SimpleKnnStrategy(storage, top_k=1)

# Insert an entry, then search
# ...
ctx = SearchContext(query_keys={"robot_state": query_vec}, checkpoint_id=CheckpointID.CP1)
results = strategy.search(ctx)
```

### Backend Test

Use the 7-test checklist from [Section 9.7](#97-testing-checklist). `InMemoryBackend` is the reference implementation for comparing behaviour.

### Direct Construction

For unit tests, bypass the YAML config and construct components directly:
```python
orchestrator = CacheOrchestrator(
    storage=CacheStorage(InMemoryBackend({"robot_state": 32})),
    key_builder=PlaceholderKeyBuilder(),
    gates={CheckpointID.CP1: AlwaysSearchGate()},
    judges={CheckpointID.CP1: ThresholdJudge(cp1_threshold=0.95)},
    search_strategies={CheckpointID.CP1: SimpleKnnStrategy(storage, top_k=1)},
)
```

---

## 15. Current Validation Status

| Checkpoint | Status | Details |
|------------|--------|---------|
| **CP1** | **Validated** | End-to-end validated with current `cache.yaml` on this repo. Full pipeline works: Stage 1 → CP1 check → cache hit returns action → stages 2–3 skipped. Background write confirmed. |
| **CP3** | **Skeleton only** | Check infrastructure runs (`orchestrator.check(CP3)` executes gate → build → search → judge). No CP3 entries are written in the current write path, so CP3 always returns MISS. Next-cycle inference skip (`schedule_next_action` / `should_skip_inference`) are stubs — not attempted. |
| **CP2** | **Suspended** | Not implemented. `HitType.WARM_START` commented out. |
