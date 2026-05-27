# Pi0.5 Inference Cache System - Architecture Specification

> Version: 0.5
> Status: Implemented — CP1 three-level judgment (FULL_HIT / WARM_START / MISS), CP2 suspended (warm start migrated to CP1)
> Scope: PyTorch inference pipeline only (JAX path disabled)
> Last updated: 2026-04-10
>
> **Reading guide:** This document covers architecture principles and component design.
> For YAML configuration, testing patterns, and hands-on tutorial, see [../cache/tutorial.md](../cache/tutorial.md).
> For implementation history and step-by-step logs, see [logs/README.md](../../logs/README.md).
>
> **AGENT: READ FIRST** — This file is a registered subsystem rule document per [`WORKING_AGREEMENT.md` §8](../../WORKING_AGREEMENT.md#8-subsystem-rules). Design principles and component boundaries carry Working Agreement authority.

---

## 1. System Goals

Introduce a multi-level cache system into the Pi0.5 inference pipeline to reduce redundant computation by reusing historical results, thereby lowering end-to-end latency. The design adheres to the following principles:

1. **Decoupled from inference pipeline**: The cache system operates as an external plugin, hooking into the inference pipeline via an interceptor pattern without modifying existing inference code internals.
2. **Multi-level progressive hit**: Cache checkpoints are placed at three key positions in the pipeline — earlier hits save more computation.
3. **Backend-agnostic storage**: The storage layer uses a `VectorStoreBackend` ABC to decouple upper-layer logic from any specific vector DB. Swapping backends (Qdrant, FAISS, TorchGPU, etc.) requires zero changes to orchestrator or business code. *(Deferred: GPU/CPU hybrid data distribution and async transfer — see Section 7)*
4. **Precise timing**: Each component (retrieval, judgment, data transfer) is independently timed to support informed performance optimization decisions.
5. **Incremental implementation**: Start with single-machine, single-task repetitive scenarios, then gradually extend to multi-task/multi-robot/distributed settings.

---

## 2. Three-Stage Inference Pipeline Model

Based on the Pi0.5 PyTorch inference path (`src/openpi/models_pytorch/pi0_pytorch.py`), inference is divided into three stages:

```
Stage 1: Token Preparation        Stage 2: LLM Backbone         Stage 3: Action Expert
+---------------------------+     +------------------------+     +---------------------------+
| SigLIP vision encoder     |     | Gemma 2B (PaliGemma)   |     | Gemma 300M + adaRMSNorm   |
| Prompt tokenization       |     | Prefix-LM attention    |     | Flow matching (10 steps)  |
| State discretization      |     | Fill prefix KV cache   |     | Euler ODE: x1 -> x0      |
| -> prefix tokens + KV     |     | (no autoregressive gen)|     | -> action chunk [50, 32]  |
+---------------------------+     +------------------------+     +---------------------------+
            |                                |                                |
         [CP1]                            [CP2]                           [CP3]
      Cache Check 1                   Cache Check 2                   Cache Check 3
```

**Estimated compute per stage (single inference)**:

| Stage | Primary Computation | Parameters | Characteristics |
|-------|---------------------|------------|-----------------|
| 1. Token Prep | SigLIP forward + tokenize | ~400M | Single forward pass, parallelizable |
| 2. LLM Backbone | Gemma 2B prefix forward (KV fill) | ~2B | Single forward pass, fills KV cache (no autoregressive generation in PyTorch path) |
| 3. Action Expert | 10x Gemma 300M forward | ~300M x10 | Iterative, partially skippable |

---

## 3. Checkpoint Semantics

### CP1: After Vision

- **Trigger**: Stage 1 complete; prefix tokens and KV cache generated.
- **Available information**: Vision embedding, prompt embedding, state embedding.
- **Three-level judgment** (configured via `warm_tiers`):
  - **FULL_HIT**: Skip Stage 2 + Stage 3, directly output cached action chunk.
  - **WARM_START**: Run Stage 2, then partial Stage 3 from cached intermediate `x_t` via `run_stage3_from()`. Judge decides `start_t` based on similarity score tiers.
  - **MISS**: Run full Stage 2 + Stage 3 (with intermediates collection for future warm starts).
- **Savings**: FULL_HIT is maximum; WARM_START saves a fraction of Stage 3 (e.g., start_t=0.3 saves 70%).
- **Risk**: FULL_HIT is highest (skips subtask prediction); WARM_START is medium (subtask is fresh, only denoising is partial).
- **Applicable scenario**: Highly repetitive operations (FULL_HIT) or similar scenes (WARM_START).

### CP2: After LLM Backbone — ⚠️ Suspended

> Warm start functionality has been migrated to CP1. CP2 remains suspended due to lack of usable retrieval key (Stage 2 produces only an opaque KV cache).

- **Current status**: **Suspended** — no usable retrieval key; warm start now handled at CP1.

### CP3: After Action Expert

- **Trigger**: Stage 3 complete; current cycle's action chunk generated.
- **Available information**: All information (vision + prompt + state + action chunk).
- **Hit behavior**: Does NOT affect the current cycle's output. Determines whether the **next inference cycle** can be skipped, directly executing cached subsequent action chunks.
- **Savings**: Maximum (skip an entire next inference).
- **Risk**: Medium — depends on the accuracy of future state prediction.
- **Applicable scenario**: Scenarios where consecutive action sequences exhibit temporal locality (e.g., the middle phase of a long object transport).

### Checkpoint Relationship Diagram

```
                    Inference Cycle N                          Cycle N+1
            ┌─────────┬─────────┬──────────┐          ┌──────────────────┐
            │ Stage 1 │ Stage 2 │ Stage 3  │          │ Stage 1,2,3      │
            │ Vision  │  LLM    │ FlowMatch│          │ (may be skipped) │
            └────┬────┴────┬────┴─────┬────┘          └────────┬─────────┘
                 │         │          │                         │
              [CP1]  [CP2:suspended] [CP3]─── predict ────> skip?
                 │                    │
          hit: skip              hit: schedule
          S2+S3                  next cycle's
                                 action from cache
```

---

## 4. System Architecture

### 4.1 Top-Level Component Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        CacheOrchestrator                                 │
│  (controls all cache workflow, decoupled from inference)                  │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │ CP1 Handler  │  │ CP2 Handler  │  │ CP3 Handler  │                   │
│  │              │  │              │  │              │   CheckpointHandler│
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                   │
│         │                 │                 │                             │
│  ┌──────┴─────────────────┴─────────────────┴───────┐                   │
│  │              QueryKeyBuilder (pluggable)          │                   │
│  │  Converts stage outputs -> fixed-dim query vector │                   │
│  │  (mean_pool / spatial_pool / clip / placeholder)  │                   │
│  └──────────────────────┬───────────────────────────┘                   │
│                         │                                                │
│  ┌──────────────────────┴───────────────────────────┐                   │
│  │              GateFunction (pluggable)             │                   │
│  │  Decides: should we even search cache?            │                   │
│  │  (always_search)                                  │                   │
│  └──────────────────────┬───────────────────────────┘                   │
│                         │                                                │
│  ┌──────────────────────┴───────────────────────────┐                   │
│  │              SearchStrategy (pluggable)           │                   │
│  │  Builds QuerySpec, calls storage, fuses results   │                   │
│  │  (weighted_rrf / weighted_score_sum)              │                   │
│  └──────────────────────┬───────────────────────────┘                   │
│                         │                                                │
│  ┌──────────────────────┴───────────────────────────┐                   │
│  │              SimilarityJudge (pluggable)          │                   │
│  │  Given search results, decide: hit or miss?       │                   │
│  │  (threshold / always_hit)                         │                   │
│  └──────────────────────┬───────────────────────────┘                   │
│                         │                                                │
│  ┌──────────────────────┴───────────────────────────┐                   │
│  │              WritePolicy (pluggable)              │                   │
│  │  Episode-end decision: write trajectory or not?   │                   │
│  │  (on_any_miss / always / never)                   │                   │
│  └──────────────────────────────────────────────────┘                   │
└──────────────────────────┬───────────────────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          │    CacheStorage (facade)         │
          │  ┌───────────────┐ ┌──────────┐ │
          │  │VectorStore-   │ │MetadataDB│ │
          │  │Backend (ABC)  │ │(reserved,│ │
          │  │               │─│ not impl)│ │
          │  │┄InMemory(now)┄│ └──────────┘ │
          │  │┄Qdrant(depr.) │               │
          │  │┄FAISS (future)│               │
          │  └───────────────┘               │
          └─────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                     InferencePipeline (existing, unmodified)              │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐               │
│  │  Stage 1    │───>│   Stage 2    │───>│   Stage 3     │               │
│  │  Vision     │    │   LLM        │    │   FlowMatch   │               │
│  └─────────────┘    └──────────────┘    └───────────────┘               │
└──────────────────────────────────────────────────────────────────────────┘
          ^                                         |
          |          Hook / Interceptor              |
          └─────── CacheOrchestrator ───────────────┘
```

### 4.2 Integration with the Inference Pipeline

The cache system hooks in via the **Interceptor pattern**, without modifying `PI0Pytorch` internals.

> **Step 1 implementation note**: `InferenceInterceptor` was implemented in Step 1 (commit `a6c9f43`) as a `BasePolicy` subclass. It accesses `PI0Pytorch` via `self._model` (borrowed from the wrapped `Policy`), and manages the transform pipeline itself. It does **not** call `self.policy.run_stage*()` — instead it calls `self._model.run_stage1/2/3()` directly.

```python
class InferenceInterceptor(BasePolicy):
    """Wraps a Policy, routing inference through the staged API.

    Implements BasePolicy so it is a transparent drop-in for WebsocketPolicyServer.
    Activated via `--cache` flag on serve_policy.py.
    """

    def __init__(self, policy: Policy):
        # Borrow references (not copies) from the wrapped policy
        self._model = policy._model          # PI0Pytorch instance
        self._input_transform = policy._input_transform
        self._output_transform = policy._output_transform
        self._timer = SystemTimer()          # Step 2: CUDA event timing

    def infer(self, observation: dict, *, noise=None) -> dict:
        """Cache-aware inference. Replaces direct policy.infer() calls."""

        # --- Stage 1: Token Preparation ---
        with self._timer.measure("stage1_vision"):
            stage1_output = self._model.run_stage1(observation)

        # --- CP1: Cache Check (TODO: Step 4+) ---
        # cp1_result = self.orchestrator.check(CP1, stage1_output)
        # if cp1_result.hit: return cp1_result.cached_action

        # --- Stage 2: LLM Backbone ---
        with self._timer.measure("stage2_llm"):
            stage2_output = self._model.run_stage2(stage1_output)

        # --- CP2: Suspended ---
        # CP2 is suspended because Stage 2 produces only an opaque
        # past_key_values (DynamicCache), with no command embedding
        # available as a retrieval key. See Section 3 for details.

        # --- Stage 3: Action Expert (full flow matching) ---
        with self._timer.measure("stage3_flow"):
            stage3_output = self._model.run_stage3(stage2_output, noise=noise)

        # --- CP3: Predictive Cache Check (TODO: Step 4+) ---
        # cp3_result = self.orchestrator.check(CP3, ...)
        # if cp3_result.hit:
        #     self.orchestrator.schedule_next_action(cp3_result.cached_next_action)

        return stage3_output.action_chunk
```

Key design points:
- `PI0Pytorch.run_stage1/2/3` are **public typed wrappers** added on top of existing private `_stage1_token_prep`, `_stage2_llm_backbone`, `_stage3_action_expert` methods. The original `sample_actions()` is unmodified.
- `InferenceInterceptor` implements `BasePolicy`, making it a transparent drop-in — WebsocketPolicyServer and clients require zero changes.
- `return_intermediates=True` causes Stage 3 to return `x_t` at selected timesteps during flow matching, for future warm start caching.
- CP2 check is intentionally omitted (suspended) — see Section 3 for the rationale.

---

## 5. Core Component Detailed Design

### 5.1 CacheOrchestrator

> **Source**: `src/openpi/cache/orchestrator.py` | **Tests**: `tests/cache/test_orchestrator.py`

The master controller. Coordinates the gate → build → search → judge workflow for each checkpoint, manages episode lifecycle (buffer steps, batch write at episode end), and broadcasts actions to trajectory-aware components.

> With CP2 suspended, the orchestrator handles CP1 and CP3 only.

**Constructor** (per-checkpoint components):

```python
CacheOrchestrator(
    storage: CacheStorage,
    key_builder: QueryKeyBuilder,
    gates: dict[CheckpointID, GateFunction],
    judges: dict[CheckpointID, SimilarityJudge],
    search_strategies: dict[CheckpointID, SearchStrategy],
    timer: Optional[SystemTimer] = None,
    write_policy: Optional[WritePolicy] = None,
)
```

**Check pipeline** (`check(checkpoint_id, **stage_outputs) -> CheckResult`):

1. `key_builder.collect()` — hold GPU tensor references
2. `key_builder.build()` — GPU→CPU transfer (the only D2H copy)
3. `gate()` — decide search or skip
4. `search_strategy.search()` — build QuerySpec, call storage, return ranked results
5. `judge()` — threshold → FULL_HIT, warm_tiers → WARM_START, or MISS (returns `JudgeResult`)
6. On FULL_HIT/WARM_START: `storage.fetch_payload()` → return cached payload (WARM_START validated for intermediates completeness)

**Episode lifecycle**:

| Method | When | Effect |
|--------|------|--------|
| `on_episode_start(task_key, episode_id)` | Episode begins | Clear buffers, broadcast to all components |
| `broadcast_action(action)` | After each inference | Propagate to strategies/gates/judges for trajectory history |
| `buffer_for_write(query_keys, action)` | After each inference | Accumulate StepRecord |
| `on_episode_end()` | Episode ends | WritePolicy decision → build linked entry chain → batch_insert |

> **Design vs Implementation**: The original spec used `CacheContext` and `CacheResult` types. Implementation replaced them with keyword arguments (`**stage_outputs`) and `CheckResult` dataclass. `check()` returns `query_keys` on all paths (hit, miss, gate skip) so the caller can always call `buffer_for_write()`.

### 5.2 CacheStorage

> **Source**: `src/openpi/cache/cache_storage.py` | **Tests**: `tests/cache/test_cache_storage.py`

The storage layer facade. `CacheOrchestrator` is the sole consumer. Wraps a `VectorStoreBackend` (ABC) and an optional `MetadataDB` (reserved, not yet implemented).

**Why an ABC abstraction?** We do not yet know which vector DB we will use long-term. Qdrant is the current experiment backend, but it **will** be replaced. The `VectorStoreBackend` ABC decouples upper-layer logic from any specific DB, so swapping backends requires zero changes to `CacheOrchestrator` or other business code.

```python
class CacheStorage:
    """Storage facade. The only storage entry point for CacheOrchestrator.

    Responsibilities:
    - Thread safety (RLock on all backend calls)
    - Dimension validation (check query_key.shape on insert/search)
    - Filter capability check (fail-fast via supported_filters(), not silent ignore)
    - CacheEntry validation (call entry.validate())
    - Two-phase search (search returns SearchResultLite, fetch_payload on demand)
    - MetadataDB reserved (vector first, metadata second; vector is source of truth)
    """

    def __init__(self, backend: VectorStoreBackend, metadata_db=None):
        self._backend = backend
        self._metadata_db = metadata_db   # reserved, not yet used
        self._dims = backend.vector_dims   # dict[str, int] — per-field dimensions
        self._lock = threading.RLock()

    def search(self, spec: QuerySpec) -> list[SearchResultLite]:
        """Vector search, returns lightweight results (no payload)."""
        ...

    def fetch_payload(self, id: str) -> CachePayload:
        """Fetch full payload by id, called only for hit candidates."""
        ...

    def search_and_fetch(self, spec: QuerySpec) -> list[SearchResult]:
        """Convenience: search + fetch payload for all results."""
        ...

    def insert(self, entry: CacheEntry) -> None:
        """Validates entry, then delegates to backend.insert()."""
        ...

    def batch_insert(self, entries: list[CacheEntry]) -> BatchInsertResult: ...
    def delete(self, ids: list[str]) -> None: ...
    def count(self) -> int: ...
    def close(self) -> None: ...
```

Key design points:
- **Two-phase search**: `search()` returns `SearchResultLite` (score only, no payload tensors). Only hit candidates call `fetch_payload()` to retrieve full `CachePayload`, avoiding unnecessary data transfer.
- **Filter fail-fast**: `CacheStorage` checks `spec.filters` against `backend.supported_filters()` before calling `search()`. Unsupported filters raise `UnsupportedFilterError` instead of being silently ignored.
- **Dimension validation**: Every `insert()` and `search()` call validates each field in `query_keys` against `backend.vector_dims`.

### 5.3 VectorStoreBackend

> **Source**: `src/openpi/cache/backend_base.py` + `src/openpi/cache/backends/`

Two backends exist:

| Backend | Source | Use case |
|---------|--------|----------|
| **InMemoryBackend** | `backends/in_memory_backend.py` | Primary. Brute-force search, multi-field fusion (RRF + score_sum), trajectory search, artifact preloading from pickle. Suitable for < 50k entries. |
| **QdrantVectorStore** | `backends/qdrant_backend.py` | Deprecated. Remote Qdrant server via HTTP/gRPC. Does NOT support trajectory search. |

```python
class VectorStoreBackend(ABC):
    """Minimal interface for vector storage backends.

    Contracts:
    - insert() is idempotent: same id re-insert overwrites silently
    - search() returns SearchResultLite (no payload), score is normalized
      cosine similarity in [-1, 1] — all backends must convert
    - search() returns at most top_k; client-side filter may reduce count
    - fetch_payload() retrieves full CachePayload by id
    - delete() tolerates non-existent ids
    - Implementations are NOT thread-safe; CacheStorage handles locking
    """

    @property
    @abstractmethod
    def vector_dims(self) -> dict[str, int]:
        """Field names → embedding dimensions. Keys are a subset of CACHE_QUERY_FIELDS."""
        ...

    @abstractmethod
    def supported_filters(self) -> frozenset[str]: ...

    @abstractmethod
    def insert(self, entry: CacheEntry) -> None: ...

    @abstractmethod
    def search(self, spec: QuerySpec) -> list[SearchResultLite]: ...

    @abstractmethod
    def fetch_payload(self, id: str) -> CachePayload: ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None: ...

    @abstractmethod
    def count(self) -> int: ...

    # --- Optional (have default implementations) ---
    def batch_insert(self, entries: list[CacheEntry]) -> BatchInsertResult: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...

    # --- Cache session capability (§5.10) ---
    # Default no-op. A backend that implements a cross-step score cache
    # (e.g. InMemoryBackend) overrides these to register / clear the
    # per-strategy active session.
    def open_search_session(self, session_id: str) -> None: ...
    def close_search_session(self, session_id: str) -> None: ...
```

**Current backend — QdrantVectorStore** (`src/openpi/cache/backends/qdrant_backend.py`):
- Connects to a remote Qdrant server via HTTP/gRPC
- Collection must be pre-created by ops; the backend does NOT auto-create
- Serialization: tensor → `torch.save` bytes → base64 string → Qdrant JSON payload
- Supported filters: `checkpoint_id`, `task_key`, `step_range`
- Score: Qdrant Cosine distance is already in [-1, 1], used directly

**Why the ABC interface is intentionally small**: Payload filter syntax, gRPC options, fusion strategy (RRF weights, etc.) — these are all backend-specific details, configured in each backend's `__init__` config, never exposed above the ABC boundary. Upper-layer code only asks: "given named query vectors (`dict[str, Tensor]`), return the top-k most similar entries." How multiple fields are fused (e.g., RRF, weighted average) is the backend's internal decision.

#### 5.3.1 Deferred Design: GPU/CPU Hybrid VectorStore

> **Status**: Deferred — not implemented. The design below is preserved for future development when a local high-performance vector store is needed (e.g., FAISS GPU index + CPU fallback). Currently all storage goes through the remote Qdrant backend.

```python
class VectorStore:
    """Hybrid GPU/CPU vector store for cache lookup.

    Design rationale:
    - Hot data (frequently accessed, recent) lives on GPU for fast search.
    - Cold data lives on CPU, searched when GPU partition misses.
    - Transfers between CPU/GPU use pinned memory + CUDA streams
      to avoid blocking the main inference CUDA stream.
    """

    def __init__(self, config: VectorStoreConfig):
        self.dim = config.embedding_dim
        self.gpu_capacity = config.gpu_capacity
        self.cpu_capacity = config.cpu_capacity
        self.gpu_vectors: torch.Tensor  # [gpu_capacity, dim] on cuda
        self.cpu_index: faiss.Index     # CPU-resident index
        self._transfer_stream = torch.cuda.Stream()

    def search(self, query, top_k):
        """Search GPU first, then CPU if needed. Both can run concurrently."""
        ...

    def promote_to_gpu(self, cpu_ids): ...
    def demote_to_cpu(self, gpu_ids): ...
```

**GPU VRAM budget management**: VectorStore controls VRAM usage via a hard `gpu_capacity` cap. Actual usage = `gpu_capacity * dim * sizeof(float16)` bytes. For example, 10k entries x 1024 dim x 2 bytes = **20MB**, negligible impact on inference VRAM.

### 5.4 QueryKeyBuilder (Pluggable)

> **Source**: `src/openpi/cache/components/key_builder.py`, `src/openpi/cache/components/clip_key_builder.py`

Converts stage outputs into named query vectors (`dict[str, torch.Tensor]`). Two-phase API:

1. `collect(checkpoint_id, stage1=...)` — hold GPU tensor references (no copy)
2. `build(checkpoint_id)` — reduce + transfer to CPU → `dict[str, Tensor]`

**Implementations:**

| Type | Source | Vision dims | Method |
|------|--------|-------------|--------|
| `placeholder` | `key_builder.py` | — | robot_state only (for testing) |
| `cp1_mean_pool` | `key_builder.py` | 2048 | Mean pool over 256 vision tokens |
| `cp1_spatial_pool_16` | `key_builder.py` | 32768 | 4×4 spatial grid → 16 tokens × 2048 |
| `cp1_spatial_pool_4` (alias `cp1_spatial_pool_64`) | `key_builder.py` | 8192 | 2×2 spatial grid → 4 tokens × 2048 |
| `cp1_max_pool` | `key_builder.py` | 2048 | Per-dimension max over tokens |
| `clip` | `clip_key_builder.py` | 512 (ViT-B-32) | CLIP image encoder on raw input images |
| `full_original` | `key_builder.py` | 524288 | Raw flatten (Qdrant only, deprecated) |

All CP1 builders extract from `Stage1Output.prefix_embs` using token layout offsets. CLIP builder encodes raw input images via open_clip instead.

> **Design vs Implementation**: The original spec used `nn.Linear` projections to a configurable `output_dim`. Implementation uses direct pooling (mean/spatial/max) without learned projections — this preserves the original embedding space and avoids training a separate projection layer. `CacheContext` was not adopted; `collect()` + `build()` two-phase API replaces it.

### 5.5 GateFunction (Pluggable)

> **Source**: `src/openpi/cache/components/gate.py`

Decides whether to search at a given checkpoint.

**Current implementation**: `AlwaysSearchGate` — always returns True. Also has episode lifecycle methods (`on_episode_start()`, `record_action()`) for future stateful gates.

> **Design vs Implementation**: The original spec included `IntervalGate` and `StateChangeGate`. These are not implemented — the current experiment phase uses `always_search` to ensure every step is evaluated. Stateful gates remain viable future extensions.

### 5.6 SimilarityJudge (Pluggable)

> **Source**: `src/openpi/cache/components/judge.py`

Determines whether search results constitute a valid hit. Returns `JudgeResult(hit_type, winner_id, start_t)`.

**Implementations:**

| Type | Behavior |
|------|----------|
| `threshold` | FULL_HIT if `score >= threshold`. With `warm_tiers` configured, scores below the threshold are matched against descending tiers for WARM_START (CP1 only). Returns `JudgeResult(hit_type, winner_id, start_t)`. |
| `always_hit` | Always returns FULL_HIT for top result. Used in experiments (threshold calibration deferred). |
| `always_warm_start` | Always returns WARM_START with a fixed `start_t` for the top result (CP1 only). Used to sweep success-rate vs `start_t` curves under a forced warm-start regime. |
| `composite` | Aggregates pluggable verdict factors (statistical / kinematic descriptors) through a Composer + optional Normalizer pipeline. See §5.12 (Verdict Factor System) for the architecture. F1a-A / F1a-T / F2 + Composers + Normalizer enabled in B1; F1b OnlineExtractor + OfflineWriter + `LibraryStats` land in B2. |

Score semantics depend on fusion method — see [../cache/tutorial.md §6](../cache/tutorial.md#6-component-judge) for details.

> Judge returns `JudgeResult` (not a tuple). Payload fetch is done by the orchestrator after judge returns. On WARM_START, the orchestrator validates payload completeness (intermediates exist, start_t is a valid key) and downgrades to MISS if validation fails.

#### Purity contract

A judge MUST NOT write to `CacheStorage`. The original phrasing "DOES NOT call CacheStorage" is refined to: **no write to storage; read-only access via the optional `PayloadView` parameter is permitted at verdict time** so composite judges can compute factor descriptors over candidate payloads + neighbor entries. The contract preserves the no-side-effects spirit while admitting the read-only fetch path required by §5.12 verdict factors.

The `__call__` signature accepts two keyword-only parameters that orchestrator may inject:
- `view: PayloadView | None` — read-only facade described in §5.11.
- `history: HistoryView | None` — per-episode action / state snapshot.

Legacy judges (`ThresholdJudge` / `AlwaysHitJudge` / `AlwaysWarmStartJudge`) accept and ignore both via `**kwargs`; only `CompositeJudge` consumes them. Orchestrator builds and injects the facades from B1+; in B0 both arrive as None.

### 5.7 Cache Data Model

> **Source**: `src/openpi/cache/storage_types.py` + `src/openpi/cache/types.py`

```python
# src/openpi/cache/types.py
class CheckpointID(Enum):
    CP1 = auto()
    CP2 = auto()
    CP3 = auto()

# src/openpi/cache/storage_types.py

@dataclass
class CachePayload:
    action_chunk: torch.Tensor                          # [action_horizon, action_dim]
    intermediates: Optional[dict[float, torch.Tensor]]  # {t: x_t}, CP1 warm start
    denoising_num_steps: Optional[int]                  # for warm start
    next_action_chunk: Optional[torch.Tensor]           # CP3 only
    task_key: str = ""
    factors: Optional[dict[str, float]] = None
    # Pre-computed verdict factor descriptors populated by the offline-write
    # path of the verdict factor system (e.g. F1b SourceWindowSmoothness).
    # OnlineExtractor implementations read this dict on cache hits. Old
    # entries leave it as None — extractors propagate NaN for missing keys.
    # Key naming follows per-factor templates documented in §5.12.

@dataclass
class CacheEntry:
    id: str                                              # semantic dedup key
    checkpoint_id: CheckpointID
    query_keys: dict[str, torch.Tensor]                  # {field: [dim] CPU float32}
    payload: CachePayload
    step_idx: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    # Trajectory linked-list fields
    prev_ids: list[str] = field(default_factory=list)    # previous step's entry id
    next_ids: list[str] = field(default_factory=list)    # next step's entry id
    trajectory_id: Optional[str] = None                  # shared UUID within episode

@dataclass
class QuerySpec:
    query_keys: dict[str, torch.Tensor]
    top_k: int = 10
    checkpoint_id: Optional[CheckpointID] = None
    filters: Optional[QueryFilter] = None
    fusion_weights: Optional[dict[str, float]] = None    # per-field fusion weights
    trajectory_history: Optional[list[dict[str, torch.Tensor]]] = None  # newest-first
    trajectory_weights: Optional[list[float]] = None
    # Opt-in cross-step score memo (see §5.10 "Search Session — Cross-Step Score Memo").
    # Both fields appear together: a non-None search_session_id activates
    # the per-(field, query_id) score cache; trajectory_query_ids carries
    # the strategy-minted monotonic ids matching `trajectory_history`.
    search_session_id: Optional[str] = None
    trajectory_query_ids: Optional[list[int]] = None

@dataclass
class SearchResultLite:
    id: str
    score: float
    checkpoint_id: CheckpointID
```

> **Design vs Implementation**: The original `CacheEntry` had no trajectory fields. Implementation added `prev_ids`, `next_ids`, `trajectory_id`, `step_idx` to support trajectory-aware search. `QuerySpec` gained `fusion_weights`, `trajectory_history`, and `trajectory_weights` — constructed exclusively by `SearchStrategy`, not by the orchestrator.

### 5.8 SearchStrategy (Pluggable) — Not in Original Design

> **Source**: `src/openpi/cache/components/search_strategy.py`

Introduced during implementation to encapsulate the query construction logic that the original design placed inside the orchestrator's `check()` method. SearchStrategy owns `QuerySpec` construction — it decides fusion weights, trajectory fields, and step filtering.

**Implementations:**

| Type | Backend | Description |
|------|---------|-------------|
| `weighted_rrf_knn` | in_memory | Rank-based fusion across fields. Good for multi-field when magnitude doesn't matter. |
| `weighted_score_sum_knn` | in_memory | Similarity-based fusion. Better for trajectory search (preserves magnitude). |
| `qdrant_weighted_rrf_knn` | qdrant | Qdrant server-side RRF. Does NOT support trajectory search. |

All strategies inherit `TrajectoryMixin`, providing history buffer management (`record_query_keys()`, `on_episode_start()`, `_build_trajectory_fields()`).

> **Design vs Implementation**: The original spec had fusion logic inside the backend (`VectorStoreBackend.search()` was the "black box" that fused multiple fields). Implementation moved fusion control to `SearchStrategy` while keeping the actual computation in `InMemoryBackend`. This separation allows experimenting with different fusion strategies without changing the backend.

#### 5.8.1 Layer-1 Score Normalization (`weighted_score_sum`)

> **Source**: `src/openpi/cache/components/score_normalizers.py`

`weighted_score_sum` is a two-layer search. **Layer 1** maps each modality's raw geometric similarity (cosine value or L2 distance from `_batch_field_scores`) to a bounded, comparable scalar via a per-field `ScoreNormalizer`; **Layer 2** takes the magnitude-faithful weighted sum and returns top-k. The normalizers live in `score_normalizers.py` and are constructed at search time by `build_field_normalizers` from `QuerySpec.score_normalization`; `InMemoryBackend._search_weighted_score_sum` (and the trajectory paths, which build the normalizers once at entry and thread them through every chain layer) only apply them and sum.

**Design contract** — each normalizer is monotone non-decreasing in similarity (strictly increasing on its unsaturated interval); candidate normalizers all emit `[0,1]` so neither the weighted sum nor offline method selection is dominated by output scale; and **no empirical-CDF / rank equalization** (that would discard magnitude and collapse the sum into rank fusion ≈ weighted RRF).

**Candidate normalizers** (offline-selectable): `affine_clip`, `zscore` (mandatory bounded tanh squash), `logit`, `neg_log_one_minus`, `power` (the last three cosine-only, for near-1 cosine saturation), `exp_l2` (l2-only). **Back-compat only** (not selectable): `legacy_percentile` (reproduces the old `score_normalization.type: percentile` exactly — note the old p5/p95 live in the post-`(cos+1)/2` / `exp(-d/τ)` space, not raw cosine) and `direction_unify` (the old `type: none` path; backend/spec-level only — `weighted_score_sum_knn` config validation still rejects `type:none`).

`score_normalization` schema (`type: per_field`): `{"fields": {"vision_0": {"method": "logit", "params": {...}}, ...}}`. Parameters are fit offline against the real query-vs-library distribution by `exp/common/calibrate_score_normalizers.py` (see the [weighted_sum runbook](../experiments/weighted_sum.md)).

### 5.9 WritePolicy (Pluggable) — Not in Original Design

> **Source**: `src/openpi/cache/components/write_policy.py`

Controls whether an episode's data is written to the cache at episode end.

| Type | Behavior |
|------|----------|
| `on_any_miss` | Write if the episode had any CP1 cache miss (default) |
| `always` | Always write |
| `never` | Read-only mode |

> **Design vs Implementation**: The original spec had write logic in `CacheOrchestrator.write_async()` with per-step writes. Implementation switched to episode-level batch writes with a `WritePolicy` decision gate, which better supports trajectory-linked entry chains.

### 5.10 Search Session — Cross-Step Score Memo

> **Source**: `logs/trajectory_search_optimization_plan.log.md`
> Added by the trajectory-search rewrite to amortise per-field cosine
> similarity across the steps of a single episode. The capability is
> opt-in: callers that do not engage the cache observe trunk behavior
> exactly.

**Why a session, not a process-wide cache.**
A trajectory search at step *t* re-uses the same query vectors that
appeared at steps *t-1, t-2, …*. Computing those cosines once per
`(field, query_id)` slot and reusing them across the trajectory layers
removes the dominant cost of multi-layer search. Because every
SearchStrategy mints a fresh `search_session_id` per episode, the cache
is naturally bounded and per-episode disjoint — there is no eviction
policy and no global LRU.

**Mutation contract (runtime invariant).**
The cache is correct only if existing score slots cannot be invalidated
silently. While any session is active, mutations that could invalidate
cached scores raise `SearchSessionActiveError`:

| Operation | Active session | Idle |
|-----------|---------------|------|
| `insert(brand-new id)` | ✅ allowed (no slot affected) | ✅ allowed |
| `insert(existing id)` (upsert) | ❌ raises | ✅ allowed |
| `delete(ids)` | ❌ raises | ✅ allowed |
| `load_artifact(path)` | ❌ raises | ✅ allowed |

This matches the deployment contract: serving inserts new entries
freely; upsert / delete / artifact loading are offline-only operations.

**Session lifecycle (orchestrator-side, single helper per phase).**
Two helpers in `CacheOrchestrator` are the *only* paths allowed to
mutate `_current_strategy_session_ids`:

  1. `_broadcast_episode_start()` — invoked from both `on_task_begin`
     and `on_episode_start`. It atomically performs:
       (a) `_close_current_search_sessions()` to release any stale
           strategy sids left over from a previous episode,
       (b) broadcasts `on_episode_start` to every component (each
           SearchStrategy mints its own `uuid4().hex` sid and stashes
           it on itself),
       (c) collects each non-None sid via
           `strategy.get_search_session_id()` and registers it with the
           backend through `storage.open_search_session(sid)` *before*
           the first `search()` runs.
  2. `_close_current_search_sessions()` — the single cleanup helper
     called from `on_episode_end` (inside `try/finally` so that
     `_episode_steps` empty / `_write_policy is None` /
     `should_write()` declines all still release the sids),
     `on_task_end`, and step (a) above.

`InferenceInterceptor.on_task_end` forwards to
`CacheOrchestrator.on_task_end`, which gives WebSocket disconnects a
guaranteed cleanup path even when the simulator never produces a clean
`on_episode_end`.

**Active-session detection — independent set.**
`InMemoryBackend._active_search_sessions: set[str]` is populated by
`open_search_session` *before* the first cache bucket exists, so the
mutation guard fires from the moment the session is registered, not
from the moment a bucket is created. The score cache itself
(`_score_memo: dict[sid, dict[(field, query_id, sim_type), dict[entry_id, float]]]`)
is created lazily on first miss and dropped on `close_search_session`.

**Defensive layer for unregistered sids.**
`_batch_field_scores` checks `sid in self._active_search_sessions` and falls
back to the uncached path with a warning if a search arrives carrying
a sid that was never opened. This prevents an upstream lifecycle bug
from creating an orphan bucket; the search still returns correct
results because the uncached path is the legacy code path.

**Thread-safety contract (lock-free).**
The capability does NOT introduce any explicit lock. Lock-freedom is
preserved because:
  - `_active_search_sessions` and per-sid `_score_memo` buckets are mutated
    only through dict/set single-step operations (atomic under the GIL);
  - sids are per-strategy uuid4, so concurrent strategies write to
    disjoint outer keys and disjoint inner slots;
  - the mutation contract guarantees that no slot a worker reads can
    be evicted by another thread mid-search.

The detailed lock-free derivation lives in
`logs/trajectory_search_optimization_plan.log.md` §4.3 / §6.

### 5.11 PayloadView (Read-Only Judge-Side Facade)

> **Source**: `src/openpi/cache/components/payload_view.py`
> **Lifetime**: Per `Orchestrator.check()` call (a fresh instance is constructed per verdict; memo dictionaries do not leak across calls).

`PayloadView` wraps `CacheStorage` to give a judge read-only access to candidate payloads + neighbor entries without exposing the storage handle. The default implementation `StoragePayloadView` memoizes both `(entry_id -> payload)` and `(entry_id -> entry)` for the verdict's lifetime, so multiple extractors that touch the same entry ids fetch from the backend at most once.

**Protocol surface**:

```python
class PayloadView(Protocol):
    def get(self, entry_id: str) -> CachePayload: ...
    def get_entry(self, entry_id: str) -> CacheEntry: ...
    def get_many(self, entry_ids: list[str]) -> list[CachePayload]: ...

    def walk_prev(
        self, entry_id: str, k: int, *,
        fork_policy: ForkPolicy = ForkPolicy.TRAJECTORY,
        cross_trajectory: bool = False,
    ) -> list[CacheEntry]: ...

    def walk_next(
        self, entry_id: str, k: int, *,
        fork_policy: ForkPolicy = ForkPolicy.TRAJECTORY,
        cross_trajectory: bool = False,
    ) -> list[CacheEntry]: ...
```

**ForkPolicy values** (signature-only in B0 — only `TRAJECTORY` is supported; the others raise `NotImplementedError`):

| Value | Intended semantics |
|---|---|
| `TRAJECTORY` | Stay on the chain whose `trajectory_id` matches the anchor entry. Default. |
| `FIRST` | Pick `prev_ids[0]` / `next_ids[0]` deterministically. |
| `STOP` | Stop on the first fork; return what was collected. |
| `ALL_BRANCHES` | Walk every branch (returns `list[list[CacheEntry]]`). |
| `SCORE` | Pick the branch with highest similarity to the anchor query. |

In B0 the chain shape produced by the write path has `len(prev_ids) <= 1` and `len(next_ids) <= 1` per entry (no real fork yet). The walk implementation raises `NotImplementedError` immediately if a node with multiple neighbors is encountered, instead of silently picking one — a real fork should be a deliberate, observable design event.

**Backend capability dependency**: `walk_prev` / `walk_next` rely on `CacheStorage.fetch_entry`, which duck-types the backend's optional `fetch_entry(id) -> CacheEntry` capability. `InMemoryBackend` exposes it as a plain public method; the Backend ABC is unchanged. Backends without the capability raise `NotImplementedError` on the facade call. Composite judges that require chain walks (the 8 online factors in the refactor — see §5.12) are config-gated to backends that support `fetch_entry`; `validate_cache_config` rejects mismatched configurations at load time.

### 5.12 Verdict Factor System (4-layer architecture)

> **Source**: `src/openpi/cache/components/factors/` (Layer 1 / 2 / 3) + `src/openpi/cache/components/{composite,dumping}_judge.py` (Layer 4 + assembly).
> **Status**: Refactored 2026-05-07 (G1 APPROVED Round 4). The pre-refactor 5-factor family (`f1a_a` / `f1a_t` / `f1b_a` / `f1b_t` / `f2`) and the legacy `Normalizer` / `all_nan_fallback` / `cold_start_strategy` apparatus have been removed.

The CompositeJudge replaces a single-threshold cosine match with a four-layer pipeline. Each layer is independently pluggable through the yaml `judge` config; the layers are orthogonal — a layer never holds a reference to another layer's instance, only to an interface contract.

```
                    ┌──────────────────────────────────────┐
raw action / state │ Layer 1   Normalization              │
─────────────────► │   stats_source: offline (LibraryStats)│
                    └──────────────┬───────────────────────┘
                                   │ normalized data injected via FactorContext
                                   ▼
SearchResultLite[]  ┌──────────────────────────────────────┐
PayloadView      ──►│ Layer 2   17 Factors                  │
HistoryView         │   `<descriptor>_<source>_<channel>`   │
                    │   + topk_action_variance              │
                    └──────────────┬───────────────────────┘
                                   │ raw factor dict[str, float]
                                   ▼
                    ┌──────────────────────────────────────┐
                    │ Layer 3   Calibration (per key)       │
                    │   PercentileRollingCalibration        │
                    │   samples_source: offline | warmup    │
                    │   bind_keys() fail-fast at startup    │
                    └──────────────┬───────────────────────┘
                                   │ calibrated dict in [0, 1]
                                   ▼
                    ┌──────────────────────────────────────┐
                    │ Layer 4   Composer                    │
                    │   declared_dependencies (instance)    │
                    │   compose(calibrated, *, winner_id)   │
                    │   subclass owns NaN handling          │
                    └──────────────┬───────────────────────┘
                                   │
                                   ▼
                    JudgeResult(hit_type, winner_id, start_t,
                                factor_outputs={schema_version=2,
                                                raw, calibrated,
                                                composer_score})
```

**17 factors** = 4 descriptors (`jerk` / `direction` / `dispersion` / `path_length`) × 2 sources (`online` / `offline`) × 2 channels (`action` / `state`) + `topk_action_variance`. Factor naming: `<descriptor>_<source>_<channel>`. Key template (with windows): `<descriptor>_<source>_<channel>__p<P>_f<F>`.

**Online factors share a unified splice** `[history[-P:], winner, walk_next(F)]` (length `P + 1 + F`). On the action channel, every cell is `payload.action_chunk[0]`; on the state channel, every cell is `query_keys["robot_state"]`. The two channels are physically aligned (cell t at history / winner / chain step k always comes from the same inference quote).

**Offline factors** read precomputed values from `payload.factors[<key>]` at verdict time; the `OfflineWriter.compute_for_episode(entries, library_stats)` path produces those values during artifact build (`exp/common/factor_postprocess.py:enrich_artifact_with_factors`). Offline factors do not require chain walk at verdict time.

**No cold-start state**:

- Layer 1 reads `library_stats` (per-DOF σ + active mask) from `backend.load_artifact` at startup. There is no warmup-σ channel in this refactor (see plan §6.3 / §6.11 #10); attempting `stats_source.type=warmup` is a yaml-load error.
- Layer 3's `bind_keys` fails fast if any factor key has fewer non-NaN samples than `window_size`; both `samples_source.type=offline` (file-on-disk) and `=warmup` (per-yaml `WarmupPool` entry populated by sibling warmup yaml) must be saturated before the first verdict.
- The legacy `cold_start_strategy: force_miss/passthrough/lenient`, `all_nan_fallback: warm_start@t`, and `JudgeResult.factor_outputs.sentinel` fields are removed. WARM_START on all-NaN is now a property of specific Composer subclasses (e.g. `WeightedSumWithWarmFallbackComposer`).

**Empty-results short-circuit**: `CompositeJudge` returns `JudgeResult(MISS)` immediately when `results == []`; the Composer is not invoked with a `winner_id=None`.

**`min_required_top_k` plumbing**: each `Factor` may set `required_top_k`; CompositeJudge takes the max and exposes it as `min_required_top_k`; `build_per_connection_components` forwards it as `min_top_k_hint` to the search strategy so factors like `topk_action_variance` get enough candidates without the yaml widening `top_k`.

**`DumpingJudge`** (refactor §6.9 mode b) wraps any `SimilarityJudge` and side-channels per-verdict factor raw values to JSONL. It owns its own Layer 1 Normalization replica (config validator forces `dump.normalization.stats_source` == `judge.normalization.stats_source` so dump-factor raw values stay wire-comparable to inner factor raw). The independent dump-factor list lets a single warmup yaml over-collect every factor a downstream eval yaml family might use, while individual eval yamls each pick a subset.

For full per-layer Protocol signatures, factor implementation matrix, yaml schema, and 12+3 validator rules see `docs/cache/verdict_factor_judge.md` and the design log `logs/verdict_factor_judge_refactor.log.md` §6 / §11 / §13.

### 5.13 Wire-Level Observability + Warmup Preload Protocol

> **Source**: `src/openpi/cache/interceptor.py`, `src/openpi/serving/websocket_policy_server.py`, `src/openpi/cache/warmup_pool.py`, `src/openpi/cache/components/factors/calibrations/percentile_rolling.py`, `scripts/serve_policy.py`, `packages/openpi-client/src/openpi_client/websocket_client_policy.py`, `exp/verdict_factor_judge/run_phase.py`.
> **Status**: Refactored 2026-05-07 alongside §5.12. The wire schema is `schema_version=2`; the warmup channel that previously fed `PercentileRollingNormalizer.preload_buffer` now feeds Layer 3 `PercentileRollingCalibration` directly through `_build_calibration` (no separate post-build preload step).

Two operational gaps the verdict factor system needs to solve:

1. **Per-verdict observability is server-stdout only by default** — re-analysing any historical yaml requires re-running it because hit_type / start_t / winner_id / cp1_score live in transient logs. Mitigation: `judge.export_factor_outputs: true` + interceptor `__hit_meta__` wiring + client per-step jsonl recorder.
2. **Per-connection cold-start amplification** — each worker connection independently builds its own `CompositeJudge`, including its own Layer 3 calibration buffer. With high-NaN factor families a fresh buffer would either short-circuit verdicts or miscalibrate during the warmup phase. Mitigation: yaml-pair `<eval>__warmup.yaml` → `DumpingJudge` writes raw factor jsonl → runner `preload_normalizer_buffer` ctrl pushes per-key list into the per-process `WarmupPool` → eval yaml `_build_calibration` pulls from the pool at `bind_keys` time and fails fast if undersized.

#### Wire schema (`__hit_meta__["factor_outputs"]`, schema_version=2)

```python
{
  "schema_version": 2,
  "raw":             {key: float | None},     # Layer 2 raw factor dict (NaN -> None)
  "calibrated":      {key: float | None},     # Layer 3 percentile rank dict
  "composer_score":  float | None,            # Layer 4 internal aggregate
}
```

`hit_type` / `winner_id` / `start_t` / `cp1_score` remain at the top level of `__hit_meta__` (not inside `factor_outputs`). Legacy clients (pre-refactor) read `__hit_meta__["factor_outputs"]` via `dict.get("schema_version", 1)` — the missing field implicitly tags v1 (`{raw, norm, score, sentinel}`) and the client falls back to its v1 reader.

#### Warmup → eval handshake (Layer 3 Calibration only)

```
   warmup yaml run                                         eval yaml run
       │                                                       │
       │  inner = AlwaysWarmStartJudge                          │
       │  judge.dump = {factors: 17-factor superset}            │
       │       ↓ verdicts emit JSONL rows of raw factor values  │
       │       ↓                                                 │
   exp/.../run_phase.py                                          │
   reads JSONL → aggregates per-key list                         │
       │                                                       │
       ▼                                                       │
   ctrl `preload_normalizer_buffer`                              │
   server WarmupPool[eval_yaml_id] = {key: list[float]}          │
       │                                                       │
       └──────────────────► load_cache_config(eval_yaml) ──────►│
                                                               │
                            _build_calibration                  │
                            samples_source.type == warmup      │
                              → WarmupPool.get(eval_yaml_id)    │
                              → CalibrationSamples              │
                              → PercentileRollingCalibration    │
                                .bind_keys(union)               │
                                  fails if any key < window_size│
       │                                                       │
       ▼                                                       │
   `unload_warmup_buffer` ctrl drops the entry + the disk dump after the eval yaml is done.
```

The Layer 1 Normalization side does **not** participate in the warmup channel in this refactor: σ + active_mask come exclusively from `backend.load_artifact`'s `library_stats` field (deferred per plan §6.3 / §6.11 #10). The wire ctrl messages and `WarmupPool` schema are unchanged.

## 6. Data Flow and Timing

> **Note**: The data flow diagrams below reference cache search/write operations that depend on the storage layer (Section 5.2/5.3). The storage layer is ⚠️ unstable — interfaces and backend implementations will change. The timing structure (stages, checkpoint positions) is stable; the storage interaction details are not.

### 6.1 Full Inference Cycle (No Cache Hit)

> **Note**: The `[GPU Transfer Stream]` and `[CPU Thread Pool]` rows below represent the **target design** with GPU/CPU hybrid VectorStore (Deferred — see Section 7). With the current remote Qdrant backend, cache search runs synchronously on the main thread; there is no GPU vector partition or dedicated transfer stream.

```
Time ──────────────────────────────────────────────────────────────>

[GPU Main Stream]
│ Stage1 ││ Stage2 ││ Stage3 (10 denoise steps)          ││
│ Vision ││ LLM    ││ step1 step2 ... step10             ││
│        ││        ││                                     ││

[GPU Transfer Stream] (Deferred — GPU/CPU hybrid, see Section 7)
         ││  CP1   ││       CP2        ││            CP3  ││  write-back
         ││ search ││      search      ││           check ││  (async)

[CPU Thread Pool] (Deferred — GPU/CPU hybrid, see Section 7)
         ││ CP1 CPU││   CP2 CPU search ││ CP3 CPU search  ││ metadata write
         ││ search ││  (if GPU miss)   ││                 ││
```

### 6.2 CP1 Hit Timing

> **Note**: `[GPU Transfer Stream]` is Deferred (see Section 7). Current reality: CP1 search is a remote Qdrant call; cached action is fetched over network, not from GPU memory.

```
Time ──────────────────────────>

[GPU Main Stream]
│ Stage1 ││ (idle - stages 2,3 skipped)
│ Vision ││

[GPU Transfer Stream] (Deferred — currently: remote fetch from Qdrant)
         ││ CP1 search ──> HIT!
         ││ load cached action

Total: Stage1 + CP1 latency only
```

### 6.3 CP2 Warm Start Timing — ⚠️ Suspended (design preserved)

```
Time ──────────────────────────────────────────────>

[GPU Main Stream]
│ Stage1 ││ Stage2 ││ Partial Stage3 (3 steps)    ││
│ Vision ││ LLM    ││ from cached x_0.3           ││

[GPU Transfer Stream]
         ││ CP1    ││ CP2 search ──> WARM START HIT
         ││ miss   ││ load cached x_0.3

Total: Stage1 + Stage2 + CP2 latency + 3 denoise steps (instead of 10)
```

### 6.4 CP3 Predictive Hit Timing

```
Cycle N:                                             Cycle N+1:
│ Full inference │ CP3: match found ──> schedule │    │ Skip inference, use cached action │
                                                      │ (only run Stage1 for state update) │
```

---

## 7. Hardware Resource Allocation Strategy — Deferred

> **Status**: Deferred — not implemented. The designs below are preserved for future development when a local GPU/CPU hybrid vector store replaces the current remote Qdrant backend. Currently, all cache storage is remote; there is no GPU vector partition, no pinned memory pool, and no dedicated CUDA stream for cache operations.

### 7.1 GPU VRAM Layout

```
GPU VRAM (e.g., 24GB)
├── Model weights (fixed)          ~5 GB  (PaliGemma 2B + Action Expert 300M, bf16)
├── KV Cache (per inference)       ~1 GB  (varies with sequence length)
├── Activations (transient)        ~2 GB  (peak during forward pass)
├── VectorStore GPU partition      ~20 MB (10k entries x 1024 dim x fp16)
├── Transfer buffers (pinned)      ~10 MB
└── Free                           ~16 GB
```

VectorStore GPU usage is minimal and will not become a bottleneck.

### 7.2 CUDA Stream Isolation

```python
class CacheHardwareManager:
    """Manages CUDA resources for cache operations."""

    def __init__(self):
        # Separate stream for cache operations, does NOT block inference
        self.cache_stream = torch.cuda.Stream(priority=-1)  # low priority
        # Pinned memory pool for CPU<->GPU transfers
        self.pinned_pool = PinnedMemoryPool(size_mb=32)

    @contextmanager
    def cache_context(self):
        """Execute cache operations on dedicated stream."""
        with torch.cuda.stream(self.cache_stream):
            yield

    def async_to_gpu(self, tensor_cpu: torch.Tensor) -> torch.Tensor:
        """Non-blocking CPU->GPU transfer via pinned memory."""
        pinned = self.pinned_pool.allocate(tensor_cpu.shape, tensor_cpu.dtype)
        pinned.copy_(tensor_cpu)
        gpu_tensor = torch.empty_like(pinned, device="cuda")
        with torch.cuda.stream(self.cache_stream):
            gpu_tensor.copy_(pinned, non_blocking=True)
        return gpu_tensor
```

### 7.3 CPU Thread Allocation

```
Thread 0 (main):       Inference orchestration
Thread 1:              CPU-side vector search (FAISS)
Thread 2:              Cache write-back (vector DB insert + metadata)
Thread 3 (optional):   Cache maintenance (eviction, compaction)
```

---

## 8. Cache Management and Dynamic Optimization — Deferred

> **Status**: Deferred — not implemented. Write strategy, eviction policies, and GPU/CPU data migration are designed for the future GPU/CPU hybrid store (Section 5.3.1 / Section 7). With the current remote Qdrant backend, eviction is handled by Qdrant server-side or manual ops. These designs are preserved for future development.

### 8.1 Write Strategy

- **Online writes**: After each normal inference completes, async write to cache. Does not block the next inference.
- **Offline pre-fill**: Batch import from training data or offline rollouts.
- **Selective writes**: Not all inference results are worth caching. A `WriteFilter` decides:
  - If the current state is too similar to an existing cache entry (< threshold), don't write (avoid redundancy).
  - If action confidence is low (poor flow matching convergence), don't write.

### 8.2 Eviction Strategy

```python
class EvictionPolicy(Protocol):
    def select_evictions(self, store: VectorStore, count: int) -> list[int]:
        ...

class CompositeEviction(EvictionPolicy):
    """Combine multiple signals for eviction."""

    def select_evictions(self, store, count):
        scores = []
        for entry in store.entries():
            score = (
                0.4 * recency_score(entry.timestamp)    # LRU component
                + 0.3 * frequency_score(entry.hit_count) # LFU component
                + 0.3 * entry.quality_score               # quality component
            )
            scores.append((entry.id, score))
        scores.sort(key=lambda x: x[1])
        return [id for id, _ in scores[:count]]
```

### 8.3 GPU/CPU Data Migration Strategy

- **Promotion (CPU -> GPU)**: When a CPU entry is hit more than N times, promote it to the GPU partition.
- **Demotion (GPU -> CPU)**: When the GPU partition is full and a new high-frequency entry needs to be added, demote the lowest-frequency GPU entry to CPU.
- **Migration runs asynchronously on `cache_stream`, not blocking inference.**

---

## 9. Timing System — ✅ Implemented (Step 2)

> Implementation: `src/openpi/cache/timing.py` | Design log: `logs/archive/step2.log`

The timing system uses a **probe-based** architecture: each pipeline component registers a named probe at startup, specifying which backend to use. The `SystemTimer` then provides a zero-overhead `measure()` context manager for the hot path.

### 9.1 Backend Protocol

```python
class TimingBackend(Protocol):
    def start(self) -> Any: ...          # Returns an opaque timing handle
    def stop(self, handle: Any) -> float: ...  # Returns elapsed ms

class CudaEventBackend:
    """GPU timing via torch.cuda.Event.
    end_event.synchronize() blocks only until that event completes,
    without flushing the entire CUDA pipeline.
    Falls back to PerfCounterBackend when CUDA is unavailable."""
    def __init__(self, stream: torch.cuda.Stream | None = None): ...

class PerfCounterBackend:
    """CPU timing via time.perf_counter_ns. Sub-microsecond resolution."""
```

Key design decisions:
- **CudaEventBackend handle**: returns `("cuda", start_evt, end_evt)` tuple with type tag — supports concurrent/nested calls without per-instance state.
- **Auto-degradation**: `CudaEventBackend` holds an internal `PerfCounterBackend`; returns `("cpu", ns)` handle when no GPU is available, so CI environments work without changes.

### 9.2 SystemTimer

```python
class SystemTimer:
    def __init__(self, enabled=True, buffer_size=10000, output_csv_dir=None): ...
    def register_probe(self, name: str, backend: str = "cuda", stream=None): ...

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        # enabled=False → yield + return (zero overhead)
        # Unregistered probe → default CudaEventBackend (lenient mode, with warning)

    # TaskLifecycle (see 9.3)
    def on_task_begin(self) -> None: ...   # Record current ring buffer position
    def on_task_end(self) -> None: ...     # Print summary + optional CSV flush + task_id++

    def summary(self, task_only=True) -> dict[str, TimingStats]: ...
    def export_csv(self, path: str): ...   # Build all rows in memory, single write

    # Reserved stubs (Step 2)
    def add_resource_monitor(self, monitor: ResourceMonitor): ...
    def record_resource_snapshot(self, name: str): ...
```

Ring buffer uses a **monotonic append counter** (`_total_appended`) for task boundary tracking. `on_task_begin()` saves the counter value; `on_task_end()` slices from deque tail. Warns on ring buffer wrap.

The **total (sum) row** in the summary aligns the three stage records per-inference, sums each inference's latencies, then computes p50/p95/p99 over the per-inference totals (not a naive sum of per-stage means).

`_DISPLAY_ORDER` constant controls summary table row order; new probes only need to be added to this list.

### 9.3 TaskLifecycle Protocol

```python
@runtime_checkable
class TaskLifecycle(Protocol):
    def on_task_begin(self) -> None: ...
    def on_task_end(self) -> None: ...
```

`InferenceInterceptor` implements this protocol, forwarding to its internal `SystemTimer`. The server uses `hasattr(policy, 'on_task_begin')` to avoid a hard dependency.

**Server-side integration** (`websocket_policy_server.py`):
- Connection open → `policy.on_task_begin()`
- Connection close (normal or exception) → `policy.on_task_end()` → prints summary / writes CSV
- Removed: old `stage_timing_records` collection, `action.pop("stage_timing")`, manual mean printing (~15 lines deleted)

### 9.4 Registered Probes

| Probe | Backend | Status | Description |
|-------|---------|--------|-------------|
| `stage1_vision` | cuda | ✅ Registered | Vision encoder + tokenization |
| `stage2_llm` | cuda | ✅ Registered | LLM backbone prefix KV fill |
| `stage3_flow` | cuda | ✅ Registered | Full flow matching (10 denoise steps) |
| `total_inference` | cpu | ✅ Registered | Wall-clock total (outer `measure()` wrapping all 3 stages) |
| `cp1_*`, `cp3_*` | cpu | ✅ Registered (Step 4) — unstable | Cache sub-step probes (gate, build, search, judge, write) |
| `cp2_*` | — | Suspended | CP2 suspended (see Section 3) |
| `write_vectordb`, `write_metadata` | cpu | Planned | Async write-back |
| `gpu_to_cpu`, `cpu_to_gpu` | cuda | Planned | Data migration on `transfer_stream` |

### 9.5 Summary Output Example

```
=== Inference Timing Summary (task #0, 4 inferences) ===
  Probe                          N     mean      p50      p95      p99  ms
  ------------------------------------------------------------------------
  stage1_vision                  4      2.1      2.1      2.3      2.3
  stage2_llm                     4      6.1      6.1      6.1      6.1
  stage3_flow                    4      3.1      3.1      3.1      3.1
  total_inference                4     11.3     11.3     11.5     11.5
  ------------------------------------------------------------------------
  total (sum)                    4     11.3     11.2     11.4     11.5
```

---

## 9.X Concurrent serving runtime — C1 / C2 contracts, BackendPool, BatchingCoordinator

The concurrent serving optimization plan
([`logs/concurrent_serving_optimization_plan.log.md`](../../logs/concurrent_serving_optimization_plan.log.md))
adds five runtime modules outside the request-side cache pipeline. The
storage / orchestrator / judge boundaries described above are unchanged;
what follows is a one-stop reference for how the cache system sits inside
the new concurrent serving fabric.

### Hard constraints

* **C1 — Non-`--concurrent` baseline preserved.** ``Policy.infer`` and
  ``InferenceInterceptor.infer`` on the single-connection path run their
  legacy stage1/2/3 + post-stage3 CP3 sequence without any new wrapper or
  coordinator. Phase 5 flips ``--concurrent`` to the default, but the
  baseline path remains reachable via ``--non-concurrent``.
* **C2 — Runtime write-frozen.** All in-memory backends are ``freeze()``'d
  immediately after ``load_artifact`` (or after construction for backends
  without artifacts). Any subsequent ``insert / batch_insert / delete /
  upsert / load_artifact`` call raises ``BackendFrozenError``. Enforcement is
  **interface-side**: ``VectorStoreBackend.__init_subclass__`` auto-wraps every
  subclass mutation method with the frozen-guard, so the contract holds for any
  pluggable backend (including ones outside this repo) with zero per-backend
  ``_check_frozen`` boilerplate — concrete backends just define their mutation
  methods normally. Derived state mutation — per-session score memos,
  active-session sets, sample counters — remains allowed; these are search-path
  caches, not database content. ``scripts/serve_policy._enforce_runtime_write_policy``
  enforces ``write_policy`` **fail-fast**: any non-``"never"`` policy raises
  ``ConfigValidationError`` at server start and on every ``load_cache_config``
  ctrl, so a write-enabled config is surfaced loudly instead of silently
  neutralised under C2.

### BackendPool (M3)

Process-local singleton mapping ``BackendFingerprint(backend_type,
resolved_preload_path, vector_dims, index_type) → frozen Backend``. The
pool ensures identical fingerprints share one in-memory backend even when
multiple yamls (or multiple ``load_cache_config`` ctrls) reference the
same artifact pkl. Qdrant and empty-preload paths bypass the pool.
Per-fingerprint locks + double-check pattern guarantee exactly one
``load_artifact`` call per distinct fingerprint under concurrent first-load
races.

### BundleDispatcher (M2)

``WebsocketPolicyServer`` stores active bundles in a module-level
``_bundles: dict[bundle_id, CurrentCacheBundle]`` registry, addressed by
``load_cache_config`` (which accepts ``bundle_id``) and selected by a new
``select_bundle`` ctrl. Old clients that omit ``bundle_id`` fall back to a
``"default"`` slot; the legacy single-latest ``_current_bundle`` pointer
is kept synchronized for callers that read it without an id (``_wrap_policy``,
test fixtures).

### BatchingCoordinator (M1)

Three stage queues (one per stage1/stage2/stage3) drive dynamic batching
in concurrent mode. A request's per-connection thread keeps full
responsibility for transform + CP1/CP3 + payload assembly; the coordinator
only owns the GPU forward. Stage 3 sub-buckets requests by
``(mode, start_t, num_steps)`` so MISS (``run_stage3``) and WARM_START
(``run_stage3_from``, no noise arg) never mix in the same forward. CP3
remains post-stage3 and next-cycle predictive only.

### Wire-level protocol additions

* ``__ctrl__: select_bundle`` — client signals which loaded ``bundle_id``
  the connection binds to. Wrapper-stack creation is deferred until this
  ctrl (or first ``episode_start`` with ``bundle_id``) so the factory can
  receive the correct ``bundle_id``.
* ``__ctrl__: load_cache_config`` now carries optional ``bundle_id``;
  servers also auto-override ``write_policy`` to ``"never"`` on receipt.

For the implementation plan, files touched, and per-module test layout,
see [`logs/concurrent_serving_optimization_plan.log.md`](../../logs/concurrent_serving_optimization_plan.log.md).

---

## 10. Configuration, File Structure, and Implementation History

These sections have been moved to dedicated documents for maintainability:

| Topic | Document |
|-------|----------|
| **YAML config system** (full annotated reference, cross-validation rules, CLI usage) | [../cache/tutorial.md §10](../cache/tutorial.md#10-yaml-config-system) |
| **File structure** (current module tree) | [../reference/openpi.md](../reference/openpi.md) — Project Structure section |
| **Component isolation rules and testing patterns** | [../cache/tutorial.md §15-16](../cache/tutorial.md#15-component-isolation-rules) |
| **Implementation history** (step-by-step logs, modification boundaries) | [logs/README.md](../../logs/README.md) — Cache System Implementation section |
| **Development roadmap** (original step plan and current status) | [logs/README.md](../../logs/README.md) |
