# Pi0.5 Inference Cache System - Architecture Specification

> Version: 0.3 (Step 1 validated, Step 2 validated, Step 3 ⚠️ landed with high-risk tag)
> Status: Implementation Phase — Steps 0-2 validated, Step 3 ⚠️ unstable (no test coverage, interfaces will change), CP2 suspended
> Scope: PyTorch inference pipeline only (JAX path disabled)
> Last updated: 2026-04-03

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
- **Hit behavior**: Skip Stage 2 + Stage 3, directly output cached action chunk.
- **Savings**: Maximum (skip LLM decoding + all flow matching).
- **Risk**: Highest — skips subtask prediction. If the scene has changed subtly (e.g., an object was removed), the cached subtask may no longer be correct.
- **Applicable scenario**: Highly repetitive operations (e.g., the same action on an assembly line).

### CP2: After LLM Backbone — ⚠️ Suspended

> **Why suspended**: The original design assumed Pi0.5's Stage 2 performs autoregressive subtask text generation, producing command tokens + command embedding that give CP2 its "same command → same action" semantic basis. However, Step 1 code analysis revealed that **the PyTorch implementation of Pi0.5's Stage 2 only fills the prefix KV cache — there is no autoregressive text generation** (JAX path is disabled). The only new information after Stage 2 is `past_key_values` (a HuggingFace DynamicCache, an opaque object), which cannot be directly used as a retrieval key. CP2's semantic premise does not hold. It is suspended until a suitable Stage 2 representation extraction approach is available (e.g., extracting embedding from the last-layer hidden state of the KV cache).

- **Trigger**: Stage 2 complete; ~~low-level command (subtask text tokens) generated~~ KV cache filled.
- **Available information**: All CP1 information + `past_key_values` (opaque KV cache, not directly usable as query key).
- **Hit behavior (two modes)** *(design preserved, not implemented)*:
  - **Full hit**: Skip all of Stage 3, directly output cached action chunk.
  - **Partial hit (warm start)**: Use cached intermediate state `x_t` (t < 1.0) as flow matching starting point, skipping some denoising steps.
- **Savings**: Medium (skip all or part of flow matching).
- **Risk**: Medium — subtask was computed by the current inference; cached action has higher consistency with the current scene.
- **Applicable scenario**: Reuse of the same subtask in similar scenes.
- **Current status**: **Suspended** — no usable retrieval key; requires a new representation extraction approach.

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
│  └──────────────────────┬───────────────────────────┘                   │
│                         │                                                │
│  ┌──────────────────────┴───────────────────────────┐                   │
│  │              GateFunction (pluggable)             │                   │
│  │  Decides: should we even search cache?            │                   │
│  │  (heuristic / lightweight model / always-on)      │                   │
│  └──────────────────────┬───────────────────────────┘                   │
│                         │                                                │
│  ┌──────────────────────┴───────────────────────────┐                   │
│  │              SimilarityJudge (pluggable)          │                   │
│  │  Given search results, decide: hit or miss?       │                   │
│  │  (threshold / learned / composite)                │                   │
│  └──────────────────────────────────────────────────┘                   │
└──────────────────────────┬───────────────────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          │    CacheStorage (facade)         │
          │  ┌───────────────┐ ┌──────────┐ │
          │  │VectorStore-   │ │MetadataDB│ │
          │  │Backend (ABC)  │ │(reserved,│ │
          │  │ ⚠️ unstable   │ │ not impl)│ │
          │  │               │─│          │ │
          │  │┄Qdrant (now) ┄│ └──────────┘ │
          │  │┄FAISS (future)│               │
          │  │┄TorchGPU(fut.)│               │
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

> **Status**: Step 4 design — not yet implemented.
> ⚠️ This design uses Step 3 storage layer types (`QuerySpec`, `SearchResultLite`, `CacheEntry`, etc.) which are unstable. `CacheContext` and `CacheResult` are orchestrator-layer types to be defined in Step 4.

The master controller. Manages the lifecycle of all checkpoints, coordinates the gate/search/judge workflow, and handles async write-back.

> **Note**: With CP2 suspended, the orchestrator currently only handles CP1 and CP3. `config.cp2_enabled` defaults to `False`.

```python
class CacheOrchestrator:
    def __init__(
        self,
        storage: CacheStorage,
        key_builder: QueryKeyBuilder,
        gate: GateFunction,
        judge: SimilarityJudge,
        config: CacheConfig,
        timer: SystemTimer,
    ):
        ...
        self._next_action_scheduled: Optional[torch.Tensor] = None  # [50, 32]
        # Write queue implementation TBD in Step 4

    def should_skip_inference(self) -> Optional[torch.Tensor]:
        """Called BEFORE inference starts. If CP3 from previous cycle
        scheduled a cached action, return it and skip entire inference."""
        if self._next_action_scheduled is not None:
            action = self._next_action_scheduled
            self._next_action_scheduled = None
            return action
        return None

    def check(self, checkpoint: CheckpointID, context: CacheContext) -> CacheResult:
        """Core cache check logic at a given checkpoint.

        CacheContext, CacheResult: Step 4 orchestrator-layer types (not yet defined).
        Storage interaction uses Step 3 types (⚠️ unstable).
        """

        # Step 1: Gate - should we even search?
        with self.timer.measure(f"{checkpoint.name}_gate"):
            if not self.gate.should_search(checkpoint, context):
                return CacheResult.miss()

        # Step 2: Build query keys (multi-named-vector)
        with self.timer.measure(f"{checkpoint.name}_key_build"):
            keys = self.key_builder.build(checkpoint, context)  # dict[str, Tensor]

        # Step 3: Search vector DB — ⚠️ uses Step 3 storage types
        with self.timer.measure(f"{checkpoint.name}_search"):
            spec = QuerySpec(
                query_keys=keys,
                top_k=self.config.top_k,
                checkpoint_id=checkpoint,
            )
            candidates: list[SearchResultLite] = self.storage.search(spec)

        # Step 4: Judge - is the best candidate good enough?
        with self.timer.measure(f"{checkpoint.name}_judge"):
            result = self.judge.evaluate(checkpoint, context, candidates)

        return result

    def write_async(self, entry: CacheEntry):
        """Non-blocking cache write. Runs on background thread.
        Parameter is a CacheEntry (⚠️ Step 3 type, unstable)."""
        ...

    def schedule_next_action(self, action: torch.Tensor):
        """CP3 schedules an action chunk for the next cycle."""
        self._next_action_scheduled = action
```

### 5.2 CacheStorage — ⚠️ Step 3 Implemented (Unstable)

> **Implementation**: `src/openpi/cache/cache_storage.py` | **Design log**: `claude_log/step3_cache.log`
> **Status**: ⚠️ Code landed, no test coverage, interfaces will change.

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

### 5.3 VectorStoreBackend — ⚠️ Step 3 Implemented (Unstable)

> **Implementation**: `src/openpi/cache/backend_base.py` + `src/openpi/cache/backends/qdrant_backend.py`
> **Design log**: `claude_log/step3_cache.log`
> **Status**: ⚠️ Code landed, no test coverage, interfaces will change.

**Vector DB choice is undecided.** Qdrant is the current experiment backend, but we **will** switch to another DB in the future (candidates: FAISS, custom TorchGPU store, or others). To isolate upper-layer logic from this decision, Step 3 introduced a `VectorStoreBackend` ABC as the minimal common-denominator interface.

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

> **Status**: Step 4 design — not yet implemented.
> ⚠️ Return type `dict[str, torch.Tensor]` aligns with `QuerySpec.query_keys` / `CacheEntry.query_keys` (Step 3 storage types, unstable). `CacheContext` is a Step 4 orchestrator-layer type (not yet defined).

```python
class QueryKeyBuilder(Protocol):
    """Converts stage outputs into named query vectors.

    Returns a dict mapping field names (subset of CACHE_QUERY_FIELDS)
    to L2-normalized tensors. The backend stores/queries only the fields
    declared in its vector_dims; extra fields are silently ignored.
    """

    def build(self, checkpoint: CheckpointID, context: CacheContext) -> dict[str, torch.Tensor]:
        """Returns named query vectors {field: [dim] tensor, L2 normalized}."""
        ...


class MeanPoolKeyBuilder(QueryKeyBuilder):
    """Baseline: project each embedding source to a fixed dim, return as named vectors.

    Note: 'command' projection is omitted — Stage 2 produces only an opaque
    KV cache with no extractable command embedding (CP2 suspended, see Section 3).
    """

    def __init__(self, output_dim: int = 1024):
        self.projections = nn.ModuleDict({
            "vision_0": nn.Linear(..., output_dim),
            "prompt_emb": nn.Linear(..., output_dim),
            "robot_state": nn.Linear(..., output_dim),
            # "command": removed — no command embedding available (CP2 suspended)
        })

    def build(self, checkpoint, context):
        keys = {}
        if context.stage1:
            keys["vision_0"] = F.normalize(
                self.projections["vision_0"](context.stage1.vision_emb.mean(dim=1)), dim=-1
            )
            keys["robot_state"] = F.normalize(
                self.projections["robot_state"](context.stage1.state_emb), dim=-1
            )
            keys["prompt_emb"] = F.normalize(
                self.projections["prompt_emb"](context.stage1.prompt_emb), dim=-1
            )
        return keys


class PlaceholderKeyBuilder(QueryKeyBuilder):
    """For early development: use raw state vector as the only query field."""

    def build(self, checkpoint, context):
        return {"robot_state": F.normalize(context.stage1.raw_state.float(), dim=-1)}
```

Designed as a Protocol so it can be swapped for a learned encoder or other approaches later without affecting the rest of the system.

### 5.5 GateFunction (Pluggable)

> **Status**: Step 4 design — not yet implemented.
> `CacheContext` is a Step 4 orchestrator-layer type (not yet defined). Gate does not interact with the storage layer directly, but its input type will be finalized when Step 4 is implemented.

Decides whether to initiate a search at a given checkpoint. Avoids the overhead of searching every time.

```python
class GateFunction(Protocol):
    def should_search(self, checkpoint: CheckpointID, context: CacheContext) -> bool:
        ...

class AlwaysSearchGate(GateFunction):
    """Baseline: always search. For benchmarking overhead."""
    def should_search(self, checkpoint, context):
        return True

class IntervalGate(GateFunction):
    """Only search every N inference cycles."""
    def __init__(self, interval: int = 3):
        self.interval = interval
        self.counter = 0

    def should_search(self, checkpoint, context):
        self.counter += 1
        return self.counter % self.interval == 0

class StateChangeGate(GateFunction):
    """Search only when state change exceeds threshold.
    If robot barely moved since last check, cache result likely same -> skip search."""
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.last_state: Optional[torch.Tensor] = None

    def should_search(self, checkpoint, context):
        current = context.stage1.raw_state
        if self.last_state is None:
            self.last_state = current
            return True
        delta = (current - self.last_state).norm().item()
        if delta > self.threshold:
            self.last_state = current
            return True
        return False
```

### 5.6 SimilarityJudge (Pluggable)

> **Status**: Step 4 design — not yet implemented.
> ⚠️ Uses `SearchResultLite` from Step 3 storage layer (unstable). `CacheContext` and `CacheResult` are Step 4 orchestrator-layer types (not yet defined). `SearchResultLite.score` range depends on backend/mode — thresholds must be calibrated accordingly.

Determines whether search results constitute a valid hit.

```python
class SimilarityJudge(Protocol):
    def evaluate(
        self, checkpoint: CheckpointID, context: CacheContext,
        candidates: list[SearchResultLite],  # ⚠️ Step 3 type
    ) -> CacheResult:
        ...

class ThresholdJudge(SimilarityJudge):
    """Simple threshold-based judge with per-checkpoint thresholds."""

    def __init__(self, storage: CacheStorage, thresholds: dict[CheckpointID, float]):
        self.storage = storage  # needed for fetch_payload on hit
        self.thresholds = thresholds
        # CP1 threshold should be stricter (higher similarity required)
        # because skipping more computation carries more risk.
        # Default: CP1=0.98, CP3=0.90 (CP2 suspended)

    def evaluate(self, checkpoint, context, candidates):
        if not candidates:
            return CacheResult.miss()

        best: SearchResultLite = candidates[0]
        threshold = self.thresholds[checkpoint]

        if best.score >= threshold:
            return self._make_hit(checkpoint, best)
        return CacheResult.miss()

    def _make_hit(self, checkpoint, candidate: SearchResultLite):
        # Two-phase: fetch full payload only for hit candidates
        payload: CachePayload = self.storage.fetch_payload(candidate.id)

        # CP2 warm start branch — suspended (no command embedding available).
        # Design preserved for future use when CP2 is re-enabled.
        # if checkpoint == CheckpointID.CP2 and payload.intermediates:
        #     if candidate.score >= self.thresholds[CheckpointID.CP2_FULL]:
        #         return CacheResult.full_hit(payload.action_chunk)
        #     else:
        #         t = max(payload.intermediates.keys())
        #         return CacheResult.warm_start(
        #             cached_noisy_action=payload.intermediates[t],
        #             cached_timestep=t,
        #             num_steps=payload.denoising_num_steps,
        #         )
        return CacheResult.full_hit(payload.action_chunk)
```

### 5.7 Cache Data Model — ⚠️ Step 3 Implemented (Unstable)

> **Implementation**: `src/openpi/cache/storage_types.py` + `src/openpi/cache/types.py`
> **Design log**: `claude_log/step3_cache.log`
> **Status**: ⚠️ Code landed, no test coverage, interfaces will change.

Step 3 replaced the original flat `CacheEntry` with a richer data model consisting of multiple dataclasses. Key differences from the original design: `CachePayload` is a separate nested dataclass (not flat fields on `CacheEntry`), `QueryFilter` was added for structured filtering, and search results are split into `SearchResultLite` (lightweight, for threshold filtering) and `SearchResult` (full, with payload).

```python
# src/openpi/cache/types.py
class CheckpointID(Enum):
    CP1 = auto()
    CP2 = auto()
    CP3 = auto()

# src/openpi/cache/storage_types.py

@dataclass
class CachePayload:
    """Payload of a cache entry. Strongly typed, shared across CP1/CP2/CP3.

    Tensor contract: all tensors must be CPU contiguous float32.
    Caller converts before construction; search() returns CPU float32 too.

    CP1: action_chunk required, rest None.
    CP2 warm start: action_chunk + intermediates + denoising_num_steps required.
    CP3: action_chunk + next_action_chunk required.
    """
    action_chunk: torch.Tensor                          # [50, 32]
    intermediates: Optional[dict[float, torch.Tensor]]  # {t: x_t}
    denoising_num_steps: Optional[int]                  # for warm start
    next_action_chunk: Optional[torch.Tensor]           # [50, 32], CP3 required
    task_key: str = ""                                  # normalized task identifier

    def validate_for_checkpoint(self, checkpoint_id: CheckpointID) -> None: ...


@dataclass
class CacheEntry:
    """Complete unit written to storage.

    id semantics: stable_hash(checkpoint_id.name + ":" + sorted_concat_of_query_key_bytes)
    Same context + same CP keeps only one record (semantic dedup key).
    checkpoint_id is part of the hash so CP1/CP2/CP3 entries for the same
    observation do not overwrite each other.
    """
    id: str
    checkpoint_id: CheckpointID
    query_keys: dict[str, torch.Tensor]  # {field: [dim] CPU float32, L2 normalized}
    payload: CachePayload
    timestamp: float = field(default_factory=time.time)

    def validate(self) -> None: ...


@dataclass
class QueryFilter:
    """Per-query constraints. Backends that don't support a field must
    declare so via supported_filters(); CacheStorage will fail-fast."""
    task_key: Optional[str] = None
    step_range: Optional[tuple[int, int]] = None


@dataclass
class QuerySpec:
    query_keys: dict[str, torch.Tensor]  # {field: [dim] CPU float32}
    top_k: int = 10
    checkpoint_id: Optional[CheckpointID] = None
    filters: Optional[QueryFilter] = None


@dataclass
class SearchResultLite:
    """Lightweight search result (no payload). Returned by search().
    score: higher = more similar. Range depends on backend/mode:
      single-field cosine: [-1, 1]; multi-field RRF fusion: small positive numbers.
    Thresholds must be calibrated per backend/mode."""
    id: str
    score: float
    checkpoint_id: CheckpointID


@dataclass
class SearchResult:
    """Full search result with payload. Populated by fetch_payload()."""
    id: str
    score: float
    payload: CachePayload
    checkpoint_id: CheckpointID
```

**Intermediate state selection for warm start** *(design preserved, not yet populated)*: Rather than caching all 10 intermediate states, only 2-3 key timesteps are cached (e.g., t=0.7, 0.5, 0.3). The `denoising_num_steps` field tells `run_stage3_from()` how many steps to resume from. This mechanism is tied to CP2 warm start, which is currently suspended.

---

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

> Implementation: `src/openpi/cache/timing.py` | Design log: `claude_log/step2.log`

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
| `cp1_*`, `cp3_*` | tbd | Planned (Step 4+) | Cache sub-step probes |
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

## 10. Configuration System

> **Note**: `CacheConfig` is not yet implemented as a standalone file. The configuration items below reflect the full design intent. Items marked **Deferred** depend on the GPU/CPU hybrid store (Section 5.3.1 / Section 7) and are not applicable with the current Qdrant backend. Items marked **⚠️** are related to Step 3 storage layer and will change as interfaces evolve.

```python
@dataclass
class CacheConfig:
    """Top-level cache system configuration."""

    # ── General (stable) ─────────────────────────────────────────
    enabled: bool = True

    # Per-checkpoint enable/disable
    cp1_enabled: bool = True
    cp2_enabled: bool = False              # Suspended — no command embedding available (see Section 3)
    cp3_enabled: bool = True

    # Retrieval
    top_k: int = 5                         # candidates per search

    # Similarity thresholds (higher = stricter)
    # Note: threshold scale depends on backend/mode — see SearchResultLite.score docs
    cp1_threshold: float = 0.98            # CP1 strictest: skipping most
    cp2_full_threshold: float = 0.96       # CP2 full hit (suspended)
    cp2_warm_threshold: float = 0.90       # CP2 warm start (suspended)
    cp3_threshold: float = 0.92            # CP3 predictive

    # Flow matching warm start
    intermediate_timesteps: list[float] = field(
        default_factory=lambda: [0.7, 0.5, 0.3]
    )  # which timesteps to cache x_t for

    # Write policy
    write_similarity_threshold: float = 0.99  # don't write if too similar to existing
    write_async: bool = True

    # Gate
    gate_type: str = "always"              # "always", "interval", "state_change"
    gate_interval: int = 1
    gate_state_threshold: float = 0.01

    # Timing (✅ Step 2 implemented)
    timing_enabled: bool = True
    timing_buffer_size: int = 10000

    # ── Storage — ⚠️ unstable, will change with backend evolution ──
    vector_db: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    metadata_db: Optional[MetadataStoreConfig] = None

    # ── Hardware — Deferred (requires GPU/CPU hybrid store, Section 5.3.1 / Section 7) ──
    gpu_capacity: int = 10000              # Deferred
    cpu_capacity: int = 100000             # Deferred
    pinned_memory_mb: int = 32             # Deferred
```

---

## 11. Modification Boundary for Existing Code

### Actual Modifications (Step 1 — Additions Only, Zero Changes to Existing Code)

> **Key finding**: `pi0_pytorch.py` already had three private stage methods (`_stage1_token_prep`, `_stage2_llm_backbone`, `_stage3_action_expert`) with `sample_actions()` calling them internally. Step 1 did **not** modify `sample_actions()` — it added public typed wrappers on top of the existing private methods.

| File | Change | Type |
|------|--------|------|
| `models_pytorch/pi0_pytorch.py` | Add 3 dataclasses (`Stage1Output`, `Stage2Output`, `Stage3Output`) + 5 methods (`run_stage1/2/3`, `run_stage3_from`, `_stage3_with_intermediates`) | **Add only** (+256 lines, 0 lines modified) |
| `src/openpi/cache/__init__.py` | New module marker | **New file** |
| `src/openpi/cache/interceptor.py` | `InferenceInterceptor(BasePolicy)` — routes inference through staged API | **New file** (+137 lines) |
| `scripts/serve_policy.py` | Add `--cache` CLI flag and `InferenceInterceptor` wrapping logic | **Add only** (+13 lines) |

### Actual Modifications (Step 3 — ⚠️ Unstable, Will Change)

> **Status**: ⚠️ Code landed, no test coverage, interfaces will change frequently.
> Step 3 has two sub-parts: data collection (stable observer pattern) and cache storage layer (unstable).

**Step 3a: Data Collection** (`claude_log/step3_data_collection.log`)

| File | Change | Type |
|------|--------|------|
| `src/openpi/collect/__init__.py` | Empty module marker | **New file** |
| `src/openpi/collect/data_collector.py` | `InferenceEmbeddings` dataclass + `EpisodeDataCollector` (buffer + HDF5 write) | **New file** |
| `src/openpi/collect/collection_policy.py` | `CollectionPolicy` wrapper (outermost, 4 forward hooks) | **New file** |
| `scripts/serve_policy.py` | Add `--collect` / `--collect_dir` args, outermost `CollectionPolicy` wrapping | **Add only** |
| `serving/websocket_policy_server.py` | In-band control message recognition (`__ctrl__: episode_start/end`) | **Modify** |
| `openpi-client/websocket_client_policy.py` | Add `episode_start()` / `episode_end()` methods | **Add only** |
| `examples/libero/main.py` | Insert `client.episode_start/end()` calls around episode loop | **Modify** |

**Step 3b: Cache Storage Layer** (`claude_log/step3_cache.log`) — ⚠️

| File | Change | Type |
|------|--------|------|
| `src/openpi/cache/types.py` | `CheckpointID` enum | **New file** |
| `src/openpi/cache/storage_types.py` | `CachePayload`, `CacheEntry`, `QueryFilter`, `QuerySpec`, `SearchResultLite`, `SearchResult`, `BatchInsertResult` | **New file** |
| `src/openpi/cache/backend_base.py` | `VectorStoreBackend` ABC | **New file** |
| `src/openpi/cache/cache_storage.py` | `CacheStorage` facade (thread-safe, validation, two-phase search) | **New file** |
| `src/openpi/cache/backends/__init__.py` | Empty module marker | **New file** |
| `src/openpi/cache/backends/qdrant_backend.py` | `QdrantVectorStore` (Qdrant implementation of `VectorStoreBackend`) | **New file** |
| `src/openpi/cache/__init__.py` | Updated exports for new symbols | **Modify** |

### Unmodified Parts

| File | Notes |
|------|-------|
| `models_pytorch/pi0_pytorch.py` — existing code | `sample_actions()`, `_stage1/2/3_*`, `denoise_step()` — zero modifications |
| `policies/policy.py` | Policy class unchanged; InferenceInterceptor wraps it externally |
| `models/pi0.py` | JAX path disabled, untouched |
| `serving/websocket_policy_server.py` | ✅ Step 2: removed old `stage_timing_records` aggregation, added `TaskLifecycle` callbacks; ✅ Step 3a: added control message handling |
| `training/` | Training code completely untouched |
| `transforms.py` | Data transforms unchanged |

---

## 12. File Structure

### Actual (as of Step 3)

```
src/openpi/cache/                           # Cache module
├── __init__.py                             # ✅ Step 1, updated Step 3
├── interceptor.py                          # ✅ Step 1 — InferenceInterceptor (wraps Policy)
├── timing.py                               # ✅ Step 2 — SystemTimer, TimingRecord, TimingStats
├── types.py                                # ⚠️ Step 3 — CheckpointID enum
├── storage_types.py                        # ⚠️ Step 3 — CachePayload, CacheEntry, QuerySpec, etc.
├── backend_base.py                         # ⚠️ Step 3 — VectorStoreBackend ABC
├── cache_storage.py                        # ⚠️ Step 3 — CacheStorage facade
├── backends/
│   ├── __init__.py                         # ⚠️ Step 3
│   └── qdrant_backend.py                   # ⚠️ Step 3 — QdrantVectorStore
└── README.md                               # ⚠️ Step 3 — module-level docs

src/openpi/collect/                         # Data collection module (Step 3a)
├── __init__.py
├── data_collector.py                       # InferenceEmbeddings + EpisodeDataCollector
└── collection_policy.py                    # CollectionPolicy wrapper (outermost)
```

### Planned (future steps, not yet created)

```
src/openpi/cache/
├── config.py                               # CacheConfig, VectorStoreConfig, etc.
├── orchestrator.py                         # CacheOrchestrator master controller
├── components/
│   ├── key_builder.py                      # QueryKeyBuilder protocol + implementations
│   ├── gate.py                             # GateFunction protocol + implementations
│   └── judge.py                            # SimilarityJudge protocol + implementations
├── hardware/                               # Deferred — GPU/CPU hybrid store
│   ├── cuda_manager.py                     # CacheHardwareManager, stream management
│   └── memory_pool.py                      # PinnedMemoryPool
├── maintenance/                            # Deferred — eviction, migration
│   ├── eviction.py
│   ├── promotion.py
│   └── writer.py
└── backends/
    ├── faiss_backend.py                    # Planned — FAISS local backend
    └── torch_gpu_backend.py                # Planned — TorchGPU in-process backend
```

---

## 13. Development Roadmap

> Core principle: **Build the skeleton and get it running first, then experiment and optimize on the working system.**
> Do not research similarity metrics before having an end-to-end pipeline; do not optimize performance before having timing data.

---

### Step 0: Understand the Existing Inference Pipeline — ✅ Merged into Step 1

> Step 0 was not executed as a standalone step. The code analysis and baseline understanding were performed as part of Step 1's planning phase. See `claude_log/step1.log` Section 1 for the full analysis results, including tensor shapes, inter-stage data flow, and Pi0 vs Pi0.5 architectural comparison.

---

### Step 1: Staged Public API + Interceptor Skeleton — ✅ Completed

> **Status**: Validated | **Commit**: `a6c9f43` on branch `Ziyang` | **Date**: 2026-03-29
> **Log**: `claude_log/step1.log`

**Goal**: Add public typed wrappers on top of the existing private stage methods, and create an `InferenceInterceptor` skeleton that routes inference through the staged API.

**Key finding from code analysis**: `pi0_pytorch.py` already had three private methods (`_stage1_token_prep`, `_stage2_llm_backbone`, `_stage3_action_expert`) with `sample_actions()` calling them internally. No need to "split" — only need to add public typed wrappers.

**Critical discovery — CP2 basis invalidated**: The original architecture assumed Pi0.5's Stage 2 performs autoregressive subtask text generation (producing command tokens + command embedding). However, the **PyTorch path does NOT have autoregressive generation** — Stage 2 only fills the prefix KV cache. The only output is an opaque `past_key_values` (DynamicCache), which cannot serve as a retrieval key. **CP2 is therefore suspended** until a suitable representation extraction approach is found.

**What was implemented**:

1.1. Added 3 dataclasses to `pi0_pytorch.py` (before `PI0Pytorch` class):

```python
@dataclass
class Stage1Output:
    state: torch.Tensor               # [B, action_dim] — raw state for cache key
    prefix_embs: torch.Tensor         # [B, prefix_len, emb_dim] (bfloat16)
    prefix_pad_masks: torch.Tensor    # [B, prefix_len] (bool)
    prefix_att_2d_masks_4d: torch.Tensor  # [B, 1, prefix_len, prefix_len]
    prefix_position_ids: torch.Tensor # [B, prefix_len] (int64)

@dataclass
class Stage2Output:
    stage1: Stage1Output
    past_key_values: Any              # HuggingFace DynamicCache — pass as-is, do NOT clone

@dataclass
class Stage3Output:
    action_chunk: torch.Tensor        # [B, action_horizon, action_dim] (float32)
    intermediates: Optional[dict[float, torch.Tensor]] = None  # For warm start
```

> **Note**: `Stage2Output` has no `command_tokens` or `command_embedding` — these do not exist in the PyTorch path. This is the root cause of CP2 suspension.

1.2. Added 5 methods to `PI0Pytorch` class (all additions, zero modifications to existing code):
  - `run_stage1(observation) -> Stage1Output` — wraps `_stage1_token_prep`
  - `run_stage2(stage1) -> Stage2Output` — wraps `_stage2_llm_backbone`
  - `run_stage3(stage2, *, noise, num_steps, return_intermediates, save_timesteps) -> Stage3Output`
  - `run_stage3_from(stage2, start_x, start_t, *, num_steps) -> Stage3Output` — warm start entry point
  - `_stage3_with_intermediates(...)` — internal helper for intermediate state capture

1.3. Created `src/openpi/cache/interceptor.py` (+137 lines):
  - `InferenceInterceptor(BasePolicy)` — borrows `_model`, `_input_transform`, `_output_transform` from wrapped Policy
  - `infer()` calls `run_stage1 → run_stage2 → run_stage3` with per-stage timing
  - `TODO(Step 4)` markers for CP1/CP3 cache check insertion points

1.4. Modified `scripts/serve_policy.py` (+13 lines):
  - Added `--cache` CLI flag
  - `if args.cache: policy = InferenceInterceptor(policy)`

**Verification**:

| Check | Result |
|-------|--------|
| AST syntax check (all new files) | ✅ Pass |
| Server starts with `--cache` | ✅ `INFO: Cache mode enabled` + `listening on 0.0.0.0:8001` |
| Existing timing system (`stage_timing` field) | ✅ Interceptor outputs same fields |
| External interface compatibility | ✅ WebsocketPolicyServer and clients require zero changes |

**Deliverable**: Modified `pi0_pytorch.py` with public staged API + `interceptor.py` + `--cache` integration.

---

### Step 2: Timing System — ✅ Completed

**Goal**: Implement `SystemTimer` to provide infrastructure for all subsequent performance quantification.

**Actual implementation** (see `claude_log/step2.log` for full details):

2.1. `src/openpi/cache/timing.py`:
  - Probe-based `SystemTimer` with `register_probe(name, backend="cuda"/"cpu", stream=None)`
  - `TimingBackend` protocol with `CudaEventBackend` (auto-degrades to CPU when no GPU) and `PerfCounterBackend`
  - `enabled=False` for zero-overhead disable; unregistered probes handled in lenient mode with warning
  - Ring buffer with monotonic counter for task boundary tracking
  - `summary()` with per-probe mean/p50/p95/p99 + correct total (sum) row
  - `export_csv()` for raw record export; auto-CSV on task end via `output_csv_dir`

2.2. `InferenceInterceptor` registers 4 probes: `stage1_vision`/`stage2_llm`/`stage3_flow` (cuda) + `total_inference` (cpu). Replaces Step 0's manual timing.

2.3. `TaskLifecycle` protocol: `on_task_begin()`/`on_task_end()` for task-level aggregation.

2.4. `websocket_policy_server.py`: removed old `stage_timing_records` aggregation (~15 lines), added `TaskLifecycle` callbacks on connection open/close.

2.5. Validation: 12 tests passing via `scripts/verify_step2.py`.

**Deliverable**: `timing.py` + timer integrated into interceptor + server-side lifecycle hooks.

---

### Step 3: Data Collection + Cache Storage Layer — ⚠️ Landed (High Risk)

> **Status**: ⚠️ Code landed, no test coverage, interfaces will change frequently.
> **Logs**: `claude_log/step3_data_collection.log` (3a), `claude_log/step3_cache.log` (3b)
> **Date**: 2026-04-02

**Goal**: Two sub-parts — (3a) build a pure-observer data collection system for gathering inference embeddings into HDF5, and (3b) implement the cache storage layer abstraction so that upper-layer logic is decoupled from any specific vector DB.

**Why the storage abstraction?** We do not yet know which vector DB we will use long-term. Qdrant is the experiment backend now, but it **will** be replaced (candidates: FAISS, custom TorchGPU store, etc.). The `VectorStoreBackend` ABC ensures that swapping backends requires zero changes to orchestrator or business code.

**What was implemented**:

3a. **Data Collection** (stable observer pattern):
  - `src/openpi/collect/collection_policy.py`: `CollectionPolicy` wrapper (outermost in wrapper chain), registers 4 temporary forward hooks per inference, writes `InferenceEmbeddings` to `EpisodeDataCollector`
  - `src/openpi/collect/data_collector.py`: `InferenceEmbeddings` dataclass + `EpisodeDataCollector` (buffer + HDF5 atomic write)
  - 4 forward hooks: `_vision_hook` (multi_modal_projector), `_lang_hook` (embed_tokens), `_action_in_hook` (action_in_proj), `_action_out_hook` (action_out_proj)
  - HDF5 schema: per-step groups with vision_0/1/2, prompt_emb, robot_state, noise_action_1..N-1, clean_action
  - In-band control messages (`__ctrl__: episode_start/end`) for client-server episode lifecycle
  - `scripts/serve_policy.py`: `--collect` / `--collect_dir` flags

3b. **Cache Storage Layer** (⚠️ unstable):
  - `src/openpi/cache/types.py`: `CheckpointID` enum (CP1/CP2/CP3)
  - `src/openpi/cache/storage_types.py`: `CachePayload`, `CacheEntry`, `QueryFilter`, `QuerySpec`, `SearchResultLite`, `SearchResult`, `BatchInsertResult` — all strongly typed, with CP-specific validation
  - `src/openpi/cache/backend_base.py`: `VectorStoreBackend` ABC — minimal interface (insert/search/fetch_payload/delete/count + supported_filters)
  - `src/openpi/cache/cache_storage.py`: `CacheStorage` facade — thread-safe (RLock), dimension validation, filter fail-fast, two-phase search
  - `src/openpi/cache/backends/qdrant_backend.py`: `QdrantVectorStore` — Qdrant implementation with tensor serialization (torch.save → base64), supported filters: checkpoint_id/task_key/step_range

**Key design decisions** (see `claude_log/step3_cache.log` for full rationale):
- `CachePayload` is a separate nested dataclass (not flat fields), with `validate_for_checkpoint()` for CP-specific invariants
- Two-phase search: `search()` returns `SearchResultLite` (no payload), `fetch_payload()` on demand — avoids transferring unused tensor data
- `QueryFilter` with `supported_filters()` fail-fast — unsupported filters raise `UnsupportedFilterError`, never silently ignored
- ABC is intentionally small: named vectors, multivector, gRPC options are backend-internal config, never exposed above the ABC boundary
- Cache system vectors (fused `[1024]` query key) vs HDF5 experiment vectors (raw embeddings) use different Qdrant collections — must not be mixed

**⚠️ Known risks**:
- No test coverage — all interfaces are subject to change
- `torch` lazy import changes not regression-tested in uv environment
- Qdrant backend will be replaced; ABC interface may evolve with new backend requirements
- `CacheConfig` not yet implemented as standalone file

**Deliverable**: Data collection system (3a) + storage layer abstraction with Qdrant backend (3b).

---

### Step 4: Orchestrator Skeleton (CP1 + CP3)

**Goal**: Connect cache check logic to the inference pipeline, achieving an end-to-end cache workflow. This step uses the simplest component implementations (PlaceholderKeyBuilder + AlwaysSearchGate + ThresholdJudge), with **CP1 and CP3 enabled** (CP2 suspended — see Section 3).

> **Note**: `InferenceInterceptor` was already created in Step 1. This step adds the `CacheOrchestrator` and plugs CP1/CP3 checks into the existing interceptor's `TODO(Step 4)` slots.

**Why CP1 + CP3 (not CP2)**:
- CP2's original rationale ("same command → same action") assumed command embedding availability, which does not exist in the PyTorch path. CP2 is suspended.
- CP1 after vision: uses `raw_state` and/or vision embeddings as key. The semantics are "same scene + same state → same action". Strictest threshold (0.98) mitigates the risk of skipping subtask prediction.
- CP3 after action expert: predictive cache for next-cycle skip. This is the highest-savings path when consecutive actions have temporal locality.

**Tasks**:

4.1. Implement `src/openpi/cache/components/key_builder.py`:
  - `QueryKeyBuilder` Protocol
  - `PlaceholderKeyBuilder`: Use `stage1_output.state` (raw state vector `[B, 32]`) with L2 normalize as the simplest key. For CP3, concatenate state + action chunk.

4.2. Implement `src/openpi/cache/components/gate.py`:
  - `GateFunction` Protocol
  - `AlwaysSearchGate`: Always returns True

4.3. Implement `src/openpi/cache/components/judge.py`:
  - `SimilarityJudge` Protocol
  - `ThresholdJudge`: cosine similarity > threshold -> hit

4.4. Implement `src/openpi/cache/orchestrator.py`:
  - `CacheOrchestrator`: Combines key_builder + gate + judge + storage
  - `check()` method: gate -> build key -> search -> judge
  - `write_async()` method: Use synchronous writes initially (async deferred to Step 8)

4.5. Integrate into `src/openpi/cache/interceptor.py`:
  - Plug CP1 check into the `TODO(Step 4)` slot after Stage 1
  - Plug CP3 check into the `TODO(Step 4)` slot after Stage 3
  - CP2 slot remains commented out

4.6. **End-to-end tests**:
  - Load model, run 10 identical inputs -> 1st should miss, 2nd-10th should hit CP1 (identical input)
  - Run 10 different inputs -> all miss
  - Verify that the action returned on CP1 hit has L2 distance = 0 from normal inference result

**Deliverable**: A working end-to-end cache system (CP1 + CP3) that passes the above tests.

---

### Step 5: Core Experiment — Cache Feasibility Validation

**Goal**: Answer the critical question — "For similar but not identical inputs, what is the quality of cache-hit actions?" This determines whether the entire cache system makes sense.

> **This is the first critical experiment milestone for the entire project.** If experiments show that similar inputs produce widely different actions, the cache approach needs fundamental reevaluation. Do not invest further development effort before this experiment.

**Experiment design**:

5.1. **Data preparation**: Collect inference episodes (100-500 steps), recording for each step:
  - Input observation (images, state, prompt)
  - Stage 1 output (vision embedding, state)
  - Final action chunk
  - Save all data to disk (HDF5 or pickle)
  - *(Note: command embedding is unavailable — CP2 suspended)*

5.2. **Experiment A: Action continuity in state space**
  - For recorded episodes, compute pairwise state cosine similarity across all steps
  - Compute corresponding action L2 distance
  - Plot scatter: x=state_similarity, y=action_distance
  - **Expected result**: Action distance is low when state similarity is high (positive correlation)
  - **If this trend is not observed**: The cache approach has a fundamental problem

5.3. **Experiment B: Cache hit action quality**
  - Using the Step 4 system, gradually lower the CP1 threshold (from 0.99 to 0.80)
  - Record at each threshold: hit rate, action L2 error (vs normal inference), latency savings
  - Plot three curves: threshold vs hit_rate, threshold vs action_error, threshold vs latency_saving
  - **Find the sweet spot**: Maximize hit rate subject to action error being acceptable (< some value)

5.4. **Experiment C: Discriminative power of different query keys**
  - Compare retrieval quality across several key construction approaches:
    - (a) raw state vector only
    - (b) vision embedding mean pool
    - (c) state + vision embedding concatenation
    - (d) state + action chunk concatenation (for CP3)
  - *(command embedding removed — unavailable in PyTorch path)*
  - Metric: precision@k (proportion of top-k retrieved entries whose action is truly close to the current inference action)
  - **This experiment guides subsequent QueryKeyBuilder design**

**Deliverable**: Experiment report with the above plots and conclusions. Decision on whether to continue, plus initial threshold range and key builder direction.

---

### Step 6: CP1/CP3 Refinement + CP3 Deferred Writer

**Prerequisite**: Step 5 experiment results are positive (cache feasibility validated).

> **Note**: Basic CP1 and CP3 integration was done in Step 4. This step focuses on CP3's deferred write mechanism and refinement based on Step 5 experiment results.

**Tasks**:

6.1. **CP1 refinement**:
  - Tune key builder based on Step 5 Experiment C results (which information sources work best)
  - CP1 uses the stricter threshold (default 0.98)
  - Test: same scene + same prompt should hit; changing objects or prompt should miss

6.2. **CP3 deferred writer**:
  - `schedule_next_action()` mechanism: maintain a `_next_action_scheduled` slot in the orchestrator
  - `should_skip_inference()`: Check at the start of each cycle for a pre-scheduled action
  - CP3 key needs to include action chunk information (since it predicts "the next step")
  - Maintain **consecutive action sequence mapping** — add `next_entry_id` field to entries, pointing to the temporally next entry

6.3. **CP3 special consideration**: CP3 cache entries need to record the "current action -> next action" mapping. This means:
  - When writing to cache, the `next_action_chunk` field cannot be filled until the **next** cycle's action is produced
  - Implement a `DeferredWriter`: Write the entry in cycle N (without next), backfill `next_action_chunk` in cycle N+1

6.4. **Experiments**:
  - Measure CP1/CP3 hit rates across episodes (CP2 suspended)
  - Quantify latency savings at each checkpoint on hit
  - CP3 predictive accuracy: L2 distance between pre-scheduled action and actually inferred action

**Deliverable**: Refined CP1 + CP3 with deferred writer + hit rate and latency reports.

---

### Step 7: Flow Matching Warm Start

**Prerequisite**: Step 6 complete, CP1/CP3 validated. *(Note: warm start was originally designed for CP2. With CP2 suspended, warm start may be repurposed for CP1 partial skip or deferred to when CP2 is re-enabled.)*

**Tasks**:

7.1. Modify `run_stage3()`:
  - When `return_intermediates=True`, save `x_t` at selected timesteps during the flow matching loop
  - Default: save at t=0.7, 0.5, 0.3 (configurable)
  - Saved tensor shape = `[B, action_horizon, action_dim]`, same as noise

7.2. Add `run_stage3_from(stage2_output, start_x, start_t)`:
  - Execute remaining Euler steps starting from `start_x` and `start_t`
  - For example, `start_t=0.3` runs only 3 steps (0.3 -> 0.2 -> 0.1 -> 0.0) instead of 10

7.3. Add warm start judgment logic to the CP2 judge:
  - similarity > `cp2_full_threshold` -> FULL hit
  - `cp2_warm_threshold` < similarity < `cp2_full_threshold` -> WARM_START hit
  - similarity < `cp2_warm_threshold` -> miss

7.4. **Critical experiment: Warm start accuracy vs speed tradeoff**

  This is the second critical experiment milestone.

  - For the same set of inputs, run:
    - (a) Full 10-step flow matching (baseline)
    - (b) Warm start from own cached x_0.7 (3 steps skipped)
    - (c) Warm start from own cached x_0.5 (5 steps skipped)
    - (d) Warm start from own cached x_0.3 (7 steps skipped)
  - Measure action L2 error vs baseline
  - "Own cached" means using intermediate states from **identical input**, isolating warm start error alone (no state similarity concerns)

  - Then warm start with cached x_t from **similar but different** inputs:
    - Take cached x_0.5 from a similar state in the episode
    - Continue denoising with the current observation's velocity field
    - Measure action L2 error
  - **Expected**: Error is within acceptable range and smaller than "directly using cached action without flow matching"

  - Plot trade-off: x=steps_skipped, y=action_error, multiple lines for different state similarity levels

**Deliverable**: Warm start implementation + trade-off experiment data. Determine default warm start timestep and threshold.

---

### Step 8: System Efficiency Optimization — Async and Hardware

**Prerequisite**: Step 7 complete, functional correctness validated. Performance optimization is deferred to this step because timing data from real operation is needed to guide where to invest effort.

**Optimization decision process**: First generate a complete latency breakdown report using Step 2's timer, identify bottlenecks, then optimize in a targeted manner. Do not optimize based on guesses.

**Potential optimization directions** (ordered by expected impact):

8.1. **Async cache writes** (almost certainly needed):
  - Step 4's writes are synchronous, blocking inference
  - Implement `AsyncWriteWorker`: Background thread consuming write requests from a queue
  - Use `threading.Thread` + `queue.Queue` (no need for multiprocessing since writes are I/O bound)
  - Verify: Write latency disappears from the inference critical path

8.2. **GPU VectorStore** (if CPU search is a bottleneck):
  - Check `cp*_search` latency in the timer report
  - If CPU FAISS search latency > 1ms and cache entries > 10k, consider GPU partition
  - Implement `torch.mm` cosine similarity search on a dedicated CUDA stream
  - Use a separate stream to avoid blocking the main inference stream
  - Verify: Search latency drops, main stream inference latency unaffected

8.3. **CUDA Stream isolation** (if cache operations block inference):
  - Implement `CacheHardwareManager`
  - All GPU cache operations (search, key building projections) run on `cache_stream`
  - Pinned memory pool for CPU<->GPU data transfers

8.4. **Gate optimization** (if cache checks themselves become a bottleneck):
  - Implement `StateChangeGate`: Skip search when state change is below threshold
  - Implement `IntervalGate`: Search only every N inference cycles
  - This can dramatically reduce search frequency, especially useful when hit rate is low

8.5. **Eviction strategy** (if cache growth slows down search):
  - Implement `CompositeEviction` (LRU + LFU + quality)
  - Set capacity caps, periodic eviction
  - Eviction runs on a background thread

**Deliverable**: Optimized system + before/after latency comparison report.

---

### Step 9: Query Key Research (Experiment-Intensive)

**Prerequisite**: Step 8 complete, system performance at an acceptable level. A stable running cache system now serves as the experiment platform.

**Why placed here and not earlier**: Query key research needs to be done on a real running system, requiring real hit/miss data and real latency numbers. Researching keys before the system is operational is building castles in the air.

**Experiment directions**:

9.1. **Key information source ablation experiment**:
  - Using collected episode data, compare retrieval quality across different information combinations as keys:

  | Key Combination | Dimension | Precision@5 | Recall@5 | Compute Cost |
  |----------------|-----------|-------------|----------|-------------|
  | raw_state | 32 | ? | ? | Minimal |
  | state + prompt_hash | 32+64 | ? | ? | Low |
  | vision_emb (mean pool) | 2048 | ? | ? | Medium |
  | state + vision_emb | 32+2048 | ? | ? | Medium |
  | state + action_chunk (CP3) | 32+1600 | ? | ? | Medium |
  | learned projection | 128/256/512 | ? | ? | Requires training |

  > *(command_emb rows removed — unavailable in PyTorch path, CP2 suspended)*

  - "Precision@5" defined as: proportion of top-5 retrieved entries whose action L2 distance from the current inference action is < epsilon

9.2. **Learned Key Builder** (if simple approaches are insufficient):
  - Train a small projection head (2-3 layer MLP), input is stage output concatenation, output is a low-dimensional key
  - Training objective: contrastive loss — similar states produce close keys, different states produce distant keys
  - Training data from offline episode collection
  - Constraint: Projection head inference latency < 0.5ms (otherwise not worth using cache)

9.3. **Optimal key may differ per checkpoint**:
  - CP1's key has vision + state information; experiment with different weightings
  - CP3's key needs action information to predict subsequent actions
  - *(CP2 suspended — no key design needed until re-enabled)*
  - Tune each active checkpoint independently

**Deliverable**: Optimal key builder approach per checkpoint + supporting experiment data.

---

### Step 10: Offline Pre-fill Pipeline

**Prerequisite**: Step 9 complete, key builder approach finalized.

**Tasks**:

10.1. Implement `scripts/prefill_cache.py`:
  - Load episodes from training data or offline rollouts
  - For each timestep, run model inference (or read directly from saved episode data)
  - Build keys, write to VectorStore
  - Support incremental pre-fill (check existing entries, avoid duplicates)

10.2. Serialization/deserialization:
  - VectorStore supports `save(path)` and `load(path)`
  - Save pre-filled cache to disk; load directly at inference time to avoid cold starts

10.3. **Pre-fill quality validation**:
  - Run inference episodes with pre-filled cache
  - Compare action trajectories with cache vs without cache
  - Quantify pre-filled cache hit rate (expected to be much higher than online accumulation from empty cache)

**Deliverable**: Offline pre-fill script + serialization support.

---

### Step 11: Integration Testing and Robustness

**Tasks**:

11.1. **Long-running stability test**:
  - Run 1000+ step episodes, monitoring:
    - Memory leaks (whether cache growth is controlled)
    - Latency stability (whether there's a gradual slowdown trend)
    - Whether hit rate changes reasonably

11.2. **Edge case testing**:
  - Behavior when cache is empty (all miss, normal inference)
  - Eviction behavior when cache is full
  - No crashes on abnormal input (all-black images, empty prompt)
  - Graceful fallback to CPU-only on GPU OOM

11.3. **A/B comparison framework**:
  - Implement `--cache_enabled` flag for one-click cache on/off
  - Compare under identical episodes with cache on/off:
    - End-to-end latency (mean, p95)
    - Action quality (L2 distance from no-cache baseline)
    - GPU utilization

**Deliverable**: Stability test report + A/B comparison data.

---

### Step 12: Advanced Features (On Demand)

The following items are mutually independent; implement based on priority:

12.1. **Metadata DB (MongoDB/SQLite)**:
  - Introduce when rich information beyond vector DB is needed (task name, episode id, success/failure label, etc.)
  - Used for cache analysis and offline quality evaluation
  - Not on the critical path; does not affect inference latency

12.2. **GPU/CPU dynamic migration**:
  - Promotion/Demotion strategies
  - Automatic data tiering based on access frequency

12.3. **Learned Gate Function**:
  - Train a binary classifier to predict "is the current state likely to cache hit?"
  - Input: state delta (vs last), task prompt hash, cache statistics
  - Reduces unnecessary search overhead

12.4. **Distributed Cache**:
  - Multi-robot shared vector DB
  - Requires consideration of network latency and consistency

12.5. **Cache-aware Training**:
  - Introduce cache-hit simulation during training so the model adapts to occasional step-skipping
  - Long-term direction requiring extensive experiments

---

### Development Dependency Graph

```
Step 0: Understand inference pipeline ─── ✅ merged into Step 1
  │
  ▼
Step 1: Staged Public API + Interceptor ─── ✅ completed (a6c9f43)
  │  Key finding: no autoregressive gen → CP2 suspended
  ▼
Step 2: Timing system ─── ✅ completed
  │
  ├──────────────────┐
  ▼                  ▼
Step 3: Data structs (parallel dev)
  │
  ▼
Step 4: Orchestrator (CP1 + CP3) ──── CP2 suspended, not on critical path
  │
  ▼
Step 5: ★ Feasibility experiment ★  ── if failed ──> Reevaluate approach
  │ (passed)
  ▼
Step 6: CP1/CP3 refinement + CP3 deferred writer
  │
  ▼
Step 7: Warm start (may defer — originally for CP2)
  │
  ▼
Step 8: System efficiency optimization (async, GPU, stream)
  │
  ▼
Step 9: ★ Query key research ★ (experiment-intensive)
  │
  ▼
Step 10: Offline pre-fill
  │
  ▼
Step 11: Integration testing
  │
  ▼
Step 12: Advanced features (on demand)
         ├── CP2 re-enablement (when representation extraction available)
```

**★-marked steps are critical experiment milestones** whose conclusions directly determine the direction of subsequent work, or even whether to continue.

---

### Estimated Effort Per Step

| Step | Status | Work Type | Primary Deliverable |
|------|--------|-----------|---------------------|
| 0 | ✅ Merged into Step 1 | Reading + analysis | (included in Step 1 log) |
| 1 | ✅ Completed | Staged API + Interceptor | pi0_pytorch.py public API + interceptor.py |
| 2 | ✅ Completed | Infrastructure dev | timing.py + SystemTimer |
| 3 | ⚠️ Code landed, no tests | Infrastructure dev | Data structures + VectorStore backend |
| 4 | Pending | Core development | Orchestrator (CP1 + CP3) |
| 5 | Pending | **Experiment** | Feasibility report |
| 6 | Pending | Core development | CP1/CP3 refinement + CP3 deferred writer |
| 7 | Pending (may defer) | Dev + **Experiment** | Warm start (originally for CP2) |
| 8 | Pending | Performance optimization | Async/GPU/Stream |
| 9 | Pending | **Experiment** | Key builder research |
| 10 | Pending | Tooling dev | Pre-fill script |
| 11 | Pending | Testing | Stability + A/B report |
| 12 | Pending | Advanced | On demand (includes CP2 re-enablement) |

> **Note**: Steps 2 and 3 have no dependency on each other and can be developed in parallel. Step 7 (warm start) was originally designed for CP2; with CP2 suspended, it may be deferred or repurposed.