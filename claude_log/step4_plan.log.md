# Step 4 Plan: Orchestrator Skeleton (CP1 End-to-End + CP3 Skeleton)

> This file contains the full Step 4 planning text moved out from the discussion log.

## Implementation Plan（完整版）

以下为从讨论阶段整理出的计划全文与答辩后修订版。

### 文件清单

| File | Action | Stability |
|------|--------|-----------|
| `src/openpi/cache/components/__init__.py` | Create | — |
| `src/openpi/cache/components/key_builder.py` | Create | 🔴 |
| `src/openpi/cache/components/gate.py` | Create | 🟡 |
| `src/openpi/cache/components/judge.py` | Create | 🔴 |
| `src/openpi/cache/orchestrator.py` | Create | 🟢 |
| `src/openpi/cache/interceptor.py` | Modify | 🟢 |
| `src/openpi/cache/__init__.py` | Modify | — |
| Step 3 文件（4个） | Modify（补耦合注释） | — |

### 核心设计决策

1. **KeyBuilder 双职责**：`collect()` 暂存 GPU tensor 引用（零拷贝），`build()` 时才 GPU normalize + D2H
2. **Gate/Judge 接收 `cached_data`**：兼容未来读取 KeyBuilder 数据的实现
3. **Orchestrator 只通过 `CacheStorage` facade 交互**：不碰底层 backend
4. **Orchestrator 持有 timer**：统一管理细粒度计时，组件不持有 timer
5. **Interceptor `orchestrator=None` 时零开销退化**：向后兼容
6. **每个文件包含 Coupling map 注释**：DEPENDS ON / CONSUMED BY / IF CHANGED
7. **补齐 Step 3 文件的耦合注释**

### 实现顺序

1. `components/__init__.py` → 2. `key_builder.py` → 3. `gate.py` → 4. `judge.py` → 5. `orchestrator.py` → 6. `interceptor.py` 修改 → 7. `__init__.py` 更新 → 8. Step 3 耦合注释补齐

---


## Implementation Plan v2（答辩后修订版全文）

# Step 4: Orchestrator Skeleton (CP1 End-to-End + CP3 Skeleton) — Implementation Plan v2

> Post-defense revision. Changes from v1 marked with `[v2]`.

## Context

Step 4 connects cache check logic to the inference pipeline. Pre-requisites:
- Step 1 (Validated): Staged API (`run_stage1/2/3`) + `InferenceInterceptor` skeleton (3 `TODO(Step 4)` slots)
- Step 2 (Validated): `SystemTimer` timing system
- Step 3 (high-risk): Storage layer (`CacheStorage` facade + `VectorStoreBackend` ABC + Qdrant backend)

**Scope (revised after defense)**:
- **CP1 end-to-end**: check + write + hit early return (the only fully closed loop)
- **CP3 skeleton only**: infrastructure + interfaces + stub, no actual write or skip (deferred to Step 6)
- CP2 remains suspended (no command embedding in PyTorch path)

**Core constraints** (from discussion log):
- KeyBuilder caches data on GPU for Gate/Judge to read; zero unnecessary copies
- All storage interaction through `CacheStorage` facade only — never touch backend directly
- No `--collect` integration; KeyBuilder reads Stage outputs directly (approach B)
- Model-transparent — zero model code changes
- End-to-end tests deferred to after code completion
- English comments with data flow direction and coupling annotations

---

## Files to Create / Modify

| File | Action | Stability |
|------|--------|-----------|
| `src/openpi/cache/components/__init__.py` | **Create** | — |
| `src/openpi/cache/components/key_builder.py` | **Create** | 🔴 |
| `src/openpi/cache/components/gate.py` | **Create** | 🟡 |
| `src/openpi/cache/components/judge.py` | **Create** | 🔴 |
| `src/openpi/cache/orchestrator.py` | **Create** | 🟢 |
| `src/openpi/cache/interceptor.py` | **Modify** | 🟢 |
| `src/openpi/cache/cache_storage.py` | **Modify** | [v2] fix `_check_entry_dims` |
| `src/openpi/cache/__init__.py` | **Modify** | — |
| Step 3 files (4) | **Modify** | coupling annotations only |

No changes to: `storage_types.py` (structure), `backend_base.py` (structure), `types.py` (structure), `timing.py`, any `models_pytorch/` file.

---

## Task 4.0: [v2] Fix `_check_entry_dims` in `cache_storage.py`

**Why**: Current `_check_entry_dims` requires ALL backend-declared fields to be present in every entry. This blocks the "flexible field" design — if backend declares `{robot_state: 32, vision_0: 2048}` but KeyBuilder only produces `{robot_state}`, insert raises ValueError. `README.md` line 51 already marks this as a known TODO.

**Change**: Modify `_check_entry_dims` from "require all backend fields" to "validate intersection only" — check that fields present in BOTH entry and backend have matching dims. Fields in backend but absent from entry are allowed. Fields in entry but not in backend are silently ignored (existing `_check_query_dims` behavior for search).

Add a minimum check: at least one field must overlap (same as `_check_query_dims`).

```python
def _check_entry_dims(self, entry: CacheEntry) -> None:
    # [v2] Intersection-based validation: only check fields present in both
    # entry and backend. At least one must overlap.
    active = set(entry.query_keys.keys()) & set(self._dims.keys())
    if not active:
        raise ValueError(
            f"entry.query_keys has no fields matching backend. "
            f"Backend fields: {set(self._dims.keys())}, "
            f"entry fields: {set(entry.query_keys.keys())}"
        )
    for field_name in active:
        expected = self._dims[field_name]
        shape = tuple(entry.query_keys[field_name].shape)
        if shape != (expected,):
            raise ValueError(
                f"entry.query_keys[{field_name!r}] shape mismatch: "
                f"expected ({expected},), got {shape}"
            )
```

---

## Task 4.1: KeyBuilder (`src/openpi/cache/components/key_builder.py`)

### Design

KeyBuilder has two responsibilities: (1) collect and cache raw data from Stage outputs; (2) build query keys on demand. Gate/Judge can read cached data.

**[v2] Available fields from Stage API** (approach B, direct read):
- `Stage1Output.state`: `[B, 32]` → maps to `ROBOT_STATE` — **available now**
- `Stage1Output.prefix_embs`: `[B, prefix_len, emb_dim]` → mixed vision+language, **not separable** into `vision_0/1/2` and `prompt_emb` without hooks or Stage API extension
- `Stage3Output.action_chunk`: `[B, 50, 32]` → available for CP3 key construction

**[v2] Conclusion**: PlaceholderKeyBuilder uses `ROBOT_STATE` only. `prefix_embs` as experimental second option (mean pool over seq dim) can be added later. Per-camera vision and prompt embedding require Stage API extension or hooks — not in Step 4 scope.

### Protocol

```python
@runtime_checkable
class QueryKeyBuilder(Protocol):
    """Build query keys from stage outputs for cache lookup.

    Data flow: Stage outputs (GPU) -> collect() -> build() -> query_keys dict
    Coupling:
      - DEPENDS ON: Stage1Output/Stage3Output field shapes (models_pytorch/pi0_pytorch.py)
      - CONSUMED BY: CacheOrchestrator.check(), Gate (via cached_data), Judge (via cached_data)
      - FEEDS INTO: CacheStorage.search() via QuerySpec (must match backend vector_dims)
      - IF CHANGED: Orchestrator QuerySpec construction, Judge threshold calibration
    """

    def collect(self, checkpoint_id: CheckpointID, **stage_outputs) -> None:
        """Collect and cache raw data from stage outputs.

        Tensors kept on original device (GPU) — no CPU transfer.
        Gate/Judge may read cached_data before build() is called.

        SAFETY: References are valid within a single infer() call.
        The staged path uses max-autotune-no-cudagraphs, so stage outputs
        are regular GPU tensors — not CUDAGraph-managed buffers.
        """
        ...

    def build(self, checkpoint_id: CheckpointID) -> dict[str, torch.Tensor]:
        """Build query key vectors from collected data.

        Returns dict of {field_name: [dim] CPU float32 L2-normalized}.
        Field names from CACHE_QUERY_FIELDS (openpi.cache.types).
        Crossing to storage boundary: tensors are materialized here via
        .cpu().float() — this is the ONLY D2H transfer point.
        """
        ...

    @property
    def cached_data(self) -> dict[str, torch.Tensor]:
        """Expose collected raw tensors (on original device) for Gate/Judge.
        Lifetime: valid from collect() until next collect() or clear().
        """
        ...

    def clear(self) -> None:
        """Release all cached references. Called at end of each inference cycle."""
        ...
```

### PlaceholderKeyBuilder

```python
class PlaceholderKeyBuilder:
    """Simplest key builder: raw state vector as query key.

    CP1: state [B, 32] -> L2 normalize on GPU -> [32] CPU float32
    CP3: state [B, 32] -> L2 normalize on GPU -> [32] CPU float32
         (same as CP1 for now; concat with action deferred to Step 6)

    Data flow: Stage1Output.state (GPU) -> _cache (GPU ref) -> build() -> CPU normalized

    Coupling:
      - DEPENDS ON: Stage1Output.state shape [B, 32]
      - IF Stage output shapes change: build() output dims change -> backend vector_dims must match

    Tensor lifecycle:
      - _cache: GPU references only, no clone (safe within single infer() call)
      - build() output: CPU float32 (materialized, safe for storage)
    """

    def __init__(self) -> None:
        self._cache: dict[str, torch.Tensor] = {}  # GPU tensors, no copy

    def collect(self, checkpoint_id, **stage_outputs):
        self._cache.clear()
        if "stage1" in stage_outputs:
            # Hold reference only — tensor stays on GPU, no clone needed.
            # SAFETY: staged path uses max-autotune-no-cudagraphs, so stage
            # outputs are regular GPU tensors, not CUDAGraph-managed buffers.
            # Reference is valid within a single infer() call.
            self._cache["state"] = stage_outputs["stage1"].state  # [B, 32] GPU
        if "stage3" in stage_outputs:
            self._cache["action_chunk"] = stage_outputs["stage3"].action_chunk  # [B, 50, 32] GPU

    def build(self, checkpoint_id):
        import torch
        import torch.nn.functional as F
        from openpi.cache.types import ROBOT_STATE, CheckpointID

        state = self._cache["state"][0]  # [32] GPU, drop batch dim

        if checkpoint_id == CheckpointID.CP1:
            key = F.normalize(state, dim=0)  # L2 norm on GPU (~1μs kernel)
            return {ROBOT_STATE: key.cpu().float()}  # single D2H transfer
        elif checkpoint_id == CheckpointID.CP3:
            # [v2] Same as CP1 for now. Step 6 will concat state + action.
            key = F.normalize(state, dim=0)
            return {ROBOT_STATE: key.cpu().float()}

    @property
    def cached_data(self):
        return self._cache

    def clear(self):
        self._cache.clear()
```

**Hardware efficiency**:
- `collect()`: GPU tensor reference only (zero copy, zero alloc)
- `F.normalize`: GPU kernel (~1μs single launch)
- `.cpu().float()`: single D2H transfer, only in `build()` output
- `clear()`: release references, no GPU ops

---

## Task 4.2: Gate (`src/openpi/cache/components/gate.py`)

### Protocol

```python
@runtime_checkable
class GateFunction(Protocol):
    """Decide whether to perform a cache lookup.

    Data flow: KeyBuilder.cached_data (GPU) -> gate() -> bool
    Coupling:
      - MAY DEPEND ON: KeyBuilder.cached_data (read-only, for state-change gates)
      - CONSUMED BY: CacheOrchestrator.check() — if False, skip search entirely
      - DOES NOT interact with: CacheStorage or any Step 3 component
      - IF CHANGED: Only affects Orchestrator's search frequency, no downstream impact
    """

    def __call__(
        self,
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
    ) -> bool:
        """Return True if cache should be searched, False to skip.

        Args:
            checkpoint_id: CP1 or CP3.
            cached_data: Raw tensors from KeyBuilder.cached_data (on GPU).
        """
        ...
```

### AlwaysSearchGate

```python
class AlwaysSearchGate:
    """Always search the cache. Simplest gate for initial development.

    Data flow: (no data consumed) -> always True
    Coupling: None — no dependencies on any other component.
    """

    def __call__(self, checkpoint_id, cached_data):
        return True
```

---

## Task 4.3: Judge (`src/openpi/cache/components/judge.py`)

### HitType + Protocol

```python
from enum import Enum, auto

class HitType(Enum):
    """Cache hit classification.

    Coupling:
      - CONSUMED BY: CacheOrchestrator (packs into CheckResult), Interceptor (controls stage skip)
    """
    MISS = auto()
    FULL_HIT = auto()
    # WARM_START = auto()  # Step 7: flow matching warm start

@runtime_checkable
class SimilarityJudge(Protocol):
    """Judge whether a search result constitutes a cache hit.

    Data flow: SearchResultLite.score (from CacheStorage) -> judge() -> (HitType, winner_id)
    Coupling:
      - DEPENDS ON: SearchResultLite.score semantics (Step 3 backend-dependent)
        * Single-field cosine: score in [-1, 1]
        * Multi-field RRF: small positive numbers, scale depends on RRF k param
        * IF backend changes: thresholds MUST be recalibrated
      - MAY DEPEND ON: KeyBuilder.cached_data (for future re-scoring judges)
      - CONSUMED BY: CacheOrchestrator.check()
      - DOES NOT call: CacheStorage or fetch_payload (pure judgment, no side effects)
      - IF CHANGED: Only affects hit/miss decision, no downstream structural impact
    """

    def __call__(
        self,
        results: list[SearchResultLite],
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
    ) -> tuple[HitType, Optional[str]]:
        """Judge the top search results.

        Args:
            results: Search results sorted by descending score (from CacheStorage).
            checkpoint_id: CP1 or CP3.
            cached_data: Raw tensors from KeyBuilder.cached_data.

        Returns:
            (hit_type, winner_id): HitType and the id of the winning entry (None if MISS).
        """
        ...
```

### ThresholdJudge

```python
class ThresholdJudge:
    """Simple threshold-based judge: top-1 score > threshold -> FULL_HIT.

    Data flow: results[0].score -> compare threshold -> HitType
    Coupling:
      - DEPENDS ON: score range from CacheStorage backend (see SimilarityJudge docstring)
      - IF backend or key builder changes: threshold value likely needs recalibration
    """

    def __init__(self, cp1_threshold: float = 0.98, cp3_threshold: float = 0.95) -> None:
        self._thresholds = {
            CheckpointID.CP1: cp1_threshold,
            CheckpointID.CP3: cp3_threshold,
        }

    def __call__(self, results, checkpoint_id, cached_data):
        if not results:
            return HitType.MISS, None
        top = results[0]
        threshold = self._thresholds.get(checkpoint_id, 0.98)
        if top.score >= threshold:
            return HitType.FULL_HIT, top.id
        return HitType.MISS, None
```

---

## Timing Integration

### Design

`SystemTimer` provides `register_probe(name, backend)` + `with timer.measure(name)`. When `enabled=False`, `measure()` is pure no-op (zero overhead).

**Orchestrator holds timer reference** and times each sub-step. Components (KeyBuilder/Gate/Judge) do NOT hold timer — timing managed centrally to avoid spreading timer refs into unstable components.

### Probe Table

| Probe | Backend | Target | Registered by |
|-------|---------|--------|---------------|
| `cp1_check` | cpu | CP1 overall check | Interceptor |
| `cp1_gate` | cpu | CP1 gate decision | Orchestrator |
| `cp1_collect` | cpu | CP1 KeyBuilder.collect() | Orchestrator |
| `cp1_build` | cpu | CP1 KeyBuilder.build() (GPU normalize + D2H) | Orchestrator |
| `cp1_search` | cpu | CP1 CacheStorage.search() | Orchestrator |
| `cp1_judge` | cpu | CP1 judge decision | Orchestrator |
| `cp1_fetch` | cpu | CP1 fetch_payload (hit only) | Orchestrator |
| `cp1_write` | cpu | CP1 overall write | Interceptor |
| `cp3_check` | cpu | CP3 overall check | Interceptor |
| `cp3_*` | cpu | (same sub-steps as CP1) | Orchestrator |

**Why all CPU backend**:
- Gate/Judge: pure Python logic
- KeyBuilder.build(): GPU `F.normalize` ~1μs, dominated by `.cpu()` D2H sync; `perf_counter` captures this accurately
- CacheStorage.search(): CPU dispatch to backend (Qdrant/FAISS)
- CUDA Event on default stream with implicit sync (`.cpu()`) would give same result but with extra event overhead

```
# NOTE: Step 2 design envisions CUDA Event timing for KeyBuilder once
# cache_stream is introduced (Step 8). Current CPU backend measures
# wall-clock time, which equals GPU time when operations run on the
# default stream with implicit sync (.cpu() calls).
```

**Performance**: `perf_counter_ns()` ~50ns/call, ~6 probes per check = ~300ns. Negligible vs ~50ms inference. `enabled=False` skips entirely.

### Orchestrator timing pattern

```python
def check(self, checkpoint_id, **stage_outputs) -> CheckResult:
    prefix = checkpoint_id.name.lower()  # "cp1" or "cp3"

    with self._timer.measure(f"{prefix}_collect"):
        self._key_builder.collect(checkpoint_id, **stage_outputs)

    with self._timer.measure(f"{prefix}_gate"):
        should_search = self._gate(checkpoint_id, self._key_builder.cached_data)
    if not should_search:
        return CheckResult(hit_type=HitType.MISS)

    with self._timer.measure(f"{prefix}_build"):
        query_keys = self._key_builder.build(checkpoint_id)

    with self._timer.measure(f"{prefix}_search"):
        spec = QuerySpec(query_keys=query_keys, top_k=1, checkpoint_id=checkpoint_id)
        results = self._storage.search(spec)

    with self._timer.measure(f"{prefix}_judge"):
        hit_type, winner_id = self._judge(
            results, checkpoint_id, self._key_builder.cached_data
        )

    if hit_type == HitType.FULL_HIT and winner_id is not None:
        with self._timer.measure(f"{prefix}_fetch"):
            payload = self._storage.fetch_payload(winner_id)
        return CheckResult(hit_type=hit_type, payload=payload,
                           score=results[0].score, entry_id=winner_id)

    return CheckResult(hit_type=HitType.MISS)
```

---

## Task 4.4: Orchestrator (`src/openpi/cache/orchestrator.py`)

### CheckResult

```python
@dataclass
class CheckResult:
    """Result of a cache check at one checkpoint.

    Data flow: Orchestrator.check() -> CheckResult -> Interceptor (decision point)
    Coupling:
      - CONSUMED BY: InferenceInterceptor (reads hit_type to decide stage skip)
      - CONTAINS: CachePayload from CacheStorage.fetch_payload() (Step 3 type)
    """
    hit_type: HitType
    payload: Optional[CachePayload] = None  # non-None only on FULL_HIT
    score: Optional[float] = None
    entry_id: Optional[str] = None
```

### CacheOrchestrator

```python
class CacheOrchestrator:
    """Orchestrate cache check and write operations.

    Combines pluggable components (KeyBuilder, Gate, Judge) with CacheStorage.
    All storage interaction goes through CacheStorage facade — never touches
    VectorStoreBackend directly.

    Data flow overview:
      check():  Interceptor -> collect -> gate -> build -> storage.search -> judge -> CheckResult
      write():  Interceptor -> collect -> build -> CacheEntry -> storage.insert

    Coupling:
      - DEPENDS ON: QueryKeyBuilder, GateFunction, SimilarityJudge (Step 4 components)
      - DEPENDS ON: CacheStorage facade (Step 3) — search/insert/fetch_payload
      - CONSUMED BY: InferenceInterceptor (calls check/write at TODO slots)
      - DOES NOT depend on: VectorStoreBackend, Qdrant, or any specific backend
      - IF CHANGED: Interceptor's cache integration logic may need updating
    """

    def __init__(
        self,
        storage: CacheStorage,
        key_builder: QueryKeyBuilder,
        gate: GateFunction,
        judge: SimilarityJudge,
        timer: Optional[SystemTimer] = None,
    ) -> None:
        ...
        self._timer = timer if timer is not None else SystemTimer(enabled=False)
        # Register fine-grained probes for each checkpoint's sub-steps
        for cp in ("cp1", "cp3"):
            for step in ("collect", "gate", "build", "search", "judge", "fetch"):
                self._timer.register_probe(f"{cp}_{step}", backend="cpu")

    def check(self, checkpoint_id: CheckpointID, **stage_outputs) -> CheckResult:
        """Cache check pipeline: collect -> gate -> build -> search -> judge -> fetch.

        Flow:
          1. key_builder.collect(checkpoint_id, **stage_outputs)
          2. gate(checkpoint_id, key_builder.cached_data) -> if False: MISS
          3. key_builder.build(checkpoint_id) -> query_keys
          4. storage.search(
                 QuerySpec(
                     query_keys=query_keys,
                     top_k=1,
                     checkpoint_id=checkpoint_id,
                 )
             )  # [审阅人修改] 避免按位置传参把 checkpoint_id 误传到 top_k
          5. judge(results, checkpoint_id, key_builder.cached_data) -> (hit_type, winner_id)
          6. if FULL_HIT: storage.fetch_payload(winner_id) -> payload
          7. return CheckResult

        Note: collect() before gate() so Gate can access cached_data.
        fetch_payload called by Orchestrator, not Judge (Judge is pure judgment).
        """
        ...  # see timing pattern above for full implementation

    def write(self, checkpoint_id: CheckpointID, payload: CachePayload,
              **stage_outputs) -> None:
        """Write a cache entry (synchronous). Async deferred to Step 8.

        Flow:
          1. key_builder.collect(...) [if not already collected this cycle]
          2. key_builder.build(...) -> query_keys
          3. Construct CacheEntry(id=stable_hash(...), checkpoint_id, query_keys, payload)
          4. storage.insert(entry)

        Caller must ensure payload tensors are CPU float32.
        """
        ...

    # [v2] CP3 stub interfaces — real implementation in Step 6
    def schedule_next_action(self, action: torch.Tensor) -> None:
        """CP3: schedule a cached action for the next inference cycle.
        Stub in Step 4 — does nothing. Step 6 implements with DeferredWriter.
        """
        pass

    def should_skip_inference(self) -> Optional[torch.Tensor]:
        """CP3: check if previous cycle scheduled an action for this cycle.
        Stub in Step 4 — always returns None (no skip). Step 6 implements.
        """
        return None

    def clear(self) -> None:
        """Release per-cycle state. Called at end of each inference cycle."""
        self._key_builder.clear()
```

**check() notes**:
- collect before gate — ensures Gate can access cached_data
- search top_k=1 (simplest, may adjust later)
- fetch_payload only on FULL_HIT (two-phase search, avoids unnecessary tensor transfer)

---

## Task 4.5: Interceptor Integration (`src/openpi/cache/interceptor.py`)

### `__init__` changes

```python
def __init__(
    self,
    policy: _policy.Policy,
    timer: Optional[SystemTimer] = None,
    orchestrator: Optional["CacheOrchestrator"] = None,  # [v2] NEW
) -> None:
    ...
    self._orchestrator = orchestrator

    # Register cache probes if orchestrator is present
    if orchestrator is not None:
        self._timer.register_probe("cp1_check", backend="cpu")
        self._timer.register_probe("cp1_write", backend="cpu")
        self._timer.register_probe("cp3_check", backend="cpu")
```

### `infer()` changes

```python
def infer(self, obs, *, noise=None):
    # ---- 1. Input transforms (unchanged) ----
    ...
    observation = _model.Observation.from_dict(inputs)
    start_noise = ...

    # [v2] CP3 consume point: check if previous cycle pre-scheduled an action.
    # Data flow: orchestrator._next_action_scheduled -> action (or None)
    # Stub in Step 4: always None. Step 6 implements actual skip.
    if self._orchestrator is not None:
        scheduled_action = self._orchestrator.should_skip_inference()
        if scheduled_action is not None:
            # Early return: skip entire inference, use pre-scheduled action
            outputs = {"state": inputs["state"], "actions": scheduled_action}
            outputs = jax.tree.map(
                lambda x: np.asarray(x[0, ...].detach().cpu()) if isinstance(x, torch.Tensor) else x,
                outputs,
            )
            outputs = self._output_transform(outputs)
            self._orchestrator.clear()
            return outputs

    # ---- 2. Staged inference with cache checks ----
    torch.compiler.cudagraph_mark_step_begin()
    with self._timer.measure("total_inference"):
        with torch.no_grad():
            with self._timer.measure("stage1_vision"):
                stage1 = self._stage1_fn(observation)

            # CP1: check cache after Stage 1.
            # Data flow: stage1 -> orchestrator.check(CP1) -> CheckResult
            # On HIT: skip stage2 + stage3, return cached action.
            if self._orchestrator is not None:
                with self._timer.measure("cp1_check"):
                    cp1_result = self._orchestrator.check(
                        CheckpointID.CP1, stage1=stage1
                    )
                if cp1_result.hit_type == HitType.FULL_HIT:
                    # Build outputs from cached payload.
                    # CachePayload.action_chunk is CPU float32 [50, 32].
                    # Move to device and add batch dim to match normal path.
                    outputs: dict[str, Any] = {
                        "state": inputs["state"],
                        "actions": cp1_result.payload.action_chunk.to(
                            self._pytorch_device
                        )[None, ...],  # [1, 50, 32]
                    }
                    outputs = jax.tree.map(
                        lambda x: np.asarray(x[0, ...].detach().cpu()), outputs
                    )
                    outputs = self._output_transform(outputs)
                    self._orchestrator.clear()
                    return outputs

            # Stage 2: Gemma 2B backbone forward pass -> KV cache.
            with self._timer.measure("stage2_llm"):
                stage2 = self._stage2_fn(stage1)

            # CP2 slot: remains commented out (suspended).
            # TODO(Step 7): insert CP2 cache check here.

            # Stage 3: Action Expert — 10-step Euler flow-matching loop.
            with self._timer.measure("stage3_flow"):
                stage3 = self._stage3_fn(stage2, noise=start_noise)

            # Post-inference cache operations (only on normal path, not on hit path).
            if self._orchestrator is not None:
                # Write CP1 entry for future lookups.
                # Data flow: stage1.state -> key, stage3.action_chunk -> payload -> storage
                with self._timer.measure("cp1_write"):
                    cp1_payload = CachePayload(
                        action_chunk=stage3.action_chunk[0].detach().cpu().float()
                    )
                    self._orchestrator.write(
                        CheckpointID.CP1, cp1_payload, stage1=stage1
                    )

                # [v2] CP3 check: infrastructure validation only.
                # No CP3 entries exist in Step 4, so this always returns MISS.
                # Real CP3 write + skip deferred to Step 6 (DeferredWriter).
                with self._timer.measure("cp3_check"):
                    cp3_result = self._orchestrator.check(
                        CheckpointID.CP3, stage1=stage1, stage3=stage3
                    )

                # Release per-cycle cached data
                self._orchestrator.clear()

    # ---- 3. Build outputs (unchanged) ----
    outputs: dict[str, Any] = {
        "state": inputs["state"],
        "actions": stage3.action_chunk,
    }
    outputs = jax.tree.map(
        lambda x: np.asarray(x[0, ...].detach().cpu()), outputs
    )
    outputs = self._output_transform(outputs)
    return outputs
```

**Key design points**:
- `orchestrator=None` degrades to original behavior (zero overhead)
- [v2] CP3 consume slot at `infer()` entry (before any stage) — stub in Step 4, always None
- CP1 hit early return duplicates output build logic (avoids modifying normal path)
- CP1 write after normal inference (not on hit path — avoids caching cached results)
- [v2] CP3 check is infrastructure-only (always MISS, no CP3 entries written)
- All cache operations wrapped in timer probes

---

## Task 4.6: Module Exports (`src/openpi/cache/__init__.py`)

Add exports:

```python
from openpi.cache.components.key_builder import QueryKeyBuilder, PlaceholderKeyBuilder
from openpi.cache.components.gate import GateFunction, AlwaysSearchGate
from openpi.cache.components.judge import SimilarityJudge, ThresholdJudge, HitType
from openpi.cache.orchestrator import CacheOrchestrator, CheckResult
```

---

## Task 4.7: Coupling Annotations for Step 3 Files

Add coupling map to module docstrings in existing Step 3 files:

**`types.py`**:
```
Coupling map:
  DEPENDS ON:  nothing (leaf module)
  CONSUMED BY: KeyBuilder (field name constants), Orchestrator (CheckpointID),
               storage_types (CheckpointID), cache_storage (via storage_types)
  IF CHANGED:  All consumers must update field references
```

**`storage_types.py`**:
```
Coupling map:
  DEPENDS ON:  types.py (CheckpointID)
  CONSUMED BY: KeyBuilder (CachePayload, CacheEntry construction),
               Orchestrator (QuerySpec, SearchResultLite, SearchResult),
               CacheStorage (all types), backends (all types)
  IF CHANGED:  Orchestrator QuerySpec construction, KeyBuilder payload construction,
               Judge threshold calibration (if score semantics change)
```

**`cache_storage.py`**:
```
Coupling map:
  DEPENDS ON:  backend_base.py (VectorStoreBackend), storage_types.py (all types)
  CONSUMED BY: CacheOrchestrator (search/insert/fetch_payload) — ONLY consumer
  IF CHANGED:  Orchestrator's search/write calls may need updating
  NOTE:        Step 4+ code must ONLY interact via this facade, never via backend directly
```

**`backend_base.py`**:
```
Coupling map:
  DEPENDS ON:  storage_types.py (CacheEntry, CachePayload, QuerySpec, etc.)
  CONSUMED BY: CacheStorage (the only caller), concrete backends (Qdrant, future FAISS)
  IF CHANGED:  CacheStorage internal calls, all concrete backend implementations
  NOTE:        Application code (Orchestrator, Interceptor) must NEVER import this
```

---

## Implementation Order

1. **`cache_storage.py`** — [v2] fix `_check_entry_dims` to intersection-based validation
2. **`components/__init__.py`** — empty module init
3. **`components/key_builder.py`** — Protocol + PlaceholderKeyBuilder
4. **`components/gate.py`** — Protocol + AlwaysSearchGate
5. **`components/judge.py`** — HitType enum + Protocol + ThresholdJudge
6. **`orchestrator.py`** — CheckResult + CacheOrchestrator (with CP3 stubs + timing)
7. **`interceptor.py`** — modify `__init__` and `infer()`, add CP1 check/write + CP3 consume slot + CP3 skeleton check
8. **`__init__.py`** — update exports
9. **Step 3 coupling annotations** — add to `types.py`, `storage_types.py`, `cache_storage.py`, `backend_base.py`

---

## Verification

- `uv run python -c "from openpi.cache import CacheOrchestrator, PlaceholderKeyBuilder, AlwaysSearchGate, ThresholdJudge, HitType, CheckResult"` — import succeeds
- `InferenceInterceptor` without orchestrator behaves identically to pre-change (regression)
- `_check_entry_dims` accepts entries with subset of backend fields (new behavior)
- End-to-end tests deferred to post-implementation
