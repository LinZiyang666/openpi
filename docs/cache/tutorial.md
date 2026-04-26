# OpenPI Inference Cache System — Complete Tutorial

> For deep design rationale and checkpoint theory, see [../architecture/cache_system.md](../architecture/cache_system.md).
>
> **AGENT: READ FIRST** — This file is a registered subsystem rule document per [`WORKING_AGREEMENT.md` §8](../../WORKING_AGREEMENT.md#8-subsystem-rules). Component isolation rules (§15) and testing patterns (§16) carry Working Agreement authority.

This is a self-contained guide for developers who want to understand, configure, or extend the multi-level inference cache.

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
13. [Trajectory Search](#13-trajectory-search)
14. [Episode Write Path](#14-episode-write-path)
15. [Component Isolation Rules](#15-component-isolation-rules)
16. [Testing Patterns](#16-testing-patterns)
17. [Current Validation Status](#17-current-validation-status)

---

## 1. Glossary

| Term | Definition |
|------|-----------|
| **CheckpointID** | Enum (`CP1`, `CP3`). Positions in inference pipeline where cache checks occur. Defined in `types.py`. |
| **CP1** | After Stage 1 (vision + tokenisation). A hit skips Stage 2 + Stage 3. |
| **CP3** | After Stage 3 (flow matching). Currently for infrastructure validation only. |
| **CACHE_QUERY_FIELDS** | Five canonical field names: `vision_0`, `vision_1`, `vision_2`, `prompt_emb`, `robot_state`. |
| **CacheEntry** | A single unit written to the vector store. Contains `id`, `checkpoint_id`, `query_keys`, `CachePayload`, `prev_ids`, `next_ids`, `trajectory_id`. |
| **CachePayload** | Stored action data: `action_chunk` [50, 32] (always required), `intermediates` (CP1 warm start, online: default 3 timesteps, offline: up to 9), `task_key`. |
| **QuerySpec** | Everything a backend needs for one search: `query_keys`, `top_k`, `filters`, `fusion_weights`, `backend_hints`, `trajectory_history`, `trajectory_weights`. Constructed **only** by SearchStrategy. |
| **QueryFilter** | Dynamic per-query constraints: `task_key` (exact match), `step_range` (inclusive range). |
| **SearchResultLite** | Lightweight search result (no payload): `id`, `score`, `checkpoint_id`. Phase 1 of two-phase search. |
| **CheckResult** | Output of `Orchestrator.check()`: `hit_type`, optional `payload`, `score`, `entry_id`, `query_keys`. |
| **HitType** | Enum: `MISS`, `FULL_HIT`, `WARM_START`. |
| **StepRecord** | Per-step staging record buffered during episode for trajectory write. |
| **EpisodeRecord** | Entire episode staging record, passed to WritePolicy to decide write. |
| **TrajectoryMixin** | Shared history buffer logic mixed into all SearchStrategy implementations. |
| **WritePolicy** | Pluggable switch deciding whether to write trajectory at episode end. |
| **RRF** | Reciprocal Rank Fusion — rank-based multi-field fusion. Score = `Σ w_f / (k + rank_f)`. |
| **Weighted Score Sum** | Similarity-based multi-field fusion. Preserves score magnitudes for trajectory search. |

---

## 2. Architecture Overview

```
InferenceInterceptor          ← BasePolicy drop-in, wraps real policy
  └─ CacheOrchestrator        ← Assembles components, runs check + episode write
       ├─ QueryKeyBuilder      ← GPU → CPU query vectors
       ├─ GateFunction         ← Should we search?
       ├─ SearchStrategy       ← Builds QuerySpec, calls storage (+ TrajectoryMixin)
       │    └─ TrajectoryMixin ← History buffer for trajectory search
       ├─ SimilarityJudge      ← Hit or miss?
       ├─ WritePolicy          ← Should we write this episode?
       └─ CacheStorage         ← Thread-safe facade
            └─ VectorStoreBackend ABC
                 └─ InMemoryBackend      (primary, supports trajectory search)
```

**Layer responsibilities:**

| Layer | File | Role |
|-------|------|------|
| Interceptor | `cache/interceptor.py` | Wraps policy, inserts CP1/CP3 checks, episode lifecycle |
| Orchestrator | `cache/orchestrator.py` | Coordinates gate → build → search → judge → fetch; episode write |
| Components | `cache/components/*.py` | Pluggable protocols: KeyBuilder, Gate, Judge, SearchStrategy, WritePolicy |
| Storage | `cache/cache_storage.py` | Thread safety, dimension validation, filter checking |
| Backend | `cache/backend_base.py` + `cache/backends/` | Vector DB implementations |
| Config | `cache/config.py` | YAML → dataclass → factory → component instances |

---

## 3. Data Flow

### 3.1 Check Pipeline

When `orchestrator.check(checkpoint_id, **stage_outputs)` is called:

```
Step 1: key_builder.collect(checkpoint_id, stage1=...)
        └─ Hold GPU tensor references (no copy)

Step 2: key_builder.build(checkpoint_id)
        └─ GPU → CPU float32 (THE ONLY D2H TRANSFER POINT)

Step 3: gate(checkpoint_id, key_builder.cached_data)
        └─ If False → record miss, record_query_keys, return CheckResult(MISS)

Step 4: Construct SearchContext(query_keys, checkpoint_id, current_step, task_key)

Step 5: search_strategy.search(ctx)
        └─ record_query_keys (TrajectoryMixin)
        └─ Build QueryFilter (step filtering)
        └─ Build QuerySpec (fusion + trajectory fields)
        └─ CacheStorage.search(spec) → Backend.search(spec)
        └─ Returns list[SearchResultLite] sorted descending

Step 6: judge(results, checkpoint_id, cached_data)
        └─ Returns JudgeResult(hit_type, winner_id, start_t)
        └─ If MISS → increment miss counter, return CheckResult(MISS)

Step 7: storage.fetch_payload(winner_id)
        └─ WARM_START: validate payload completeness, downgrade to MISS if incomplete

Step 8: Increment step counter (CP1 only)

Step 9: Return CheckResult(FULL_HIT, payload, score, entry_id, query_keys)
```

**query_keys is returned on ALL paths** (hit, miss, gate skip) — needed by `buffer_for_write()`.

### 3.2 Episode Write Pipeline

Replaces the old per-step `write()` method. Trajectory-aware:

```
During episode:
  Interceptor calls orchestrator.broadcast_action(action)
  Interceptor calls orchestrator.buffer_for_write(query_keys, action)
    └─ Appends StepRecord to _episode_steps

At episode end:
  orchestrator.on_episode_end()
    └─ Build EpisodeRecord from buffered steps + miss counts
    └─ WritePolicy.should_write(episode_record)
    └─ If yes: _build_entry_chain() → linked CacheEntry list → batch_insert
    └─ Reset episode buffers
```

Entry chain format:
- Entry ID: `"{trajectory_id}:{step_idx}"`
- `prev_ids`: points to previous step's entry
- `next_ids`: points to next step's entry
- All entries in a chain share the same `trajectory_id` (UUID)

---

## 4. Component: KeyBuilder

**Source**: `src/openpi/cache/components/key_builder.py`

### Existing Implementations

| Type | Output dims | Fields | Use case |
|------|-------------|--------|----------|
| `placeholder` | `{robot_state: 32}` | robot_state only | Testing |
| `cp1_mean_pool` | `{vision_0: 2048, robot_state: 32, ...}` | Mean pool over 256 tokens → 2048d | **Recommended** |
| `cp1_spatial_pool_16` | `{vision_0: 32768, ...}` | 4×4 spatial pool | High resolution |
| `cp1_spatial_pool_4` | `{vision_0: 8192, ...}` | 2×2 spatial pool = 4 tokens. Legacy alias: `cp1_spatial_pool_64` (named after 64× compression). | Medium |
| `cp1_max_pool` | `{vision_0: 2048, ...}` | Max pool over tokens | Alternative to mean |
| `clip` | `{vision_0: 512, ...}` (ViT-B-32) | CLIP image encoder on raw input images | External vision encoder; dim depends on CLIP model |
| `full_original` | `{vision_0: 524288, ...}` | Raw flatten (Qdrant only) | Deprecated for in_memory |

---

## 5. Component: Gate

**Source**: `src/openpi/cache/components/gate.py`

Current implementation: **AlwaysSearchGate** — always returns `True`.

Gate also has lifecycle methods for trajectory support:
- `on_episode_start()` — reset per-episode state
- `record_action(action)` — receive broadcast action (currently no-op)

---

## 6. Component: Judge

**Source**: `src/openpi/cache/components/judge.py`

Judge returns `JudgeResult(hit_type, winner_id, start_t)`.

| Type | Description |
|------|-------------|
| `threshold` | FULL_HIT if `score >= threshold`. With `warm_tiers`, scores below the threshold are matched against descending tiers for WARM_START (CP1 only). |
| `always_hit` | Always returns FULL_HIT for top result. Good for testing. |

**Warm start configuration** (optional, CP1 only):

```yaml
judge:
  type: threshold
  threshold: 0.98
  warm_tiers:
    - {threshold: 0.95, start_t: 0.3}   # high similarity → skip 70% of Stage 3
    - {threshold: 0.90, start_t: 0.5}   # medium → skip 50%
    - {threshold: 0.85, start_t: 0.7}   # lower → skip 30%
```

Tiers must be strictly decreasing in threshold. `start_t` determines the flow matching resume point. When `warm_tiers` is omitted or null, warm start is disabled (backward compatible).

**Score semantics depend on fusion method:**

| Fusion | Score range | Notes |
|--------|-------------|-------|
| Single-field cosine | [-1, 1] | `0.98` = very similar |
| Weighted RRF | Small positives | Top-1 always = `Σ weights / (rrf_k + 1)` |
| Weighted Score Sum | [0, max_weight_sum] | Preserves similarity magnitude |

Judge also has `on_episode_start()` and `record_action()` lifecycle methods.

---

## 7. Component: SearchStrategy

**Source**: `src/openpi/cache/components/search_strategy.py`

### Available Strategies

| Type | Backend | Description |
|------|---------|-------------|
| `weighted_rrf_knn` | in_memory | Rank-based fusion. Good for multi-field when magnitude doesn't matter. |
| `weighted_score_sum_knn` | in_memory | Similarity-based fusion. Better for trajectory search (preserves magnitude). |
| `qdrant_weighted_rrf_knn` | qdrant | Qdrant server-side RRF. Does NOT support trajectory search. |

### TrajectoryMixin

All strategies inherit `TrajectoryMixin`, providing:

- **`record_query_keys(keys)`** — buffers current step's query keys into history and assigns it a monotonic `query_id` (used by the search session below)
- **`on_episode_start()`** — clears history buffers and mints a fresh per-strategy `search_session_id` (`uuid.uuid4().hex`); orchestrator collects this via `get_search_session_id()` and registers it with the backend
- **`get_search_session_id() → Optional[str]`** — exposes the active sid to `CacheOrchestrator._broadcast_episode_start`
- **`record_action(action)`** — receives broadcast action
- **`_build_trajectory_fields()`** — returns `{trajectory_history, trajectory_weights}` for QuerySpec when history is sufficient (else `{}`); when a session is active also adds `search_session_id` and `trajectory_query_ids` to engage the cross-step score memo

History build-up:
- Step 0: 1 entry → insufficient → single-step search
- Step 1: 2 entries → depth=2 trajectory search
- Step 2+: 3 entries → depth=3 trajectory search (if configured for depth=3)

### Search Session — Cross-Step Score Memo

Trajectory searches at successive steps share most of their per-field
cosine work — the same query vector appears at depth-1 in step *t* and
at depth-2 in step *t+1*. The "search session" capability removes that
redundant work by storing per-`(field, query_id)` similarity scores
inside `InMemoryBackend` for the lifetime of a single episode.

**Engagement is opt-in.** A SearchStrategy that does not inherit
`TrajectoryMixin` (or whose configuration leaves `trajectory_depth=1`)
emits no search_session_id, and the backend takes the trunk path with
zero overhead.

**Lifecycle in normal usage** (driven by the simulator → interceptor →
orchestrator chain):

1. `on_task_begin` *or* `on_episode_start` → orchestrator runs
   `_broadcast_episode_start()`, which (a) closes any stale strategy
   sids from a previous episode, (b) broadcasts `on_episode_start` so
   each strategy mints its own sid, (c) collects each non-None sid via
   `strategy.get_search_session_id()` and calls
   `storage.open_search_session(sid)`. From this point on, the backend
   refuses upsert / delete / `load_artifact` until the sid is closed.
2. Search calls flow through `SearchStrategy.search()`, which packages
   `search_session_id` + `trajectory_query_ids` into the `QuerySpec`.
   `InMemoryBackend._batch_field_scores` looks up cached scores by
   `(sid, field, query_id, sim_type)` and only computes the missing
   slots.
3. `on_episode_end` → orchestrator runs `_close_current_search_sessions()`
   inside a `try/finally` so all early-return branches still release
   the sids. `on_task_end` is a redundant safety net for connection
   close paths.

**Mutation contract you should know about.** While *any* strategy sid
is registered on the backend:

| Operation | Result |
|-----------|--------|
| `insert(brand-new id)` | ✅ allowed |
| `insert(existing id)` (upsert) | ❌ raises `SearchSessionActiveError` |
| `delete(ids)` | ❌ raises `SearchSessionActiveError` |
| `load_artifact(path)` | ❌ raises `SearchSessionActiveError` |

If you need to upsert / delete / load_artifact at runtime (e.g. a
maintenance script), wait for the connection to close so on_task_end
fires, or call `backend.close_search_session(sid)` for every active sid
first.

**Manual usage** (tests, ad-hoc scripts):

```python
backend = InMemoryBackend({"robot_state": 32})
backend.insert(...)                       # offline insertion: free

backend.open_search_session("my-sid")
spec = QuerySpec(..., search_session_id="my-sid", trajectory_query_ids=[5,4,3])
backend.search(spec)                      # populates cache bucket
backend.search(spec)                      # bucket hit — no recompute
backend.close_search_session("my-sid")     # bucket dropped, mutations unlocked
```

For deterministic parity testing the backend exposes a context manager
`backend.force_legacy_path()` that forces the legacy DAG path
regardless of the new fast path; cache state is not affected.

### Step Filter Modes

| Mode | Effect |
|------|--------|
| `"all"` | No step filtering |
| `"exact"` | Only entries at same step index |
| `"window"` | Entries within ±`step_window` of current step |

---

## 8. Storage Layer

**Source**: `src/openpi/cache/cache_storage.py`

`CacheStorage` is a thread-safe facade over any `VectorStoreBackend`.

| Method | Description |
|--------|-------------|
| `search(spec) → list[SearchResultLite]` | Phase 1: lightweight vector search |
| `fetch_payload(id) → CachePayload` | Phase 2: fetch tensors for winner |
| `insert(entry)` | Validate + upsert one entry |
| `batch_insert(entries) → BatchInsertResult` | Bulk insert (partial failures tolerated) |

---

## 9. Adding a Custom Vector DB Backend

### The VectorStoreBackend ABC

**Source**: `src/openpi/cache/backend_base.py`

Required methods: `vector_dims`, `supported_filters()`, `insert()`, `search()`, `fetch_payload()`, `delete()`, `count()`

### Existing Backends

**InMemoryBackend** (`backends/in_memory_backend.py`) — **Primary backend**:
- Python dict storage, brute-force search
- Multi-field fusion: `weighted_rrf` and `weighted_score_sum`
- **Trajectory search**: two-pass recursive algorithm with cross-step weighted fusion
- Supported filters: `{"checkpoint_id", "task_key", "step_range"}`
- Artifact loading from pickle files (with old artifact backward compat)
- Suitable for < 50k entries

### Wiring into Config

1. Add backend type to `_build_backend()` in `config.py`
2. Add config dataclass if needed
3. Add validation in `validate_cache_config()`

---

## 10. YAML Config System

### Full Annotated Config Reference

```yaml
enabled: true

timer:
  enabled: true
  buffer_size: 10000
  output_csv_dir: null

keys:
  vision_0:    { enabled: true,  weight: 1.0 }
  vision_1:    { enabled: false, weight: 1.0 }
  vision_2:    { enabled: false, weight: 1.0 }
  prompt_emb:  { enabled: false, weight: 1.0 }
  robot_state: { enabled: true,  weight: 1.0 }

key_builder:
  type: cp1_mean_pool   # "cp1_mean_pool" | "cp1_spatial_pool_16" | "cp1_spatial_pool_4" (alias "cp1_spatial_pool_64")
                        # | "cp1_max_pool" | "clip" | "placeholder"
                        # Note: "clip" uses open_clip ViT-B-32 by default.
                        # CLIP model variant is set at artifact build time, not in YAML.

checkpoints:
  _defaults: &cp_defaults
    gate:
      type: always_search
    search_strategy:
      type: weighted_rrf_knn       # "weighted_rrf_knn" | "weighted_score_sum_knn" (in_memory)
      top_k: 1
      step_filter: all             # "all" | "exact" | "window"
      step_window: 5
      rrf_k: 60                    # only for weighted_rrf_knn
      trajectory_depth: 1          # 1 = single-step; >1 enables trajectory search
      # trajectory_weights: [0.6, 0.3, 0.1]  # newest-first, length == trajectory_depth

  cp1:
    <<: *cp_defaults
    enabled: true
    judge:
      type: always_hit             # "always_hit" | "threshold"
      # threshold: 0.98

  cp3:
    <<: *cp_defaults
    enabled: true
    judge:
      type: always_hit

backend:
  type: in_memory                  # "in_memory" (primary)
  vector_dims:
    vision_0: 2048                 # must match artifact dims
    vision_1: 2048
    prompt_emb: 2048
    robot_state: 32
  in_memory:
    preload_path: null             # path to .pkl artifact

# Episode-end write policy
write_policy:
  type: on_any_miss                # "on_any_miss" | "always" | "never"
```

### Cross-Validation Rules

| Rule | Description |
|------|-------------|
| Enabled keys ⊆ vector_dims | Field must be declared in backend |
| vector_dims keys ⊆ CACHE_QUERY_FIELDS | No unknown field names |
| Valid checkpoint names | Only `cp1`, `cp3` |
| key_builder type valid | Must be in supported set |
| placeholder only supports robot_state | Cross-check with enabled keys |
| cp1_* builders require vision_0 + robot_state | Cross-check |
| trajectory_depth >= 1 | 0 is invalid |
| trajectory_weights length == depth | When depth > 1 |
| trajectory_weights non-negative, sum > 0 | Sanity check |
| in_memory only for trajectory_depth > 1 | Qdrant rejected at config time |
| write_policy.type valid | on_any_miss / always / never |

### CLI Usage

```bash
# Serve with cache
uv run scripts/serve_policy.py --cache_config cache.yaml --env LIBERO

# Concurrent mode (supports dynamic config switching)
uv run scripts/serve_policy.py --concurrent --cache_config cache.yaml --env LIBERO
```

---

## 11. Orchestrator

**Source**: `src/openpi/cache/orchestrator.py`

### Constructor

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

### Episode Lifecycle

| Method | When | Effect |
|--------|------|--------|
| `on_task_begin(task_key)` | Connection opens | Reset step counter, episode buffers |
| `on_episode_start(task_key, episode_id)` | Episode starts | Clear buffers, broadcast to components |
| `broadcast_action(action)` | After inference | Propagate to strategies/gates/judges |
| `buffer_for_write(query_keys, action)` | After inference | Accumulate StepRecord |
| `on_episode_end()` | Episode ends | WritePolicy → build entry chain → batch_insert |
| `clear()` | End of each cycle | Release KeyBuilder cache |

### Miss Counting

`_miss_by_checkpoint` tracks misses per checkpoint during an episode. Used by `OnAnyMissWritePolicy` to decide writes.

---

## 12. Interceptor

**Source**: `src/openpi/cache/interceptor.py`

`InferenceInterceptor` wraps a `Policy` as a `BasePolicy` drop-in.

### Inference Flow

```
infer(obs)
  ├─ Input transforms
  ├─ Stage 1 (vision + tokenisation)
  ├─ CP1 check → if FULL_HIT: broadcast_action + buffer_for_write → return cached action
  ├─ Stage 2 (LLM backbone)
  ├─ Stage 3 (flow matching)
  ├─ CP3 check
  ├─ broadcast_action + buffer_for_write
  └─ Output transforms → return actions
```

### Episode Lifecycle

| Method | When | Effect |
|--------|------|--------|
| `on_task_begin()` | Connection opens | Reset timer + orchestrator |
| `on_episode_start(experiment, task, episode_id)` | Simulator sends episode_start | Forward to orchestrator |
| `on_episode_end(success)` | Simulator sends episode_end | Trigger orchestrator.on_episode_end(), reset timer |
| `on_task_end()` | Connection closes | Print timing summary |

---

## 13. Trajectory Search

### Concept

Single-step search matches the current observation against stored entries. Trajectory search additionally considers whether the **preceding steps** in the stored trajectory match the agent's recent history, favoring temporally coherent sequences.

### Algorithm (InMemoryBackend)

Two-pass recursive with batched per-level scoring:

```
Phase A — Collect:
  For each candidate entry, walk prev_ids backwards up to trajectory_depth.
  Collect entry sets per depth level.
  Checkpoint guard: skip entries with wrong checkpoint_id.

Phase B — Score:
  Batch-score each level using configured fusion method (RRF / score_sum).
  Query keys come from trajectory_history (newest-first).

Phase C — Aggregate:
  Walk same paths again, look up pre-computed scores.
  trajectory_score = Σ weights[i] × step_score[i]
  For branching paths (multiple prev_ids), take max.
```

### Configuration

```yaml
search_strategy:
  trajectory_depth: 3              # look back 3 steps
  trajectory_weights: [0.6, 0.3, 0.1]  # newest-first
```

### RRF vs Score Sum for Trajectory

- **RRF**: Rank-based. Top-1 score is always `Σ weights / (k + 1)` regardless of actual similarity. Trajectory fusion degrades to binary "has chain or not".
- **Score Sum**: Preserves similarity magnitude. Trajectory fusion reflects "this step matches well vs poorly". **Recommended for trajectory search.**

### Limitations

- Only InMemoryBackend supports trajectory_depth > 1
- History buffer resets on `on_episode_start()`
- First `trajectory_depth - 1` steps fall back to partial/single-step search

---

## 14. Episode Write Path

### Overview

Instead of per-step `write()`, the new path buffers all steps during an episode, then writes them as a linked chain at episode end.

### WritePolicy

| Type | Behavior |
|------|----------|
| `on_any_miss` | Write if episode had any CP1 cache miss (default) |
| `always` | Always write, regardless of hit/miss |
| `never` | Read-only mode, no writes |

### Entry Chain Structure

```
Episode: [step_0] → [step_1] → [step_2] → [step_3]

step_0: id="uuid:0", prev_ids=[], next_ids=["uuid:1"], trajectory_id="uuid"
step_1: id="uuid:1", prev_ids=["uuid:0"], next_ids=["uuid:2"], trajectory_id="uuid"
step_2: id="uuid:2", prev_ids=["uuid:1"], next_ids=["uuid:3"], trajectory_id="uuid"
step_3: id="uuid:3", prev_ids=["uuid:2"], next_ids=[], trajectory_id="uuid"
```

### Building Artifacts

```bash
# Build from HDF5 demo data — produces artifact with trajectory links
mkdir -p exp/common/data/cache_artifacts/libero_spatial

for bt in cp1_mean_pool cp1_spatial_pool_16 cp1_spatial_pool_4 cp1_max_pool; do
    uv run exp/common/build_in_memory_cache_artifact.py \
        --data-dir exp/common/data/libero_spatial \
        --builder-type $bt \
        --output exp/common/data/cache_artifacts/libero_spatial/${bt}.pkl
done
```

Artifacts are backward compatible — old experiments work with new artifacts (trajectory fields have defaults).

---

## 15. Component Isolation Rules

| Rule | Reason |
|------|--------|
| Components must **NOT** import `config.py` | Config is factory layer; components are pure logic |
| Gate / Judge must **NOT** call `CacheStorage` | Gate: predicate. Judge: evaluator. Neither does IO. |
| Only `SearchStrategy` calls `storage.search()` | Single query construction point |
| Only `Orchestrator` calls `storage.insert()` / `batch_insert()` | Write path centralized |

---

## 16. Testing Patterns

### Direct Construction (bypass YAML)

```python
orchestrator = CacheOrchestrator(
    storage=CacheStorage(InMemoryBackend({"robot_state": 32})),
    key_builder=PlaceholderKeyBuilder(),
    gates={CheckpointID.CP1: AlwaysSearchGate()},
    judges={CheckpointID.CP1: ThresholdJudge(cp1_threshold=0.95)},
    search_strategies={CheckpointID.CP1: my_strategy},
    write_policy=OnAnyMissWritePolicy(),
)
```

### Direct Entry Insertion (test helper)

```python
from tests.cache.conftest import insert_entry

# Insert directly into storage for test setup
entry = insert_entry(storage, CheckpointID.CP1, state_tensor, payload,
                     prev_ids=["prev_id"], trajectory_id="traj_001")
```

### Test Organization

| File | Coverage |
|------|----------|
| `test_orchestrator.py` | Check pipeline, query_keys return, broadcast, step counter |
| `test_trajectory_search.py` | Rerank, chain breakage, checkpoint guard, weighted score, branching |
| `test_episode_write.py` | WritePolicy, buffer, entry chain linking, episode lifecycle |
| `test_search_strategy.py` | Strategy basics + TrajectoryMixin history/fields |
| `test_config.py` | YAML loading, validation, trajectory/write_policy config |
| `test_cache_storage.py` | Dimension checks |
| `components/test_*.py` | Gate, Judge, KeyBuilder unit tests |

---

## 17. Current Validation Status

| Feature | Status | Details |
|---------|--------|---------|
| **CP1 check** | Validated | Full pipeline: Stage 1 → CP1 check → cache hit → skip stages 2–3 |
| **CP3 check** | Infrastructure only | Runs but no CP3-specific entries written; always MISS |
| **InMemory backend** | Primary | Multi-field fusion (RRF + score_sum), trajectory search, artifact loading |
| **Trajectory search** | Validated | Two-pass recursive, depth=3 tested on LIBERO data |
| **Episode write** | Implemented | buffer_for_write → on_episode_end → linked entry chain |
| **WritePolicy** | Implemented | on_any_miss / always / never |
| **Qdrant backend** | Deprecated | Not actively maintained; trajectory search not supported |
| **CP2** | Suspended | Not implemented |
