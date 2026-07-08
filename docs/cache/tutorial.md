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

`gate(checkpoint_id, cached_data, request_context) -> bool` — `True` searches the cache, `False` skips (falls through to full inference).

| Type | Description |
|------|-------------|
| `always_search` | Always `True` (default). |
| `always_skip` | Always `False`. |
| `random` | Per-connection deterministic Bernoulli skip. Params: `p_inference`, `seed`. |
| `periodic` | `cache_len` searches then `inference_len` skips, repeating. Params: `cache_len`, `inference_len`. |
| `client_controlled` | Decision from the per-request `__gate_decision__` client signal (exp-layer N1/N2 prototyping). |
| `score_hysteresis` | Server-side N1 score-hysteresis gate, plus the optional N4 V2 injection. Params: `theta_low`, `theta_high`, `j`, `probe_interval` (optional; `None`/omitted = never probe), `L` (optional; `None`/omitted = pure N1 latency profile; `L: 6` = N4 SR profile that caps a continuous cache-execution run at `L` and injects a skip). |
| `follow_winner` | Server-side N2 lockstep blind-replay gate (Stage 4a). Locks a stable lockstep hit segment and blind-replays the winner episode's next cached action (no search/judge/inference) for `budget` steps. Params: `lock_streak`, `budget`. **Requires an `in_memory` backend.** Note: `lock_streak` counts adjacent lockstep *transitions*, so `lock_streak=N` needs `N+1` consecutive lockstep FULL_HITs. |

Gate lifecycle methods (all optional; the orchestrator guards each call):
- `on_episode_start(task_key="")` — reset per-episode state; receives the episode `task_key` (broadcast filtered by signature, so a no-arg override still works).
- `record_action(action)` — receive the broadcast action (trajectory-aware gates).
- `record_verdict(checkpoint_id, *, hit_type, cp1_score, winner_id, start_t, searched)` — receive this step's verdict after the judge runs, so a stateful gate can condition the next decision. Consumed by `score_hysteresis`: `cp1_score`/`searched` drive the N1 hysteresis, and `hit_type` (a `HitType` enum) drives the N4 V2 cache-execution run counter. `follow_winner` also consumes it (blind-replay steps arrive as `searched=False`). See [architecture/cache_system.md §5.5](../architecture/cache_system.md#55-gatefunction-pluggable).
- `replay_target() -> str | None` — Stage 4a N2 blind-replay hook: a locked `follow_winner` gate returns the winner cursor `entry_id` so the orchestrator replays the successor cached action (no search/judge/inference). Docstring-only optional hook (never a Protocol method).

**`score_hysteresis` YAML** — two deployment profiles (θ / j / probe_interval from Stage-1b calibration; `L=6` from the Stage-3a live verdict):

```yaml
# Latency profile (N1-A): omit L -> pure N1 hysteresis, no V2 injection.
checkpoints:
  cp1:
    gate:
      type: score_hysteresis
      theta_low: 0.968929    # stop searching once score drops below this
      theta_high: 0.975336   # a probe must reach this to recover
      j: 3                   # consecutive low-score searched steps before skipping
      probe_interval: 3      # probe every N steps while skipping (omit -> never probe)

# SR profile (N4): add L to cap a continuous cache-execution run and inject a skip.
checkpoints:
  cp1:
    gate:
      type: score_hysteresis
      theta_low: 0.968929
      theta_high: 0.975336
      j: 3
      probe_interval: 3
      L: 6                   # inject one skip after 6 continuous FULL_HIT replays
```

**`follow_winner` YAML** (Stage 4a N2) — must pair with an `in_memory` backend:

```yaml
checkpoints:
  cp1:
    gate:
      type: follow_winner
      lock_streak: 3         # lock after 3 lockstep transitions (== 4 consecutive lockstep FULL_HITs)
      budget: 5              # blind-replay up to 5 steps before unlocking
backend:
  type: in_memory           # required: blind replay walks the winner chain via walk_next/fetch_entry
  in_memory:
    preload_path: exp/common/data/cache_artifacts/<suite>/<keybuilder>.pkl
```

---

## 6. Component: Judge

> ⚠️ **2026-05-07 — Verdict Factor Judge refactor (G1 APPROVED Round 4)**:
> the `composite` judge has been rewritten as a 4-layer architecture
> (Normalization → Factor → Calibration → Composer) with a flat
> 17-factor naming scheme (`<descriptor>_<source>_<channel>` +
> `topk_action_variance`). The legacy 5 factor names +
> `cold_start_strategy` + `all_nan_fallback` yaml fields are removed.
> The `ThresholdJudge` / `AlwaysHitJudge` / `AlwaysWarmStartJudge` paths
> described below are unchanged. For `composite` configuration, see
> [`verdict_factor_judge.md`](verdict_factor_judge.md) (refactored
> 2026-05-07) and
> [`logs/verdict_factor_judge_refactor.log.md`](../../logs/archive/verdict_factor_judge_refactor.log.md)
> §6 / §11 / §13.

**Source**: `src/openpi/cache/components/judge.py`

Judge returns `JudgeResult(hit_type, winner_id, start_t)`.

| Type | Description |
|------|-------------|
| `threshold` | FULL_HIT if `score >= threshold`. With `warm_tiers`, scores below the threshold are matched against descending tiers for WARM_START (CP1 only). |
| `always_hit` | Always returns FULL_HIT for top result. Good for testing. |
| `always_warm_start` | Always emits WARM_START with a fixed `start_t` for the top result (CP1 only). Used to sweep success-rate vs `start_t` curves. |
| `composite` | 4-layer pluggable verdict pipeline (Normalization → Factor → Calibration → Composer). 17 registered factors: 4 descriptors (`jerk` / `direction` / `dispersion` / `path_length`) × 2 sources (`online` / `offline`) × 2 channels (`action` / `state`) + `topk_action_variance`. Calibration is per-key rolling-window percentile rank with no cold-start state (samples preloaded from offline file or per-yaml `WarmupPool`). Composer subclasses (weighted_sum / and / or / weighted_sum_with_warm_fallback) own NaN handling. Full lifecycle (build pkl → enrich-existing-pkl → warmup → eval → custom extension) lives in [verdict_factor_judge.md](verdict_factor_judge.md). |

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
| `weighted_score_sum_knn` | in_memory | Two-layer similarity fusion (Layer-1 normalize → Layer-2 weighted sum). Preserves magnitude. Requires `score_normalization`. |
| `qdrant_weighted_rrf_knn` | qdrant | Qdrant server-side RRF. Does NOT support trajectory search. |
| `dynamic_depth_knn` | in_memory | Per-step adaptive trajectory depth (TRACER Phase 1 / M3). Wraps a `base_fusion` (`weighted_rrf` / `weighted_score_sum`) and consults a `depth_policy` to choose the depth `T_t` each step. A `constant` policy at the full `trajectory_depth` is value-identical to the fixed-depth strategy; a `heuristic` policy buckets action smoothness. See the note below. |

`weighted_score_sum_knn` requires `score_normalization` (config rejects `type:none`). Two forms:

```yaml
search_strategy:
  type: weighted_score_sum_knn
  field_similarity: { vision_0: {type: cosine}, robot_state: {type: l2, to_similarity: {type: exp, tau: 0.33}} }
  score_normalization:
    type: per_field          # Layer-1 normalizers, params fit by calibrate_score_normalizers.py
    fields:
      vision_0:    { method: logit,  params: { lo: 0.9, hi: 0.999, eps: 1.0e-4 } }
      robot_state: { method: exp_l2, params: { tau: 0.33 } }
    # legacy form: type: percentile, fields: { vision_0: { p5: .., p95: .. } }
```

Method registry + design contract (monotone, bounded `[0,1]`, no rank equalization) live in `src/openpi/cache/components/score_normalizers.py` and [`cache_system.md` §5.8.1](../architecture/cache_system.md); the calibration + weight-search workflow is the [weighted_sum runbook](../experiments/weighted_sum.md).

### Dynamic chain depth (`dynamic_depth_knn`)

`dynamic_depth_knn` chooses the trajectory depth per step instead of fixing it (TRACER Phase 1 / M3). `trajectory_depth` / `trajectory_weights` become the **max** depth and the max-depth (newest-first) weight vector; `allowed_depths` (default `[trajectory_depth]`) is the set the policy may pick from; `base_fusion` selects the underlying fusion (a `weighted_score_sum` base still requires `score_normalization`). Requires an `in_memory` backend. Trajectory weights use **un-renormalized prefixes**, so a `constant` policy at the full depth reproduces `weighted_rrf_knn` / `weighted_score_sum_knn` exactly at every history length. Depth convention: depth `>= 1` (1 = single-step), so the proposal's `{0, 3, 5, 8}` maps to `{1, 3, 5, 8}` here.

```yaml
search_strategy:
  type: dynamic_depth_knn
  base_fusion: weighted_rrf        # or weighted_score_sum (+ score_normalization)
  trajectory_depth: 5              # max depth
  trajectory_weights: [0.4, 0.3, 0.15, 0.1, 0.05]   # newest-first, length == trajectory_depth
  allowed_depths: [1, 3, 5]        # subset of [1, trajectory_depth]
  depth_policy:
    type: heuristic                # or: { type: constant, depth: 5 }
    smoothness_thresholds: [0.5, 1.5]   # ascending, length == len(allowed_depths) - 1
    fallback_depth: 1              # depth when < 2 actions recorded (episode start)
```

The `heuristic` policy maps smaller action smoothness (steadier motion, `||a_{t-1} - a_{t-2}||` over the first action of each chunk) to a deeper context and larger to a shallower one; buckets are half-open `[t_i, t_{i+1})` with `>=` resolving ties. Learned depth policies are out of scope for Phase 1.

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
