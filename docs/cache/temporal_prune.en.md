# Temporal Prune KeyBuilder Guide

> **Prerequisite**: read [tutorial.md](tutorial.md) §4 for KeyBuilder component basics.
>
> **Design document**: full design at [`logs/archive/redundant_token_prune_plan.log.md`](../../logs/archive/redundant_token_prune_plan.log.md).

---

## 1. Overview

`CP1TemporalPruneKeyBuilder` is a two-stage KeyBuilder that inserts a **temporal-redundancy pruning** stage in front of the existing vision-token pooling:

```
raw vision tokens             Step 1: temporal prune          Step 2: token reduce
[256, 2048]  ──────►  [K, 2048] (K<256)  ──────►  [output_dim]
                       drop cross-frame static tokens          pluggable reducer
```

**Core idea**: of the 256 vision tokens produced by SigLIP, most are nearly invariant background tokens across time steps. By comparing cosine changes between adjacent frames, we identify and drop those redundant tokens so the final cache key focuses on motion-relevant visual regions.

**Use case**: CP1 checkpoint vision-token dimensionality reduction; usable both at online inference time and during offline artifact construction.

---

## 2. Quickstart

### 2.1 Build an offline artifact

```bash
uv run python exp/common/build_in_memory_cache_artifact.py \
    --data-dir exp/common/data/db/libero_cache/libero_spatial \
    --builder-type cp1_temporal_prune \
    --reducer-type mean_pool \
    --prune-window-size 4 \
    --temporal-keep-ratio 0.5 \
    --output exp/common/data/cache_artifacts/libero_spatial/cp1_temporal_prune_mean.pkl
```

### 2.2 Online inference YAML

```yaml
# cache_temporal_prune.yaml
enabled: true

key_builder:
  type: cp1_temporal_prune
  prune_window_size: 4          # temporal window size in frames (minimum 2)
  temporal_keep_ratio: 0.5      # keep top-50% most-changing tokens
  reducer:
    type: mean_pool             # Step 2 reduction strategy

keys:
  vision_0: { enabled: true, weight: 1.0 }
  vision_1: { enabled: true, weight: 1.0 }
  robot_state: { enabled: true, weight: 0.5 }
  prompt_emb: { enabled: true, weight: 0.3 }

backend:
  type: in_memory
  vector_dims:
    vision_0: 2048              # must match reducer output_dim
    vision_1: 2048
    prompt_emb: 2048
    robot_state: 32
  in_memory:
    preload_path: exp/common/data/cache_artifacts/libero_spatial/cp1_temporal_prune_mean.pkl

checkpoints:
  cp1:
    enabled: true               # cp1_temporal_prune requires CP1 enabled
    judge:
      type: threshold
      threshold: 0.98
    search_strategy:
      type: weighted_rrf_knn
      top_k: 1
  cp3:
    enabled: true
    judge:
      type: threshold
      threshold: 0.95
    search_strategy:
      type: weighted_rrf_knn
      top_k: 1
```

### 2.3 Launch the inference server

```bash
uv run python scripts/serve_policy.py \
    --env LIBERO \
    --cache_config cache_temporal_prune.yaml
```

---

## 3. Two-Stage Architecture

### 3.1 Step 1: Temporal Pruning

**Applies to**: vision modality only (vision_0 / vision_1 / vision_2), each camera processed independently. `prompt_emb` and `robot_state` bypass this step.

**Algorithm**:
1. Maintain a FIFO history window (W frames); push the current frame at each CP1 `collect`.
2. When the window is full, compute the temporal score = average cosine change between adjacent frames, per token position.
3. Keep the top `keep_ratio` tokens by temporal score.
4. Emit a `PruneResult` (containing the kept tokens, their original position indices, and their temporal scores).

**Degenerate mode**: while the window is not yet full (early episode), pruning is skipped and all 256 tokens flow straight to Step 2. Output dimensionality stays the same.

### 3.2 Step 2: Token Reduction (pluggable)

Step 2 is defined by the `TokenReducer` protocol and can be freely swapped:

| Reducer | Params | Output dim | Notes |
|---------|--------|------------|-------|
| `mean_pool` | — | 2048 | Average over time and tokens — simplest baseline |
| `max_pool` | — | 2048 | Average over time, then per-dim max over tokens |
| `spatial_pool` | `output_tokens` | output_tokens * 2048 | Fill back into a 16×16 grid → adaptive avg pool |
| `task_scoring` | `select_k`, `temperature` | 2048 | Pick top-K via cos(token, prompt_emb) then weighted pool |

---

## 4. Parameters

### 4.1 KeyBuilder parameters

| Param | Type | Default | Constraint | Notes |
|-------|------|---------|------------|-------|
| `prune_window_size` | int | 4 | >= 2 | Temporal window size in frames. Adjacent-frame diff needs at least 2 frames. |
| `temporal_keep_ratio` | float | 0.5 | (0, 1] | Fraction of tokens to keep. 0.5 = keep 128 out of 256. |

### 4.2 Reducer parameters

| Param | Applies to | Type | Default | Constraint | Notes |
|-------|------------|------|---------|------------|-------|
| `output_tokens` | spatial_pool | int | 16 | >= 1, perfect square | Output token count; determines pooled grid size. |
| `select_k` | task_scoring | int | 32 | >= 1 | Number of top-K tokens to pick by task score. |
| `temperature` | task_scoring | float | 1.0 | > 0 | Softmax temperature; smaller = sharper. |

### 4.3 `vector_dims` vs reducer correspondence

In the YAML, the vision field dimensionality under `backend.vector_dims` **must** match the reducer output dim, otherwise config validation fails:

| reducer.type | vision-field dim |
|--------------|------------------|
| `mean_pool` | 2048 |
| `max_pool` | 2048 |
| `task_scoring` | 2048 |
| `spatial_pool` | output_tokens * 2048 (e.g. output_tokens=16 → 32768) |

---

## 5. Reducer Selection Guide

| Scenario | Recommended reducer | Reason |
|----------|---------------------|--------|
| Initial baseline experiment | `mean_pool` | Simplest; directly comparable to the no-prune `cp1_mean_pool`. |
| Want to preserve spatial info | `spatial_pool` | Filling back into the grid preserves spatial relationships between tokens. |
| Mixed-task database | `task_scoring` | Uses prompt_emb to separate "dynamic and relevant" from "dynamic but irrelevant" tokens. |
| Single-task database | `mean_pool` or `max_pool` | prompt_emb has low discriminative power, so task scoring offers little. |

---

## 6. Offline Artifact Builder CLI Reference

```bash
uv run python exp/common/build_in_memory_cache_artifact.py \
    --data-dir <HDF5 data dir> \
    --builder-type cp1_temporal_prune \
    --output <output .pkl path> \
    --reducer-type <mean_pool|max_pool|spatial_pool|task_scoring> \
    --output-tokens <int>              # spatial_pool only, default 16 \
    --select-k <int>                   # task_scoring only, default 32 \
    --temperature <float>              # task_scoring only, default 1.0 \
    --prune-window-size <int>          # default 4 \
    --temporal-keep-ratio <float>      # default 0.5 \
    --workers <int>                    # 0=all CPUs, -1=serial mode
```

**Important**: the offline artifact and online inference must use **exactly the same reducer parameters**, or key semantics diverge. The artifact's metadata records a `reducer_params` field for manual cross-check.

### 6.1 Batch-build example

```bash
# Compare different reducers
for rt in mean_pool max_pool; do
    uv run python exp/common/build_in_memory_cache_artifact.py \
        --data-dir exp/common/data/db/libero_cache/libero_spatial \
        --builder-type cp1_temporal_prune \
        --reducer-type $rt \
        --output exp/common/data/cache_artifacts/libero_spatial/cp1_tp_${rt}.pkl
done

# Compare different keep_ratios
for kr in 0.25 0.5 0.75; do
    uv run python exp/common/build_in_memory_cache_artifact.py \
        --data-dir exp/common/data/db/libero_cache/libero_spatial \
        --builder-type cp1_temporal_prune \
        --reducer-type mean_pool \
        --temporal-keep-ratio $kr \
        --output exp/common/data/cache_artifacts/libero_spatial/cp1_tp_mean_kr${kr}.pkl
done
```

---

## 7. Lifecycle and Stateful Behavior

Unlike the existing stateless KeyBuilders (e.g. `cp1_mean_pool`), `CP1TemporalPruneKeyBuilder` is **stateful** and maintains a cross-step history buffer:

| Event | Behavior |
|-------|----------|
| `collect(CP1, ...)` | Push current frame's vision tokens into the history buffer (cloned). |
| `collect(CP3, ...)` | Does **not** push history (avoids CP1+CP3 same-frame double-count). |
| `build(CP1/CP3)` | Read from the history window → prune → reduce → CPU key. |
| `clear()` | Clears only the per-cycle cache; does **not** clear the history buffer. |
| `on_episode_start()` | Resets the history buffer (broadcast automatically by the Orchestrator). |

**Offline artifact build**: `_process_episode()` calls `on_episode_start()` at the start of each episode; the builder persists across steps, performing `collect → build → clear` per step so the history accumulates naturally.

---

## 8. Comparison with Existing KeyBuilders

| Feature | `cp1_mean_pool` etc. | `cp1_temporal_prune` |
|---------|----------------------|----------------------|
| Vision-token handling | All 256 tokens → pool directly | Prune first → then pool |
| Stateful? | No | Yes (history buffer) |
| Episode boundary | No handling needed | Requires `on_episode_start()` reset |
| Step 2 pluggable? | Fixed (mean/max/spatial) | Choice of 4 reducers |
| Offline artifact build | Each step independent | Cross-step continuous (builder instance persists) |

---

## 9. Module File Map

| File | Contents |
|------|----------|
| `src/openpi/cache/components/token_reducer.py` | PruneResult, TokenReducer Protocol, 4 reducer implementations |
| `src/openpi/cache/components/key_builder.py` | _VisionHistoryBuffer, CP1TemporalPruneKeyBuilder |
| `src/openpi/cache/config.py` | ReducerConfig, KeyBuilderConfig extensions, _build_reducer factory, validation rules |
| `src/openpi/cache/orchestrator.py` | on_episode_start broadcast to key_builder |
| `exp/common/build_in_memory_cache_artifact.py` | Offline artifact build (includes cp1_temporal_prune support) |
| `tests/cache/components/test_temporal_prune.py` | 46 test cases |

---

## 10. FAQ

### Q: Why is `prune_window_size` minimum 2?

Temporal scoring is computed as the cosine change **between adjacent frames**. With only 1 frame there are no adjacent frames and the diff cannot be computed — it would produce NaN.

### Q: Does key quality drop in the first few steps of an episode?

While the window is not yet full, pruning is skipped (`PruneResult.pruned=False`) and all 256 tokens go straight to the reducer. Output dimensionality matches the steady-state, but key contents are equivalent to the no-prune version. This is intentional: prefer degenerating to no-prune over computing incomplete temporal scoring.

### Q: How do I add a new reducer?

1. Implement the `TokenReducer` protocol in `token_reducer.py` (`reduce()` method and `output_dim` property).
2. Add a branch in the `_build_reducer()` factory in `config.py`.
3. Add parameter validation in `validate_cache_config()` in `config.py` (if there are new parameters).
4. Register it in the `_valid_reducer_types` set.
5. Add a branch in `_build_artifact_reducer()` in `exp/common/build_in_memory_cache_artifact.py`.
6. Write tests.

### Q: Will offline and online keys ever disagree?

As long as the same reducer parameters are used, the keys produced are deterministically equal. The artifact metadata records `reducer_params` for manual cross-check, and the config layer validates that reducer output dim matches `backend.vector_dims` — mismatch fails fast at startup.
