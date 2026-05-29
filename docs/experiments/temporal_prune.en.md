# Temporal Prune Experiment Tutorial

> **Prerequisites**:
> - [../cache/temporal_prune.md](../cache/temporal_prune.md) — KeyBuilder component and parameter reference.
> - [cp1_cache.md](cp1_cache.md) — generic CP1 experiment pipeline.

---

## 1. Experiment Overview

This experiment runs a grid search on the `cp1_temporal_prune` KeyBuilder to explore the effect of temporal-pruning parameters on cache retrieval quality.

### 1.1 Grid

| Dimension | Values | Notes |
|-----------|--------|-------|
| reducer | mean_pool, max_pool | Step 2 pooling strategy |
| prune_window_size | 3, 4, 5, 6 | Temporal window in frames |
| temporal_keep_ratio | 0.25, 0.5, 0.75 | Fraction of tokens kept |
| weights | WA, WB | Retrieval-fusion weights (see below) |

**Weight configurations**:

| ID | vision_0 | vision_1 | robot_state | Character |
|----|----------|----------|-------------|-----------|
| WA | 0.1 | 0.1 | 0.8 | robot_state-dominated |
| WB | 0.5 | 0.25 | 0.25 | vision-dominated |

### 1.2 Volume

- **Artifacts**: 2 × 4 × 3 = **24 .pkl files**.
- **YAMLs**: 24 × 2 = **48 files**.
- **Search strategy**: all use `weighted_rrf_knn`, `top_k=1`, `step_filter=all`, `trajectory_depth=1`.

---

## 2. Step 1: Build Artifacts

### 2.1 One-shot generation of build commands

```bash
uv run exp/temporal_prune/generate_temporal_prune_yamls.py \
    --print-artifact-commands \
    --data-dir exp/common/data/db/libero_cache/libero_spatial \
    --artifact-dir exp/common/data/cache_artifacts/libero_spatial/temporal_prune
```

This prints all 24 `build_in_memory_cache_artifact.py` commands.

### 2.2 Execute the builds

Save the output as a script and run:

```bash
uv run exp/temporal_prune/generate_temporal_prune_yamls.py \
    --print-artifact-commands \
    --data-dir exp/common/data/db/libero_cache/libero_spatial \
    --artifact-dir exp/common/data/cache_artifacts/libero_spatial/temporal_prune \
    > /tmp/build_tp_artifacts.sh

bash /tmp/build_tp_artifacts.sh 2>&1 | tee logs/build_tp_artifacts.log
```

> **Note**: use `--workers -1` (serial mode) to avoid ProcessPoolExecutor fork issues under WSL2. Building 24 artifacts serially takes a while.

### 2.3 Verify the artifacts

```bash
ls exp/common/data/cache_artifacts/libero_spatial/temporal_prune/*.pkl | wc -l
# expected: 24
```

Expected file naming:

```
cp1_tp_mean_3w_025kr.pkl   cp1_tp_max_3w_025kr.pkl
cp1_tp_mean_3w_05kr.pkl    cp1_tp_max_3w_05kr.pkl
cp1_tp_mean_3w_075kr.pkl   cp1_tp_max_3w_075kr.pkl
cp1_tp_mean_4w_025kr.pkl   cp1_tp_max_4w_025kr.pkl
...                        ...
cp1_tp_mean_6w_075kr.pkl   cp1_tp_max_6w_075kr.pkl
```

---

## 3. Step 2: Emit YAML Configs

```bash
uv run exp/temporal_prune/generate_temporal_prune_yamls.py \
    --artifact-dir exp/common/data/cache_artifacts/libero_spatial/temporal_prune \
    --output-dir exp/temporal_prune
```

Verify:

```bash
ls exp/temporal_prune/config/*.yaml | wc -l
# expected: 48
```

Example naming:

```
tp_run_001_mean_3w_025kr_wa.yaml   # mean_pool, window=3, kr=0.25, weights A
tp_run_002_mean_3w_025kr_wb.yaml   # mean_pool, window=3, kr=0.25, weights B
...
tp_run_048_max_6w_075kr_wb.yaml    # max_pool, window=6, kr=0.75, weights B
```

---

## 4. Step 3: Launch the GPU Server

On the GPU machine, launch the inference service:

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache_config cache.yaml \
    --env LIBERO \
    --port 8000 \
    --stage1_device cuda:0 \
    --stage2_device meta \
    --stage3_device meta \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

> `--concurrent` mode lets the cache config be swapped via WebSocket without restarting the server.

---

## 5. Step 4: Run the Experiment

On the eval host:

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/temporal_prune \
    --episodes-per-run 5 \
    --num-workers 5 \
    --host <GPU_HOST> --port <GPU_PORT> \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env libero_sim
```

### 5.1 Resume

The experiment supports resume. On interruption, re-run the same command; finished runs are skipped:

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/temporal_prune \
    --resume \
    ...  # other args same as above
```

Progress is saved at `exp/temporal_prune/data/experiment_state.json`.

---

## 6. Step 5: Analyze Results

```bash
uv run exp/common/analyze_cache_results.py \
    --state-file exp/temporal_prune/data/experiment_state.json \
    --output exp/temporal_prune/config/analysis.json
```

Analysis dimensions:
- Group by reducer: mean_pool vs max_pool — which is better?
- Group by keep_ratio: where is the sweet spot among 0.25 / 0.5 / 0.75?
- Group by window: is there a monotonic trend across 3 / 4 / 5 / 6?
- Group by weights: WA (robot_state-dominated) vs WB (vision-dominated).

---

## 7. File Map

| File | Purpose |
|------|---------|
| `exp/temporal_prune/generate_temporal_prune_yamls.py` | Emits 24 artifact-build commands + 48 YAML configs |
| `exp/common/build_in_memory_cache_artifact.py` | Builds .pkl artifacts (existing) |
| `exp/common/run_cache_experiments.py` | Executes experiments (existing) |
| `exp/common/analyze_cache_results.py` | Analyzes results (existing) |
| `exp/temporal_prune/config/` | YAML configs and experiment state |
| `exp/common/data/cache_artifacts/libero_spatial/temporal_prune/` | .pkl artifact files |
