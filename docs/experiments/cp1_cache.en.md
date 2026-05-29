# CP1 Cache Experiment Runbook

> **⚙ New direction for client orchestration**: the multi-worker / multi-terminal client launches below (`--num-workers` / `run_cache_experiments.py` / multiple `main.py` processes) are the **legacy approach during the migration**. For new experiments, use the [experiment conductor framework](conductor_tutorial.md) — write an `ExperimentStrategy` + let the generic driver handle scheduling (episode-level no-gaps, cross-GPU / cross-host, resume, retry, monitoring); the legacy commands still work until the corresponding entrypoints have migrated.

> This document, based on the experiment design in `cache_cp1_impl_plan.log.md`, provides complete run instructions.

---

## Network Topology

```
┌─────────────────────────┐         frp tunnel         ┌────────────────────────┐
│  GPU server (no public IP) │ ◄─────────────────────── │  LIBERO eval host       │
│  serve_policy.py            │   155.98.36.32:9000       │  run_cache_experiments  │
│  listening on localhost:8000│   → localhost:8000        │  examples/libero/main   │
└─────────────────────────┘                           └────────────────────────┘
```

- **GPU server**: runs model inference, listens on `0.0.0.0:8000`.
- **Eval host**: runs LIBERO env + experiment controller, connects via frp at `155.98.36.32:9000`.
- Both ends need this repo's code and the `uv` environment.

---

## Prerequisites

1. Both ends have completed `GIT_LFS_SKIP_SMUDGE=1 uv sync`.
2. Pi0.5 checkpoint is available on the GPU server (default path `gs://openpi-assets/checkpoints/pi05_base`).
3. LIBERO benchmark data is available on the eval host.
4. Collected HDF5 demos are at `exp/common/data/db/libero_cache/libero_spatial/` (50 episodes).
5. frp tunnel is configured: `155.98.36.32:9000` → GPU server `localhost:8000`.

---

## Step 0: Verify the frp tunnel

On the eval host:

```bash
curl http://155.98.36.32:9000/healthz
# Expected: OK
```

If the server is not up, the connection fails. Start the Step 1 server first, then test.

---

## Step 1: Build cache artifacts (eval host or GPU server)

Convert HDF5 demos into `.pkl` vector indices under 4 different reductions.

```bash
# Run on a machine that has exp/common/data/db/libero_cache/libero_spatial/*.h5

mkdir -p exp/common/data/cache_artifacts/libero_spatial

# CP1 series (reduce from stage1 prefix_embs)
for bt in cp1_mean_pool cp1_spatial_pool_16 cp1_spatial_pool_4 cp1_max_pool; do
    uv run exp/common/build_in_memory_cache_artifact.py \
        --data-dir exp/common/data/db/libero_cache/libero_spatial \
        --builder-type $bt \
        --output exp/common/data/cache_artifacts/libero_spatial/${bt}.pkl
    echo "Done: $bt"
done

# CLIP ViT-B-32 (encode from raw images)
uv run exp/common/build_clip_cache_artifact.py \
    --data-dir exp/common/data/db/libero_cache/libero_spatial \
    --clip-model ViT-B-32 \
    --clip-pretrained openai \
    --output exp/common/data/cache_artifacts/libero_spatial/clip_vit_b_32.pkl \
    --device cuda \
    --batch-size 64 \
    --fields vision_0,vision_1,vision_2,prompt_emb,robot_state
```

Outputs:
```
exp/common/data/cache_artifacts/libero_spatial/
├── cp1_mean_pool.pkl          # A: mean pool → 2048d
├── cp1_spatial_pool_16.pkl    # B1: 4×4 spatial → 32768d
├── cp1_spatial_pool_64.pkl    # B2: 2×2 spatial → 8192d
├── cp1_max_pool.pkl           # C: max pool → 2048d
└── clip_vit_b_32.pkl          # D: CLIP ViT-B-32 → 512d
```

**Note**: these `.pkl` files must be accessible on the GPU server (since `serve_policy.py` loads them). If built on the eval host, scp them to the GPU server.

---

## Step 2: Calibrate Score-Sum statistics (eval host or GPU server)

Compute per-field p5/p95 percentile statistics for the `weighted_score_sum` fusion strategy (**legacy percentile route**).

> After the two-layer refactor, `weighted_score_sum`'s Layer-1 normalization uses `exp/common/calibrate_score_normalizers.py` (real query×whole-library distribution + multiple candidate methods); see the [weighted_sum runbook](weighted_sum.md). The percentile calibration below is only for loading legacy YAMLs.

```bash
uv run exp/common/calibrate_score_sum_stats.py \
    --artifact-dir exp/common/data/cache_artifacts/libero_spatial \
    --output exp/common/data/cache_artifacts/libero_spatial/calibration.json \
    --num-pairs 50000 \
    --seed 42
```

Output: `exp/common/data/cache_artifacts/libero_spatial/calibration.json`.

Check the separation warnings — if a field's same-task vs cross-task separation is < 0.05, that field has poor discriminative power.

---

## Step 3: Emit Phase 1 experiment YAML configs

A total of 10 combos (5 reductions × 2 fusions), but with `SKIP_SCORE_SUM = True` (line 66 of `generate_cache_run_yamls.py`), the script currently emits **5 combos × 8 weights = 40 YAMLs**. To restore the Score Sum series, set `SKIP_SCORE_SUM = False` to emit all 80.

```bash
uv run exp/common/generate_cache_run_yamls.py \
    --phase 1 \
    --artifact-dir exp/common/data/cache_artifacts/libero_spatial \
    --calibration-file exp/common/data/cache_artifacts/libero_spatial/calibration.json \
    --output-dir exp/common/config
```

> `--output-dir` is the parent; the script appends `phase1/` / `phase1_5/` / `phase2/` based on `--phase`. The command above writes YAMLs to `exp/common/config/phase1/`.

Outputs:
```
exp/common/config/phase1/
├── phase1_run_001_a_rrf_w1.yaml
├── phase1_run_002_a_rrf_w2.yaml
├── ...
└── phase1_run_040_d_rrf_w8.yaml   # D: CLIP ViT-B-32
                                    # 40 files total (SKIP_SCORE_SUM=True)
                                    # 80 files if SKIP_SCORE_SUM=False
```

**Important**: the generated YAMLs have `preload_path` pointing to absolute paths under `exp/common/data/cache_artifacts/libero_spatial/`. Make sure the paths match on the GPU server, or modify them manually after generation. If the two ends have different paths, generate the YAMLs on the GPU server or override `--artifact-dir` to match the GPU server's filesystem.

---

## Step 4: Launch the GPU server (GPU server side)

Launch the policy service on the GPU server in `--concurrent` mode so that dynamic config switching works.

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache_config cache.yaml \
    --env LIBERO \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

Key parameters:
- `--concurrent`: **must be enabled**. The experiment runner dynamically swaps the cache config via WebSocket control messages; only concurrent mode supports it.
- `--cache_config cache.yaml`: initial cache config (after the server starts, the experiment runner dynamically swaps in each run's YAML via the `load_cache_config` control message).
- `--env LIBERO`: LIBERO environment.
- `policy:checkpoint --policy.config pi05_libero --policy.dir ...`: points to a local Pi0.5 LIBERO checkpoint, avoiding GCS download.

Sanity-check the server:
```bash
curl http://localhost:8000/healthz
# Output: OK
```

---

## Step 5: Run Phase 1 experiments (eval host)

### 5a. Full run (current: 40 configs × 10 tasks × 5 episodes = 2000 episodes)

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/common/config/phase1 \
    --state-path exp/common/data/phase1/experiment_state.json \
    --episodes-per-run 5 \
    --num-workers 5 \
    --host 155.98.36.32 \
    --port 9000 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env libero_sim
```

Parameters:
- `--host 155.98.36.32 --port 9000`: connect to the GPU server through the frp tunnel.
- `--episodes-per-run 5`: 5 episodes per task (matching collection).
- `--num-workers 5`: 5 concurrent workers per task (requires the `--concurrent` server).
- `--task-suite libero_spatial`: the 10-task suite.
- `--seed 42`: fixed seed (collection used the default seed=7; eval uses a different seed to avoid overfitting).
- `--conda-env libero_sim`: run LIBERO eval inside the conda env (`main.py` needs LIBERO deps, which are not in the uv env).

### 5b. Run a subset of configs (debug)

```bash
# Run only configs 1–8 (all weights of the first combo)
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/common/config/phase1 \
    --state-path exp/common/data/phase1/experiment_state.json \
    --episodes-per-run 5 \
    --num-workers 5 \
    --host 155.98.36.32 \
    --port 9000 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env libero_sim \
    --runs 1-8
```

### 5c. Resume

The runner persists progress after each task completes. On interruption (Ctrl+C, crash, network drop), use `--resume` to continue from the last checkpoint:

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/common/config/phase1 \
    --state-path exp/common/data/phase1/experiment_state.json \
    --episodes-per-run 5 \
    --num-workers 5 \
    --host 155.98.36.32 \
    --port 9000 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env libero_sim \
    --resume
```

**Note**: `--resume` strictly validates that `--episodes-per-run` and `--task-suite` match the previous run; otherwise it errors out.

### 5d. Run status

Progress is saved at `exp/common/data/phase1/experiment_state.json`. Inspect directly:

```bash
# Summary view of progress
python3 -c "
import json
states = json.load(open('exp/common/data/phase1/experiment_state.json'))
done = sum(1 for s in states if s['status'] == 'done')
running = sum(1 for s in states if s['status'] == 'running')
failed = sum(1 for s in states if s['status'] == 'failed')
pending = sum(1 for s in states if s['status'] == 'pending')
print(f'Done: {done}, Running: {running}, Failed: {failed}, Pending: {pending}')
for s in states:
    if s['status'] == 'done':
        print(f'  {s[\"run_id\"]}: success_rate={s[\"success_rate\"]:.4f}')
"
```

Per-run detailed logs are at `exp/common/config/phase1/<run_id>.log`.

---

## Step 6: Analyze Phase 1 results

```bash
uv run exp/common/analyze_cache_results.py \
    --state-file exp/common/data/phase1/experiment_state.json \
    --output exp/common/data/phase1/analysis.json
```

Outputs:
- A ranking of all configs by success_rate.
- The best weight per combo.
- **Top 3 combos**: candidates for Phase 1.5.

Check the `top3` field in `exp/common/data/phase1/analysis.json` (Step 7's `--phase1-analysis` consumes the same file).

---

## Step 7: Emit Phase 1.5 configs

Fine-grained weight search around Phase 1's top 3 combos.

```bash
uv run exp/common/generate_cache_run_yamls.py \
    --phase 1.5 \
    --artifact-dir exp/common/data/cache_artifacts/libero_spatial \
    --calibration-file exp/common/data/cache_artifacts/libero_spatial/calibration.json \
    --phase1-analysis exp/common/data/phase1/analysis.json \
    --output-dir exp/common/config
```

Output: ~45 YAMLs under `exp/common/config/phase1_5/` (the `phase1_5/` subdirectory is appended by the script).

---

## Step 8: Run Phase 1.5 experiments

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/common/config/phase1_5 \
    --state-path exp/common/data/phase1_5/experiment_state.json \
    --episodes-per-run 5 \
    --num-workers 5 \
    --host 155.98.36.32 \
    --port 9000 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env libero_sim
```

Resume the same way with `--resume`.

---

## Step 9: Analyze Phase 1.5 results

```bash
uv run exp/common/analyze_cache_results.py \
    --state-file exp/common/data/phase1_5/experiment_state.json \
    --output exp/common/data/phase1_5/analysis.json
```

---

## Step 10: Emit Phase 2 configs (add prompt_emb)

```bash
uv run exp/common/generate_cache_run_yamls.py \
    --phase 2 \
    --artifact-dir exp/common/data/cache_artifacts/libero_spatial \
    --calibration-file exp/common/data/cache_artifacts/libero_spatial/calibration.json \
    --phase1-5-analysis exp/common/data/phase1_5/analysis.json \
    --output-dir exp/common/config
```

Output: ~3 YAMLs under `exp/common/config/phase2/` (prompt_emb weight 0.0 / 0.1 / 0.2; the `phase2/` subdirectory is appended by the script).

---

## Step 11: Run Phase 2 experiments

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/common/config/phase2 \
    --state-path exp/common/data/phase2/experiment_state.json \
    --episodes-per-run 5 \
    --num-workers 5 \
    --host 155.98.36.32 \
    --port 9000 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env libero_sim
```

---

## Step 12: Analyze Phase 2 final results

```bash
uv run exp/common/analyze_cache_results.py \
    --state-file exp/common/data/phase2/experiment_state.json \
    --output exp/common/data/phase2/analysis.json
```

The `best` field in `analysis.json` is the final optimal configuration.

---

## Full-pipeline time estimate

| Phase | Config count | Tasks | Episodes/Task | Total episodes | Estimated time |
|-------|--------------|-------|---------------|----------------|----------------|
| 1     | 40 (SKIP_SCORE_SUM) / 80 (full) | 10 | 5 | 2,000 / 4,000 | Depends on per-episode time |
| 1.5   | ~45 | 10 | 5 | ~2,250 | — |
| 2     | 3 | 10 | 5 | 150 | — |

Per-episode time depends on the LIBERO task's max_steps (libero_spatial: 220 steps) and inference latency.

---

## Troubleshooting

### Server connection failure

```bash
# Check the frp tunnel
curl http://155.98.36.32:9000/healthz

# Check the GPU server locally
curl http://localhost:8000/healthz
```

### load_cache_config errors

- Confirm the server was launched with `--concurrent`.
- Confirm the `preload_path` in the YAML exists on the GPU server.
- Check the error log in the GPU server's terminal.

### Recovery after interruption

```bash
# See which runs are unfinished
python3 -c "
import json
states = json.load(open('exp/common/data/phase1/experiment_state.json'))
for s in states:
    if s['status'] != 'done':
        remaining = sum(1 for v in s['task_progress'].values() if v != 'done')
        print(f'{s[\"run_id\"]}: status={s[\"status\"]}, remaining_tasks={remaining}')
"

# Continue running
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/common/config/phase1 \
    --state-path exp/common/data/phase1/experiment_state.json \
    --episodes-per-run 5 \
    --num-workers 5 \
    --host 155.98.36.32 \
    --port 9000 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env libero_sim \
    --resume
```

### Manual single-task test

Sanity-check the env by running one task directly, bypassing the runner:

```bash
MUJOCO_GL=egl conda run --no-capture-output -n libero_sim python examples/libero/main.py \
    --host 155.98.36.32 \
    --port 9000 \
    --task-suite-name libero_spatial \
    --num-trials-per-task 2 \
    --num-workers 1 \
    --task-ids 0
```

### Manual multi-worker test

Verify all workers work in concurrent mode:

```bash
MUJOCO_GL=egl conda run --no-capture-output -n libero_sim python examples/libero/main.py \
    --host 155.98.36.32 \
    --port 9000 \
    --task-suite-name libero_spatial \
    --num-trials-per-task 2 \
    --num-workers 5 \
    --task-ids 0 1 2 3 4 \
    --seed 42
```

> **Note**: `--num-workers` should not exceed the number of `--task-ids`; extra workers will exit immediately because the queue is empty (task dispatch is at task granularity, not episode granularity).

> **Note**: `main.py` depends on the LIBERO env and must be launched with `conda run -n libero_sim`, not `uv run`. The experiment runner `run_cache_experiments.py` itself is launched via `uv run` (it only needs `msgpack` / `websockets`), but internally invokes `main.py` via `conda` through the `--conda-env libero_sim` parameter.

---

## File-Dependency Overview

```
exp/common/data/db/libero_cache/libero_spatial/*.h5           ← raw HDF5 demos
    │
    ├──▶ build_in_memory_cache_artifact.py         (CP1 series: A/B1/B2/C)
    ├──▶ build_clip_cache_artifact.py              (CLIP series: D)
    ▼
exp/common/data/cache_artifacts/libero_spatial/*.pkl          ← vector-index artifacts
    │
    ├──▶ calibrate_score_sum_stats.py
    │       ▼
    │   calibration.json                           ← percentile statistics
    │       │
    ├───────┤
    ▼       ▼
generate_cache_run_yamls.py --phase 1
    ▼
exp/common/config/phase1/*.yaml                   ← 64 experiment configs
    │
    ├──▶ serve_policy.py --concurrent              (GPU server loads artifacts + YAML)
    │
    ▼
run_cache_experiments.py                           (eval host executes)
    ▼
experiment_state.json                              ← experiment progress + results
    │
    ▼ analyze_cache_results.py
exp/common/data/phase1/analysis.json               ← ranking + top 3
    │
    ▼ generate_cache_run_yamls.py --phase 1.5
exp/common/config/phase1_5/*.yaml → ... → exp/common/data/phase1_5/analysis.json
    │
    ▼ generate_cache_run_yamls.py --phase 2
exp/common/config/phase2/*.yaml → ... → exp/common/data/phase2/analysis.json (final result)
```
