# CP1 LLM Layer Extract Experiment Runbook

> **⚙ New direction for client orchestration**: the multi-worker / multi-terminal client launches below (`--num-workers` / `run_cache_experiments.py` / multiple `main.py` processes) are the **legacy approach during the migration**. For new experiments, use the [experiment conductor framework](conductor_tutorial.md) — write an `ExperimentStrategy` + let the generic driver handle scheduling (episode-level no-gaps, cross-GPU / cross-host, resume, retry, monitoring); the legacy commands still work until the corresponding entrypoints have migrated.

> **This document**: end-to-end experiment runbook — data collection → artifact build → YAML authoring → run → analyze.
>
> **Component API reference**: [`../cache/llm_layer_extract.md`](../cache/llm_layer_extract.md) (KeyBuilder class, reducer choices, YAML field semantics).
>
> **Design document**: [`../../logs/cp1_llm_layer_extract_key_builder_plan.log.md`](../../logs/archive/cp1_llm_layer_extract_key_builder_plan.log.md).
>
> **Artifact layout rules**: [`artifact_layout.md`](artifact_layout.md).

---

## 1. Network Topology

Same as other cache experiments — remote inference via frp tunnel:

```
┌───────────────────────────┐         frp tunnel          ┌───────────────────────────┐
│  GPU server (no public IP) │ ◄─────────────────────── │  LIBERO eval host           │
│  serve_policy.py             │   155.98.36.32:9000       │  run_cache_experiments.py    │
│  listening on 0.0.0.0:8000   │   → localhost:8000         │  examples/libero/main.py     │
└───────────────────────────┘                            └───────────────────────────┘
```

- **GPU server**: runs PI0Pytorch inference; ≥ 16 GB VRAM required.
- **Eval host**: runs the LIBERO benchmark + experiment controller.
- Both ends need this repo's code + `uv` env.
- Public entry is `155.98.36.32:9000` — **do not default to localhost**.

---

## 2. Prerequisites

| Resource | Check command / path |
|----------|----------------------|
| Pi0.5 checkpoint (with `model.safetensors`) | Default `$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch` |
| `uv sync` done (both ends) | `GIT_LFS_SKIP_SMUDGE=1 uv sync` |
| LIBERO benchmark data (eval host) | See [`../deployment/libero.md`](../deployment/libero.md) |
| frp tunnel working | `curl http://155.98.36.32:9000/healthz` |

> **GPU VRAM**: the server needs 5–10 GB for the model; artifact build needs an extra ~3 GB for layer-N forward activations; during inference, KeyBuilder per-step layer forward is ~1–2 ms (A100/4090 bf16).

---

## 3. End-to-End Pipeline Overview

```
┌─────────────────┐   Step 1     ┌─────────────────┐   Step 2     ┌─────────────────┐
│  Collect HDF5   │ ──────────► │ Build .pkl       │ ──────────► │ Artifact files  │
│ (serve --collect│             │ artifact (cp1_  │             │ /cache_artifacts │
│  + LIBERO env)   │             │  llm_layer_*)   │             │  /<task>/*.pkl  │
└─────────────────┘             └─────────────────┘             └─────────────────┘
                                          │
                                          ▼
┌─────────────────┐   Step 4     ┌─────────────────┐   Step 3     ┌─────────────────┐
│  Analyze        │ ◄────────── │ Run experiment   │ ◄────────── │ Write YAML config│
│ (analyze_cache_ │             │ (run_cache_     │             │ (cp1_llm_*.yaml)│
│  results.py)     │             │  experiments.py │             │                 │
└─────────────────┘             └─────────────────┘             └─────────────────┘
   Step 5
```

---

## Step 1 — Data Collection (skip if HDF5 already exists)

If you already have `exp/common/data/db/libero_cache/<task_suite>/*.h5` (shared with other builders like cp1_mean_pool), **skip to Step 2**. Otherwise, collect as follows.

### 1.1 GPU server: launch serve_policy with `--collect`

```bash
uv run scripts/serve_policy.py \
    --collect \
    --collect_dir exp/common/data/db/libero_cache \
    --env LIBERO \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

**Key flags**:
- `--collect`: enables HDF5 dumping (one file per episode).
- `--collect_dir`: write root; the final path is `<root>/<experiment_name>/episode_*.h5`.

**Collection behavior**: a forward hook captures, at each inference:
- `vision_0/1/2`: SigLIP 256 × 2048 tokens (per camera).
- `prompt_emb`: lang tokens 200 × 2048 (already padded).
- `robot_state`: 32-d.
- `clean_action / noise_action_*`: action-expert intermediate results.
- `attrs.task / attrs.success`: episode metadata.

Data collection details: [`../data_collection/guide.md`](../data_collection/guide.md).

### 1.2 Eval host: run LIBERO tasks to collect episodes

See [`cp1_cache.md`](cp1_cache.md) §3 for concrete commands; typical use is driving `examples/libero/main.py` directly or batch-driving with `run_cache_experiments.py`. 50 successful episodes are enough for a baseline experiment.

### 1.3 Verify HDF5 schema

```bash
uv run python -c "
import h5py, os, sys
d = sys.argv[1]
fp = os.path.join(d, sorted(os.listdir(d))[0])
f = h5py.File(fp, 'r')
print('attrs:', dict(f.attrs))
g = f[sorted(k for k in f.keys() if k.startswith('step_'))[0]]
for k in g.keys():
    o = g[k]
    print(' ', k, getattr(o, 'shape', '(group)'),
          getattr(o, 'dtype', ''))
" exp/common/data/db/libero_cache/libero_spatial
```

Expected output (matches G1 plan §6.2 validation):
```
attrs: {'episode_id': 0, 'experiment_name': 'libero_spatial',
         'num_steps': 16, 'success': True, 'task': '...', 'timestamp': '...'}
  clean_action  (10, 32)  float32
  prompt_emb    (200, 2048) float16    ← already padded to max_token_len=200
  robot_state   (32,)     float32
  vision_0      (256, 2048) float16
  vision_1      (256, 2048) float16
  vision_2      (256, 2048) float16
  ...
```

> **Important**: HDF5 does not store `lang_masks`. The Step 2 builder reconstructs `lang_masks` deterministically by re-tokenizing `attrs['task']` + `robot_state` with `PaligemmaTokenizer`. So nothing changes on the collection side.

---

## Step 2 — Build Cache Artifact

### 2.1 Command template

```bash
mkdir -p exp/common/data/cache_artifacts/<task_suite>

uv run python exp/common/build_in_memory_cache_artifact.py \
    --data-dir exp/common/data/db/libero_cache/<task_suite> \
    --builder-type cp1_llm_layer_extract \
    --extract-layer 0 \
    --prefix-reducer-type prefix_mean_pool \
    --checkpoint-dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch" \
    --config-name pi05_libero \
    --device cuda \
    --output exp/common/data/cache_artifacts/<task_suite>/cp1_llm_l0_meanpool.pkl
```

### 2.2 Key parameters

| Parameter | Values | Notes |
|-----------|--------|-------|
| `--builder-type` | `cp1_llm_layer_extract` | Selects this builder (distinct from SigLIP-only builders like cp1_mean_pool). |
| `--extract-layer` | `0..17` | gemma_2b layer index; pick 0 for the v1 baseline; sweep multiple yamls. |
| `--prefix-reducer-type` | `prefix_mean_pool` / `per_modality_mean_pool` / `per_modality_max_pool` / `per_modality_spatial_pool_16` / `per_modality_spatial_pool_4` | A = single-key original spec; B = four modality-independent keys (mean / max / 16-token 4×4 spatial / 4-token 2×2 spatial). The two spatial tiers align with legacy `cp1_spatial_pool_{16,4}`. See [`../cache/llm_layer_extract.md §3.2`](../cache/llm_layer_extract.md). |
| `--checkpoint-dir` | PI0Pytorch weights dir | Must be the **same checkpoint** used at HDF5 collection time, otherwise self-check fails. |
| `--config-name` | `pi05_libero` (or similar) | Used to load TrainConfig. |
| `--device` | `cuda` (default) | CPU mode is for smoke tests only. |
| `--workers` | Forced to `-1` automatically | The model takes 5–10 GB VRAM; cannot parallelize with ProcessPool. |

### 2.3 Startup self-check (automatic)

At build start, a tokenizer self-check runs: re-tokenize the first step of the first episode → `embed_language_tokens × √2048` → `allclose(rtol=1e-2, atol=1e-2)` against the HDF5 `prompt_emb` at mask=True positions.

- ✅ Pass → log `Tokenizer self-check passed (max abs diff: ...)`, build continues.
- ❌ Fail → **abort immediately** with `Tokenizer/embed self-check failed (max abs diff: ...)`, hint to verify checkpoint / tokenizer source.

Implementation: `exp/common/build_in_memory_cache_artifact.py::_self_check_tokenizer_consistency`.

### 2.4 Batch build (layer sweep + reducer comparison)

```bash
TASK=libero_spatial
CKPT="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
ART_DIR=exp/common/data/cache_artifacts/$TASK
mkdir -p $ART_DIR

# Layer sweep × reducer matrix
for L in 0 2 5; do
  for R in prefix_mean_pool per_modality_mean_pool; do
    uv run python exp/common/build_in_memory_cache_artifact.py \
        --data-dir exp/common/data/db/libero_cache/$TASK \
        --builder-type cp1_llm_layer_extract \
        --extract-layer $L \
        --prefix-reducer-type $R \
        --checkpoint-dir "$CKPT" \
        --config-name pi05_libero \
        --device cuda \
        --output $ART_DIR/cp1_llm_l${L}_${R}.pkl
    echo "Done: layer=$L reducer=$R"
  done
done
```

Building 50 episodes (typically ~5000 steps) per combo takes ≈ 5–8 min (model load ~30s + ~1–2 ms per step). Larger layer = slower (linearly).

### 2.5 Inspect artifact metadata

```bash
uv run python -c "
import pickle, sys
with open(sys.argv[1], 'rb') as f:
    art = pickle.load(f)
print('builder:', art['key_builder_type'])
print('vector_dims:', art['vector_dims'])
print('entries:', len(art['entries']))
print('reducer_params:', art.get('reducer_params'))
" exp/common/data/cache_artifacts/libero_spatial/cp1_llm_l0_prefix_mean_pool.pkl
```

Expected `reducer_params` includes `extract_layer / prefix_reducer_type / apply_final_norm / checkpoint_dir / config_name / tokenizer_class / tokenizer_source / tokenizer_max_len`. Provenance is used for future online/offline drift triage.

---

## Step 3 — Write YAML Config

### 3.1 Template A — `prefix_mean_pool` (original-spec baseline, single field)

```yaml
# exp/<your_exp>/config/cp1_llm_l0_meanpool.yaml
enabled: true

timer:
  enabled: true
  buffer_size: 10000

key_builder:
  type: cp1_llm_layer_extract
  extract_layer: 0                # must match the artifact
  prefix_reducer:
    type: prefix_mean_pool

keys:
  vision_0:    { enabled: true,  weight: 1.0 }    # the only allowed vision field
  vision_1:    { enabled: false, weight: 1.0 }
  vision_2:    { enabled: false, weight: 1.0 }
  prompt_emb:  { enabled: false, weight: 1.0 }    # must be false
  robot_state: { enabled: true,  weight: 0.5 }    # optional

backend:
  type: in_memory
  vector_dims:
    vision_0: 2048                # gemma_2b width, must = 2048
    robot_state: 32
  in_memory:
    preload_path: exp/common/data/cache_artifacts/libero_spatial/cp1_llm_l0_prefix_mean_pool.pkl

checkpoints:
  cp1:
    enabled: true                 # cp1_llm_layer_extract requires cp1 enabled
    gate:
      type: always_search
    judge:
      type: threshold
      threshold: 0.98
    search_strategy:
      type: weighted_rrf_knn
      top_k: 1
      step_filter: all
      rrf_k: 60
  cp3:
    enabled: true
    gate:
      type: always_search
    judge:
      type: threshold
      threshold: 0.95
    search_strategy:
      type: weighted_rrf_knn
      top_k: 1
      step_filter: all

write_policy:
  type: never                     # read-only during the experiment; do not let this inference pollute the artifact
```

### 3.2 Template B — `per_modality_mean_pool` (preserves modality ablation)

Differences only in `key_builder` / `keys` / `backend.vector_dims` / `preload_path`:

```yaml
key_builder:
  type: cp1_llm_layer_extract
  extract_layer: 0
  prefix_reducer:
    type: per_modality_mean_pool

keys:
  vision_0:    { enabled: true,  weight: 1.0 }
  vision_1:    { enabled: true,  weight: 1.0 }
  vision_2:    { enabled: true,  weight: 1.0 }
  prompt_emb:  { enabled: true,  weight: 0.5 }
  robot_state: { enabled: true,  weight: 0.5 }

backend:
  vector_dims:
    vision_0:    2048
    vision_1:    2048
    vision_2:    2048
    prompt_emb:  2048
    robot_state: 32
  in_memory:
    preload_path: exp/common/data/cache_artifacts/libero_spatial/cp1_llm_l0_per_modality_mean_pool.pkl
```

### 3.3 Config validation rules (common startup failures)

`validate_cache_config()` runs these cross-checks before startup (see `src/openpi/cache/config.py:546+`):

| Check | Error symptom | Fix |
|-------|---------------|-----|
| `extract_layer` ∈ [0, 17] | `extract_layer=18 out of range` | Change layer. |
| `prefix_reducer.type` valid | `prefix_reducer.type 'X' unknown` | Pick one of the five. |
| `prefix_mean_pool` + vision_1/2/prompt_emb enabled | `... would never be populated` | Disable those fields. |
| `vector_dims.<f>` does not match reducer output dim | `does not match prefix_reducer output dim N` | Match the reducer: mean/max → 2048; spatial_16 vision → 32768; spatial_4 vision → 8192; spatial_* prompt → 2048. |
| `cp1.enabled = true` | `requires checkpoints.cp1.enabled=true` | Enable cp1. |
| `in_memory.preload_path` missing | `requires backend.in_memory.preload_path` | Point it to the Step 2 output. |

---

## Step 4 — Run the Experiment

### 4.1 GPU server: launch serve_policy with the cache config

```bash
uv run scripts/serve_policy.py \
    --cache_config exp/<your_exp>/config/cp1_llm_l0_meanpool.yaml \
    --env LIBERO \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

Expected startup log:
```
CP1LLMLayerExtractKeyBuilder: attached to model (depth=18, extract_layer=0)
Cache config loaded: backend=in_memory, key_builder=cp1_llm_layer_extract, ...
```

> If you need concurrent multi-connection runs with multiple cache configs, switch to `--concurrent` mode. See [`cp1_cache.md`](cp1_cache.md) §5.

### 4.2 Eval host: drive the cache experiment in batch

If you only run one yaml, use `examples/libero/main.py` directly. If you run multiple configs from a yaml directory (typical layer × reducer sweep), use `run_cache_experiments.py`:

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/<your_exp>/config \
    --episodes-per-run 10 \
    --num-workers 5 \
    --host 155.98.36.32 --port 9000 \
    --task-suite libero_spatial \
    --seed 42 \
    --conda-env libero_sim \
    --log-dir exp/<your_exp>/data \
    --state-path exp/<your_exp>/data/experiment_state.json \
    --resume                       # if a previous run was interrupted
```

**Key flags** (see `exp/common/run_cache_experiments.py::_build_arg_parser` for details):
- `--yaml-dir`: each `.yaml` is one experiment config.
- `--episodes-per-run`: episodes per task.
- `--num-workers`: parallel LIBERO instances (one GPU process per worker).
- `--task-ids`: run only the listed tasks (default: whole suite).
- `--resume`: continue if the state file exists (per-task granularity).
- `--state-path` / `--log-dir`: **explicitly point to `data/`** to avoid polluting `config/`.

### 4.3 Single-YAML smoke check

```bash
uv run examples/libero/main.py \
    --host 155.98.36.32 --port 9000 \
    --task-suite-name libero_spatial \
    --num-trials-per-task 1 \
    --task-ids 0 \
    --num-workers 1 \
    --seed 42 \
    --cuda-visible-devices 0
```

---

## Step 5 — Result Analysis

```bash
uv run exp/common/analyze_cache_results.py \
    --state-file exp/<your_exp>/data/experiment_state.json \
    --output exp/<your_exp>/analysis/cp1_llm_l0_meanpool_summary.json
```

The output JSON contains per-task / per-config success rate and cache hit rate.

If you ran a layer × reducer comparison, consider writing a small plot script in `analysis/` (modeled on `exp/common/analysis/<other-experiment>/plot_results.py`) to render the figure from the summary JSON.

---

## 6. Verify: Online / Offline Parity (strongly recommended after every new artifact)

Does the new artifact actually produce the same query keys as online inference? Run the manual parity test:

```bash
PI05_CHECKPOINT_DIR="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch" \
PI05_CONFIG_NAME=pi05_libero \
uv run pytest tests/cache/test_llm_layer_extract_parity.py -m manual -v
```

The test asserts (details in [`../cache/llm_layer_extract.md`](../cache/llm_layer_extract.md) §6):
1. Online and offline `prefix_pad_masks` are exactly equal.
2. Online and offline layer-N hidden states `allclose(rtol=1e-2, atol=1e-2)` at mask=True positions.
3. Online and offline `KeyBuilder.build()` outputs `allclose`.
4. KeyBuilder single-layer replay equals HF `output_hidden_states[1]`.
5. Real model + InMemoryBackend's `collect→build→search→fetch` chain hits the correct entry for two distinct obs.

Any failure indicates the offline contract is broken (most common: `--checkpoint-dir` does not match collection time). **The verify stage must pass locally before you trust the experiment results.**

---

## 7. FAQ

### Q: What if the startup self-check fails?

Error like `Tokenizer/embed self-check failed (max abs diff: 12.34)`. Most common causes:

1. `--checkpoint-dir` differs from the checkpoint used at HDF5 collection.
2. The PaligemmaTokenizer download source has been replaced or the local cache is corrupted.
3. The HDF5 was collected with a different model or older tokenizer version.

Triage path: first cross-check the artifact's `reducer_params` metadata (`tokenizer_source` / `checkpoint_dir`) against the collection-time setup; then check the local cache of `gs://big_vision/paligemma_tokenizer.model`.

### Q: VRAM is insufficient — what should I do?

- Artifact build: default cuda; if GPU is tight, free up other processes, or use `--device cpu` (**not recommended**, seconds per step).
- Inference server: Pi0.5 + cache system ≈ 6–8 GB; on OOM, lower `--num-workers` or switch to a larger GPU.

### Q: Which `extract_layer` should I pick?

Start at `0` for v1 (cost = 5.5% of Stage 2). If baseline hit quality is insufficient:
- `2`: ~16.7% of Stage 2, 3 cross-modal attention rounds.
- `5`: ~33.3% of Stage 2, 6 rounds (the ROI knee).
- Not recommended ≥ 8 (past the ROI knee; reviving CP2 is preferable).

### Q: Want to run a cross-task experiment?

The motivation behind `prefix_mean_pool` is exactly multi-task / cross-task generalization. Suggested workflow:
1. Collect HDF5 with mixed multi-task episodes.
2. Build a unified artifact.
3. During eval, swap in different task suites and observe hit rate / SR transfer.

### Q: How much does online inference slow down?

0.5–2 ms per step on A100/4090 bf16 (layer 0). Pi0.5's total inference is ~50 ms, so a ~2–4% relative increase — acceptable.

### Q: Can I reuse `cp1_temporal_prune`'s reducer?

No. The two reducer protocols take different inputs; forcing reuse would break padding semantics. See [`../cache/llm_layer_extract.md`](../cache/llm_layer_extract.md) §9.

### Q: Can artifact build be parallelized?

No. `--workers` is forced to `-1` (serial). Each worker needs to load the 5–10 GB model and VRAM can't fit. If you have multiple GPUs, **manually parallelize different task-suite builds** (one `build_in_memory_cache_artifact` process per GPU, with `CUDA_VISIBLE_DEVICES`).

---

## 8. Relationship to Other Cache Experiments

| Experiment | Key differences |
|------------|-----------------|
| [cp1_cache.md](cp1_cache.md) | SigLIP token pool (no LLM forward), 4 reducers. **Fastest but weakest key representation.** |
| [temporal_prune.md](temporal_prune.md) | SigLIP + cross-step token pruning. **Stateful.** |
| **This experiment** | LLM layer-N hidden state (**with cross-modal fusion**), 2 reducers. **A bit slower, strongest key representation.** |
| [warm_start_sweep.md](warm_start_sweep.md) | warm-start hit threshold sweep. Can be combined with this experiment's artifacts. |

---

## 9. Reference File List

| File | Purpose |
|------|---------|
| [`../cache/llm_layer_extract.md`](../cache/llm_layer_extract.md) | Component API / YAML field semantics |
| [`../../logs/cp1_llm_layer_extract_key_builder_plan.log.md`](../../logs/archive/cp1_llm_layer_extract_key_builder_plan.log.md) | Design document (with G1/G2 review history) |
| `exp/common/build_in_memory_cache_artifact.py` | Step 2 builder entrypoint |
| `exp/common/run_cache_experiments.py` | Step 4 experiment driver |
| `exp/common/analyze_cache_results.py` | Step 5 result aggregation |
| `tests/cache/test_llm_layer_extract_parity.py` | Verify-stage parity test |
| [`artifact_layout.md`](artifact_layout.md) | `exp/<exp>/{config,data,analysis}/` layout rules |
| [`../data_collection/guide.md`](../data_collection/guide.md) | Step 1 data collection mechanism |
