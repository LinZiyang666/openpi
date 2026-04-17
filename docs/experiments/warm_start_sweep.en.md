# Warm Start Success-Rate Sweep — Runbook

> Derived from [`logs/warm_start_sweep_plan.log.md`](../../logs/warm_start_sweep_plan.log.md). Companions: `exp/warm_start/config/`, `src/openpi/cache/components/judge.py::AlwaysWarmStartJudge`, `exp/common/build_clip_cache_artifact.py`. Chinese version: [warm_start_sweep.md](warm_start_sweep.md).

---

## Overview

Under a forced-cache-hit regime (`gate: always_search` + `judge: always_warm_start`), sweep `start_t ∈ {0.7, 0.5, 0.3}` to test whether warm-starting the flow-matching denoiser from a cached intermediate `x_t` can pull success rate back from the always-hit baseline toward the pure-inference ceiling.

- 3 keybuilders × 3 start_t = **9 YAML × 500 ep = 4500 episodes** (or 6 × 500 = 3000 if CLIP is deferred).
- **Baselines are reused, not rerun**: B0 (inference ceiling) and B1 (always-hit floor) come straight from [trajectory_deviation](trajectory_deviation.en.md) Step 1a (`exp/trajectory_deviation/data/cache_eval_results_*.json`).
- Primary outputs: `exp/warm_start/data/results/cache_eval_results.json` (9 configs merged) + `exp/warm_start/data/timing/<cfg>/*.csv` latency probes.

Pipeline:

```
Step 0   Start GPU server(s), each pinned to one keybuilder
Step 1   Rebuild 3 warm artifacts (max_pool / spatial16 / clip → libero_spatial_warm/*.pkl)
Step 2   Smoke pass (one YAML, one task, 5 ep — verifies WARM_START path)
Step 3   Full run (3 servers in parallel, each serving 3 start_t YAMLs)
Step 4   Analyze + plot (exp/warm_start/analyze_warm_sweep.py)
```

## Network topology

Identical to [trajectory_deviation.en.md](trajectory_deviation.en.md): the eval host runs `exp/common/run_cache_experiments.py` and reaches the GPU server through frp. The server starts with `--concurrent + --cache_config`; the driver sends a `load_cache_config` control message before iterating each YAML to swap the active bundle.

---

## Prerequisites

1. Both hosts have run `GIT_LFS_SKIP_SMUDGE=1 uv sync`; the eval host has the `libero_sim` conda env.
2. `exp/common/data/db/libero_cache/libero_spatial/*.h5` (50 episodes) is present — the source for Step 1.
3. **Trajectory Deviation Step 1a has already run** and `exp/trajectory_deviation/data/cache_eval_results_{clip,maxpool,spatial16}.json` exist on disk. Warm start reuses them as the comparison set.
4. `exp/warm_start/data/baseline_failures.json` exists (split those three JSONs by `config_id` into `{cfg: {inference: {fails: [...]}, always_hit: {...}}}`).
5. Pi0.5 LIBERO checkpoint at `$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch`.

---

## Step 0: Start the GPU server

One server serves one keybuilder's three start_t YAMLs. Plan A uses three servers (ports 8000/7999/7998, frp 9000/8999/8998). Start each server with its keybuilder's **AlwaysSkip inference bundle** as the initial `--cache_config`; the driver swaps in the warm bundle automatically on the first run.

Start each server in its own terminal, pinned to one GPU. Port mapping:

| Slot | Initial `--cache_config` | GPU | Server `--port` | frp external port |
|------|--------------------------|-----|-----------------|-------------------|
| 1 | `deviate_exp/inference_max_pool_w3_d5.yaml` | 0 | 8000 | 9000 |
| 2 | `deviate_exp/inference_spatial16_w8_d4.yaml` | 1 | 7999 | 8999 |
| 3 | `deviate_exp/inference_clip_w7_d4.yaml` | 2 | 7998 | 8998 |

Slot 1 (max_pool, GPU 0, port 8000/9000):

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
    --concurrent \
    --cache_config exp/trajectory_deviation/config/inference_max_pool_w3_d5.yaml \
    --env LIBERO \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

Slot 2 (spatial16, GPU 1, port 7999/8999):

```bash
CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py \
    --concurrent \
    --cache_config exp/trajectory_deviation/config/inference_spatial16_w8_d4.yaml \
    --env LIBERO \
    --port 7999 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

Slot 3 (clip, GPU 2, port 7998/8998):

```bash
CUDA_VISIBLE_DEVICES=2 uv run scripts/serve_policy.py \
    --concurrent \
    --cache_config exp/trajectory_deviation/config/inference_clip_w7_d4.yaml \
    --env LIBERO \
    --port 7998 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

> No `--collect-images`: for the CLIP slot, `serve_policy.py` auto-promotes `need_images` at startup via `need_images = ... or key_builder.type == "clip"`; for max_pool / spatial16 slots the KeyBuilder does not read raw images, and keeping the flag on only adds a wasted `extract_valid_images` copy per step. This relies on each server being pinned to one keybuilder — no cross-type `load_cache_config` swaps — which the warm_start_sweep topology guarantees.

If a single GPU cannot host all three servers, fall back to Plan B: run only two (max_pool + spatial16), defer CLIP.

Health probe (frp external):

```bash
curl http://155.98.36.13:9000/healthz   # max_pool
curl http://155.98.36.13:8999/healthz   # spatial16
curl http://155.98.36.13:8998/healthz   # clip
```

All three must return `OK` before Step 1.

---

## Step 1: Build warm artifacts

The older `exp/common/data/cache_artifacts/libero_spatial/*.pkl` (built 2026-04-09) did not write `payload.intermediates`, so every WARM_START would silently downgrade to MISS. **Rebuild to `libero_spatial_warm/`** — do not overwrite the path trajectory_deviation depends on.

### 1.1 max_pool + spatial16 (CPU, 20 workers)

```bash
mkdir -p exp/common/data/cache_artifacts/libero_spatial_warm

uv run python exp/common/build_in_memory_cache_artifact.py \
    --data-dir exp/common/data/db/libero_cache/libero_spatial \
    --builder-type cp1_max_pool \
    --output exp/common/data/cache_artifacts/libero_spatial_warm/cp1_max_pool.pkl

uv run python exp/common/build_in_memory_cache_artifact.py \
    --data-dir exp/common/data/db/libero_cache/libero_spatial \
    --builder-type cp1_spatial_pool_16 \
    --output exp/common/data/cache_artifacts/libero_spatial_warm/cp1_spatial_pool_16.pkl \
    --reducer-type spatial_pool --output-tokens 16
```

50 episodes × 1018 steps, ~1 minute each on CPU. Outputs ~49 MB (max_pool) / ~407 MB (spatial16).

### 1.2 CLIP (GPU or CPU; `--fields` is mandatory)

```bash
uv run python exp/common/build_clip_cache_artifact.py \
    --data-dir exp/common/data/db/libero_cache/libero_spatial \
    --fields vision_0,vision_1,prompt_emb,robot_state \
    --device cuda \
    --output exp/common/data/cache_artifacts/libero_spatial_warm/clip_vit_b_32.pkl
```

> `--fields` must list all four: the YAMLs `clip_w7_d4_warm_t*.yaml` enable `vision_0 / vision_1 / prompt_emb / robot_state` and declare matching `backend.vector_dims`. The builder defaults to `vision_0,robot_state`, so missing fields trigger a `vector_dims` mismatch when `InMemoryBackend.load_artifact()` runs.

On GPU (batch=64, ViT-B-32) this takes ~1 minute and produces ~194 MB. With < 2 GB VRAM drop to `--device cpu` (~10 minutes).

### 1.3 Verify intermediates (always run this)

```bash
uv run python - <<'PY'
import pickle
expected = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for name in ["cp1_max_pool","cp1_spatial_pool_16","clip_vit_b_32"]:
    with open(f"exp/common/data/cache_artifacts/libero_spatial_warm/{name}.pkl","rb") as f:
        obj = pickle.load(f)
    e = obj["entries"][0]
    keys = sorted(e.payload.intermediates.keys())
    assert keys == expected, (name, keys)
    assert e.payload.denoising_num_steps == 10
    full = sum(1 for x in obj["entries"]
               if x.payload.intermediates and len(x.payload.intermediates) == 9)
    print(f"{name}: {len(obj['entries'])} entries, {full} full intermediates")
PY
```

All three should print `1018 entries, 1018 full intermediates`. If any artifact fails this check, Step 2 smoke will fail — rebuild before continuing.

---

## Step 2: Smoke pass (one YAML × one task × 5 ep)

Minimal end-to-end check that (i) the YAML loads on the server, (ii) the WARM_START branch fires, (iii) the orchestrator does not downgrade.

```bash
uv run python exp/common/run_cache_experiments.py \
    --yaml-dir exp/warm_start/config/max_pool \
    --runs 2 \
    --task-ids 0 \
    --task-suite libero_spatial \
    --host 155.98.36.13 --port 9000 \
    --episodes-per-run 5 \
    --num-workers 1 \
    --seed 42 \
    --conda-env libero_sim \
    --state-path exp/warm_start/data/state_smoke.json
```

Flag notes:

- `--runs 2` indexes `sorted(yaml_dir.glob("*.yaml"))` (lexicographic) → `max_pool_w3_d5_warm_t0.5.yaml`. The order is `t0.3 → t0.5 → t0.7`.
- `--task-ids 0` + `--episodes-per-run 5` + `--num-workers 1` = five serial episodes, the fastest viable end-to-end probe.

**Pass criteria** (check both driver output and server log):

1. `exp/warm_start/data/max_pool/cache_eval_results.json` contains `config_id == "max_pool_w3_d5_warm_t0.5"` with exactly 5 records.
2. Server log: `grep -cE "judge: WARM_START"` **≥ 1** (WARM_START branch actually hit).
3. Server log: `grep -c "WARM_START payload incomplete"` **== 0** (no downgrade to MISS; if non-zero, return to Step 1.3).
4. At least one `timing_task_*.csv` under `exp/warm_start/data/timing/max_pool_w3_d5_warm_t0.5/` (timer CSV path works).

Optional 3-tier load check: rerun with `--runs 1-3 --episodes-per-run 1`; server log should show three `Cache bundle updated to v*` entries.

---

## Step 3: Full run (3 servers in parallel)

Three terminals, each client pinned to one server that serves one keybuilder's three start_t YAMLs.

```bash
# Terminal 1 — max_pool (frp 9000)
uv run python exp/common/run_cache_experiments.py \
    --yaml-dir exp/warm_start/config/max_pool \
    --task-suite libero_spatial \
    --host 155.98.36.13 --port 9000 \
    --episodes-per-run 50 --num-workers 5 --seed 42 \
    --conda-env libero_sim \
    --state-path exp/warm_start/data/state_full_max_pool.json

# Terminal 2 — spatial16 (frp 8999)
uv run python exp/common/run_cache_experiments.py \
    --yaml-dir exp/warm_start/config/spatial16 \
    --task-suite libero_spatial \
    --host 155.98.36.13 --port 8999 \
    --episodes-per-run 50 --num-workers 5 --seed 42 \
    --conda-env libero_sim \
    --state-path exp/warm_start/data/state_full_spatial16.json

# Terminal 3 — clip (frp 8998)
uv run python exp/common/run_cache_experiments.py \
    --yaml-dir exp/warm_start/config/clip \
    --task-suite libero_spatial \
    --host 155.98.36.13 --port 8998 \
    --episodes-per-run 50 --num-workers 5 --seed 42 \
    --conda-env libero_sim \
    --state-path exp/warm_start/data/state_full_clip.json
```

> The three `--state-path` values **must differ** or RunState entries will clobber each other. `--num-workers 5` × 3 clients = 15 libero env subprocesses; drop to 3 if the eval host has `nproc < 16`.

Before each warm YAML the driver sends `send_load_cache_config`, and the server calls `build_shared_storage` → `InMemoryBackend.load_artifact(preload_path)` again. Each YAML therefore re-pickles its artifact (max_pool < 1 s, spatial16 a few seconds, CLIP 1–2 s) — negligible versus the minute/hour-scale run time per 500 episodes.

Archive afterwards (keeps the next run from overwriting):

```bash
mkdir -p exp/warm_start/data/results
for cfg in max_pool spatial16 clip; do
    src=exp/warm_start/config/$cfg/cache_eval_results.json
    [ -f "$src" ] && cp "$src" exp/warm_start/data/results/cache_eval_results_${cfg}.json
done
uv run python - <<'PY'
import json, glob
out = []
for fn in sorted(glob.glob("exp/warm_start/data/results/cache_eval_results_*.json")):
    out.extend(json.load(open(fn)))
json.dump(out, open("exp/warm_start/data/results/cache_eval_results.json","w"), indent=2)
print("merged", len(out), "records")
PY
```

Expect 9 × 500 = **4500 records** (3000 if max_pool + spatial16 only). Retry records with `attempt > 1` indicate GPU/CPU pressure — filter them before sanity-checking success_rate.

---

## Step 4: Analysis

`exp/warm_start/analyze_warm_sweep.py` reads `exp/warm_start/data/results/cache_eval_results.json` + `exp/warm_start/data/baseline_failures.json` and emits:

| Artifact | Meaning |
|----------|---------|
| `success_rate_sweep.png` | x = start_t, y = success_rate; B0 / B1 drawn as horizontal dashed lines |
| `recovery_on_b1_failure.png` | `|warm_pass ∩ B1_fail| / |B1_fail|` — how many always-hit failures were rescued |
| `incurred_loss.png` | `|warm_fail ∩ B0_pass| / |B0_pass|` — how many originally passing inits warm start killed |
| `mean_step_latency.png` (auxiliary) | Mean of `stage3_warm / stage3_flow / cp1_sum` from timer CSVs |
| `summary.csv` | `(cfg, start_t, n_total, n_success, success_rate, recovery_rate, incurred_loss, p_value)` |
| `failure_intersection.csv` | For each `(cfg, start_t)`, overlap between warm-fail inits and the 51-init "hard core" (B1 failure intersection across cfgs) |

**Reading priority**: first check whether `recovery_rate` is monotonic in start_t (higher start_t → more recovery). Then look at `incurred_loss`: if `start_t=0.7` has a large `incurred_loss`, overshooting warm is hurting previously-passing inits — cross-check the latency plot.

---

## Parameter quick reference

| Where | Parameter | Default | When to change |
|-------|-----------|---------|----------------|
| YAML `judge.start_t` | 0.3 / 0.5 / 0.7 | — | Must be one of {0.1..0.9}; extending to 0.1 / 0.9 does **not** require rebuilding pkl (it already stores all 9) |
| YAML `timer.output_csv_dir` | `exp/warm_start/data/timing/<cfg>_warm_t<st>` | — | Must not be null; otherwise the latency plot has no data |
| YAML `backend.in_memory.preload_path` | `exp/common/data/cache_artifacts/libero_spatial_warm/<builder>.pkl` | — | Change when switching datasets (e.g. libero_10 / libero_object) |
| `run_cache_experiments.py --runs` | all | — | For smoke, `--runs 2` maps to `t0.5` under lexicographic sort |
| `run_cache_experiments.py --num-workers` | 5 | 5 | Drop to 3 when eval `nproc < 16`; to 1 when server OOMs |

## Troubleshooting

- **WARM_START downgrades to MISS**: server log contains `WARM_START payload incomplete`. The log level was bumped to warning in `orchestrator.py`, so it is visible at INFO. Root cause is always a Step 1 artifact missing `intermediates`; rerun Step 1.3.
- **CLIP `vector_dims` mismatch on startup**: rebuild with `--fields vision_0,vision_1,prompt_emb,robot_state`. The builder's default 2-field output does not match the YAML's 4-field `vector_dims`.
- **Wrong YAML picked by `--runs N`**: current filenames sort lexicographically as `t0.3 → t0.5 → t0.7`. Do not prefix with `01_` / `02_` — that would drift the trajectory_deviation state-path conventions downstream.
- **Eval-host CPU saturated with 3 parallel clients**: 15 libero subprocesses is too much for `nproc < 16`; lower `--num-workers` to 3.
