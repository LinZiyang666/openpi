# Trajectory Deviation Step 2 Parallel Commands

Date: 2026-04-15

Goal: run Step 2 deviate-score computation in parallel with three independent
policy servers and three client processes.

Important constraints:

- The GT directory is shared and read-only. All three clients can use the same
  `data/deviation_experiment/gt`; it does not need to be copied or isolated per
  config.
- Each client filters the shared GT directory with
  `--config-fail-results data/deviation_experiment/cache_eval_results_cache_fail.json`,
  so a config only runs init states that failed under that same config.
- The Step 2 output directory must be isolated per config, because each client
  writes its own state JSON, Phase 1/2 JSONL, and final deviate-score JSON.
- Do not point two Step 2 clients at the same server. Each client below owns one
  server, so `load_cache_config` cannot race across configs.
- Server-side `--collect` is not needed for Step 2. Step 2 replays saved GT
  observations through policy inference and writes client-side JSONL outputs.
  Keep `--collect` for Step 1b / Step 3 workflows, not for this Step 2 run.

Assumed port mapping:

| config | server local port | frp client port |
|---|---:|---:|
| `clip_w7_d4` | `7998` | `8998` |
| `spatial16_w8_d4` | `7999` | `8999` |
| `max_pool_w3_d5` | `8000` | `9000` |

Public host used by clients: `155.98.36.13`

## Server Commands

Run these on the GPU server side, one command per server/session.

### Server A: clip, local port 7998

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config configs/cache_runs/deviate_exp/inference_clip_w7_d4.yaml \
    --env LIBERO \
    --port 7998 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

### Server B: spatial16, local port 7999

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config configs/cache_runs/deviate_exp/inference_spatial16_w8_d4.yaml \
    --env LIBERO \
    --port 7999 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

### Server C: max_pool, local port 8000

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config configs/cache_runs/deviate_exp/inference_max_pool_w3_d5.yaml \
    --env LIBERO \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

Health checks from the client/eval host:

```bash
curl http://155.98.36.13:8998/healthz
curl http://155.98.36.13:8999/healthz
curl http://155.98.36.13:9000/healthz
```

Expected output for each:

```text
OK
```

## Client Commands

Run these on the LIBERO eval/client host, ideally in three terminals. Each
process uses 5 websocket workers and writes to a separate output directory.

This uses `--M 10` as a practical first full run. Change to `--M 20` if you
want the original default background estimate.

With the current Step 1b GT set and default success-only filtering, expected
episode counts are:

| config | GT episodes run |
|---|---:|
| `clip_w7_d4` | 159 |
| `spatial16_w8_d4` | 154 |
| `max_pool_w3_d5` | 150 |

### Client 1: clip via frp port 8998

```bash
uv run python -m exp.trajectory_deviation.compute_deviate_scores \
    --configs clip_w7_d4 \
    --gt-dir data/deviation_experiment/gt \
    --out-dir data/deviation_experiment/deviate_scores_clip \
    --M 10 \
    --num-workers 5 \
    --host 155.98.36.13 --port 8998 \
    --floor 0.1 \
    --config-yaml-dir configs/cache_runs/deviate_exp \
    --config-fail-results data/deviation_experiment/cache_eval_results_cache_fail.json \
    --resume
```

### Client 2: spatial16 via frp port 8999

```bash
uv run python -m exp.trajectory_deviation.compute_deviate_scores \
    --configs spatial16_w8_d4 \
    --gt-dir data/deviation_experiment/gt \
    --out-dir data/deviation_experiment/deviate_scores_spatial16 \
    --M 10 \
    --num-workers 5 \
    --host 155.98.36.13 --port 8999 \
    --floor 0.1 \
    --config-yaml-dir configs/cache_runs/deviate_exp \
    --config-fail-results data/deviation_experiment/cache_eval_results_cache_fail.json \
    --resume
```

### Client 3: max_pool via frp port 9000

```bash
uv run python -m exp.trajectory_deviation.compute_deviate_scores \
    --configs max_pool_w3_d5 \
    --gt-dir data/deviation_experiment/gt \
    --out-dir data/deviation_experiment/deviate_scores_maxpool \
    --M 10 \
    --num-workers 5 \
    --host 155.98.36.13 --port 9000 \
    --floor 0.1 \
    --config-yaml-dir configs/cache_runs/deviate_exp \
    --config-fail-results data/deviation_experiment/cache_eval_results_cache_fail.json \
    --resume
```

## Merge Final Step 2 Outputs

After all three clients finish, collect the final score JSONs into the standard
Step 2 directory for Step 3 / analysis.

```bash
mkdir -p data/deviation_experiment/deviate_scores

cp data/deviation_experiment/deviate_scores_clip/deviate_score_clip_w7_d4.json \
   data/deviation_experiment/deviate_scores/

cp data/deviation_experiment/deviate_scores_spatial16/deviate_score_spatial16_w8_d4.json \
   data/deviation_experiment/deviate_scores/

cp data/deviation_experiment/deviate_scores_maxpool/deviate_score_max_pool_w3_d5.json \
   data/deviation_experiment/deviate_scores/
```

Optional: also merge raw Phase 1/2 JSONL dumps for inspection.

```bash
cp data/deviation_experiment/deviate_scores_clip/bg_clip_w7_d4.jsonl \
   data/deviation_experiment/deviate_scores/
cp data/deviation_experiment/deviate_scores_clip/cache_clip_w7_d4.jsonl \
   data/deviation_experiment/deviate_scores/

cp data/deviation_experiment/deviate_scores_spatial16/bg_spatial16_w8_d4.jsonl \
   data/deviation_experiment/deviate_scores/
cp data/deviation_experiment/deviate_scores_spatial16/cache_spatial16_w8_d4.jsonl \
   data/deviation_experiment/deviate_scores/

cp data/deviation_experiment/deviate_scores_maxpool/bg_max_pool_w3_d5.jsonl \
   data/deviation_experiment/deviate_scores/
cp data/deviation_experiment/deviate_scores_maxpool/cache_max_pool_w3_d5.jsonl \
   data/deviation_experiment/deviate_scores/
```

## Quick Verification

```bash
find data/deviation_experiment/deviate_scores_clip \
     data/deviation_experiment/deviate_scores_spatial16 \
     data/deviation_experiment/deviate_scores_maxpool \
     -maxdepth 1 -type f -printf '%p %s bytes\n' | sort
```

Expected final score files:

```text
data/deviation_experiment/deviate_scores_clip/deviate_score_clip_w7_d4.json
data/deviation_experiment/deviate_scores_spatial16/deviate_score_spatial16_w8_d4.json
data/deviation_experiment/deviate_scores_maxpool/deviate_score_max_pool_w3_d5.json
```
