# Trajectory Deviation Experiment Runbook

> Based on `logs/trajectory_deviation_experiment_plan.log.md` and `logs/trajectory_deviation_corrective_experiment.log.md`, aligned with the post-reorg `exp/trajectory_deviation/` script paths and CLIs. Chinese version: [trajectory_deviation.md](trajectory_deviation.md).

---

## Overview

For each cache configuration (`clip_w7_d4` / `spatial16_w8_d4` / `max_pool_w3_d5`) we answer two questions:

1. How far does the cache pull inference off the GT trajectory, and can that divergence be quantified by a **deviate score**?
2. For the top-k highest-scoring cycles, if we teleport the env back to the GT `sim_state` and let a pure-cache rollout continue, does the policy recover?

Full pipeline:

```
Step 1a  baseline cache eval            → cache_eval_results.json (per-episode success)
Step 1b-pre dump_step1a_failed_inits    → per-task .init + step1b_filter.json
Step 1b  run_step1b_gt.py               → GT HDF5 (AlwaysSkip, recommended max_pool inference bundle)
Step 2   compute_deviate_scores         → deviate_score_{cfg}.json (Phase1 M× + Phase2 1× + Phase3 aggregate)
Step 3   run_step3_per_cycle_policy     → results.jsonl (one server + one client per cfg)
         merge_step3_cfgs               → summary.csv (cross-cfg aggregate)
```

## Topology (same as CP1 cache experiment)

```
┌─────────────────────────┐        frp tunnel / LAN         ┌────────────────────────┐
│  GPU server (no public) │ ◄──────────────────────────────│  LIBERO eval host       │
│  serve_policy.py         │   155.98.36.32:9000             │  exp/trajectory_        │
│  listens on :8000        │   → localhost:8000              │  deviation/*.py          │
│  --concurrent            │                                  │  examples/libero/main   │
└─────────────────────────┘                                  └────────────────────────┘
```

- **GPU server**: Pi0.5 + cache backend; started once, survives the whole pipeline.
- **Eval host**: runs the libero env plus every `exp/trajectory_deviation/*.py` driver. `main.py` subprocesses are launched under `--conda-env libero_sim`.
- Networking identical to CP1 — if you already run `cp1_cache.md` Step 4/5, reuse it verbatim.

---

## Parallelism cheat sheet (read before scaling out)

| Stage | Parallelisable axis | Must be sequential | Notes |
|-------|--------------------|--------------------|-------|
| Step 1a | task level (`run_cache_experiments.py --num-workers`) | different YAML runs serialise | workers share one server; server needs `--concurrent` |
| Dump failed inits | — | single process | offline JSON → tensor slicing, seconds |
| Step 1b GT collection | **not parallel** | each unit spawns its own `main.py` + libero env; runner uses serial `run()` (not `parallel_run()`) | parallel subprocesses would fight over GPU/RAM; switch one AlwaysSkip bundle once (recommend `inference_max_pool_w3_d5.yaml` to avoid CLIP) |
| Step 2 Phase 1 / Phase 2 | episode × sample (`--num-workers`) | configs serialise; inside one config Phase1 → Phase2 is sequential | driver issues `send_load_cache_config` between phases — concurrent drivers would race |
| Step 2 Phase 3 (aggregate) | — | single process, pure numpy | one output per cfg |
| Step 3 per-cycle | episode × (τ, n) (`--num-workers` ≤ 5, MuJoCo EGL cap) | different cfgs serialise on one server | Recommended: 3 cfgs × 3 servers fully parallel; each client process binds a single `--cfg` |

> **Turn on `--num-workers` wherever possible**; **do NOT fan out where the table says "sequential"** — parallel drivers race on `load_cache_config` (corrupting bundle state) or overwrite the same HDF5.

---

## Prerequisites

1. Both sides: `GIT_LFS_SKIP_SMUDGE=1 uv sync`.
2. GPU server has a local Pi0.5 LIBERO checkpoint (default path `$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch`; point `--policy.dir` at it to skip GCS downloads).
3. Eval host has the `libero_sim` conda env.
4. Cache artifacts + calibration built per [cp1_cache.md](cp1_cache.md) Steps 1–2:
   - `clip_vit_b_32.pkl` (used by `clip_w7_d4.yaml` / `inference_clip_w7_d4.yaml`)
   - `cp1_spatial_pool_16.pkl` (used by `spatial16_w8_d4.yaml` / `inference_spatial16_w8_d4.yaml`)
   - `cp1_max_pool.pkl` (used by `max_pool_w3_d5.yaml` / `inference_max_pool_w3_d5.yaml`)
5. All 6 YAMLs under `exp/trajectory_deviation/config/` present, with `preload_path` pointing to the server-side artifact locations.

---

## Step 0 — Start the GPU server (once, fully configured)

One server must survive the whole pipeline; pass every option up front. Each downstream driver calls `send_load_cache_config` to switch bundles, so no restarts are needed.

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache_config exp/trajectory_deviation/config/inference_max_pool_w3_d5.yaml \
    --env LIBERO \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

Flag reference:

| Flag | Meaning | Required here? |
|------|---------|----------------|
| `--concurrent` | Multi-connection mode; allows multi-worker clients **and** the `load_cache_config` control message | **Yes.** Step 2/3 `--num-workers>1` and dynamic bundle switching both depend on it |
| `--cache_config` | Initial bundle at startup | Any AlwaysSkip YAML works; preloading the GT bundle (`inference_max_pool_w3_d5.yaml`) lets Step 1b use `--skip-config-switch` and avoids loading CLIP during GT collection |
| `--env LIBERO` | Select env branch | Fixed |
| `--port 8000` | Listener | Match the client's `--port`; if you tunnel via frp, clients hit the remapped external port (e.g. `9000`) |
| `policy:checkpoint --policy.config pi05_libero --policy.dir <local>` | Use the local checkpoint instead of GCS | Strongly recommended |

Sanity check from the eval host:

```bash
curl http://<server-host>:<server-port>/healthz
# → OK
```

> Step 0 matches `cp1_cache.md` Step 4; if you already ran CP1, reuse the same server.

---

## Step 1a — Baseline cache evaluation

Goal: per-episode `success` flags, to identify episodes where the cache actually breaks.

```bash
uv run exp/common/run_cache_experiments.py \
    --yaml-dir exp/deviate_exp \
    --task-suite libero_spatial \
    --host <server-host> --port <server-port> \
    --episodes-per-run 50 \
    --num-workers 4 \
    --seed 42 \
    --conda-env libero_sim \
    --state-path exp/trajectory_deviation/data/step1a_state.json
```

Flag reference & tuning:

| Flag | Meaning | Tuning guidance |
|------|---------|-----------------|
| `--yaml-dir` | Directory of YAMLs the driver iterates | `exp/deviate_exp` (6 files; the 3 `inference_*` are AlwaysSkip controls — fine to include, their success_rate will mirror pure inference) |
| `--episodes-per-run` | Episodes per task; LIBERO has 50 inits per task | 50 for a clean failure set; 10 for a quick smoke run |
| `--num-workers` | Task-level parallelism, workers share one server | 4 is safe with `--concurrent`; raise to 8 on bigger GPUs |
| `--seed` | Seed forwarded to `main.py`, controls libero env randomness | Fix at 42 for reproducibility; deliberately different from data-collection seed=7 |
| `--conda-env` | `main.py` must run inside the libero env | Fixed `libero_sim` |
| `--state-path` | Resume state file | Point into the experiment dir so `--resume` works across sessions |

Output: `<yaml-dir>/cache_eval_results.json`. Symlink/copy it into the experiment root so Step 1b-pre finds it:

```bash
cp exp/trajectory_deviation/config/cache_eval_results.json \
   exp/trajectory_deviation/data/cache_eval_results.json
```

---

## Step 1b-pre — Dump failed inits (offline single-process, seconds)

```bash
uv run python exp/trajectory_deviation/dump_step1a_failed_inits.py \
    --step1a-results exp/trajectory_deviation/data/cache_eval_results.json \
    --task-suite libero_spatial \
    --out-dir exp/trajectory_deviation/data/inits
```

Flag reference:

| Flag | Meaning | Tuning guidance |
|------|---------|-----------------|
| `--step1a-results` | Aggregated Step 1a JSON | Must be the deduped `cache_eval_results.json`, not a single-run log |
| `--task-suite` | LIBERO suite name; used to look up `task.name` and init states | Same as Step 1a |
| `--out-dir` | Artifact root | `exp/trajectory_deviation/data/inits`, referenced by later steps |
| `--no-torch` | Skip `.init` tensor write, emit only JSON | CI smoke only; don't pass in production runs |

Outputs:
- `<out>/<task_name>.init` — `(K, 92)` subset tensor of failed inits
- `<out>/<task_name>.init_map.json` — subset → orig index map
- `<out>/step1b_filter.json` — fed to the next stage's `--episode-filter`

---

## Step 1b — GT trajectory collection (**sequential, AlwaysSkip locked**)

Step 1b only needs an **AlwaysSkip / pure-inference** bundle; all three Step 2 configs must share the same GT or the deviate-score denominator stops being comparable. Which `inference_*.yaml` you use does not change GT action semantics because the gate is `always_skip` and no cached action is read. Recommended: `inference_max_pool_w3_d5.yaml`, which avoids the CLIP key builder and does not spend VRAM on an unused CLIP model. The runner switches the bundle once at startup and then serialises every unit through `main.py` (each unit is its own libero env subprocess).

```bash
uv run python -m exp.trajectory_deviation.run_step1b_gt \
    --inits-dir exp/trajectory_deviation/data/inits \
    --out-dir exp/trajectory_deviation/data/gt \
    --task-suite libero_spatial \
    --host <server-host> --port <server-port> \
    --inference-yaml exp/trajectory_deviation/config/inference_max_pool_w3_d5.yaml \
    --seed 7 \
    --conda-env libero_sim \
    --resume
```

Flag reference:

| Flag | Meaning | Tuning guidance |
|------|---------|-----------------|
| `--inits-dir` | Step 1b-pre output dir | Required; driver reads `step1b_filter.json` from here |
| `--out-dir` | GT HDF5 root (forwarded to `main.py --save-trajectory-dir`) | Must match Step 2's `--gt-dir`; typically `exp/trajectory_deviation/data/gt` |
| `--task-suite` | LIBERO suite | Same as earlier steps |
| `--host / --port` | Server endpoint | frp mapped port on eval host; `localhost:8000` locally |
| `--seed` | Seed forwarded to `main.py` | **Fix at 7** (plan §9.2 default); Phase 1 background noise estimation is built on this seed. Changing it would de-align the M samples from the GT distribution |
| `--conda-env` | libero env | Fixed `libero_sim` |
| `--inference-yaml` | GT bundle YAML path | Must be a pure-inference bundle with `gate.type: always_skip`; recommended `exp/trajectory_deviation/config/inference_max_pool_w3_d5.yaml` to avoid CLIP. Do not use a real cache YAML such as `clip_w7_d4.yaml` or `max_pool_w3_d5.yaml` |
| `--state-path` | Runner state file | Default `<inits-dir>/step1b_state.json`; pair with `--resume` |
| `--resume` | Resume from state | Almost always on |
| `--skip-config-switch` | Skip `load_cache_config` | Only when sharding across servers and you switched bundles by hand |

Tuning notes:

- **Faster?** You can't — the runner is serial by design. Options are (a) raise `max_retries` only if units flake out, or (b) rewrite the runner around `parallel_run` + per-worker env, which plan doesn't authorise.
- **One unit keeps failing?** Raise `max_retries` (default 2) or reproduce it manually by invoking `examples/libero/main.py` with that single `(task_id, orig_init)` for debugging.
- Outputs: `gt/task_{id}/episode_{subset_idx}.h5`.

---

## Step 2 — Deviate score (parallel within a cfg, serial across cfgs)

Run all three cfgs at once (the driver switches bundles in order):

```bash
uv run python -m exp.trajectory_deviation.compute_deviate_scores \
    --configs clip_w7_d4 spatial16_w8_d4 max_pool_w3_d5 \
    --gt-dir exp/trajectory_deviation/data/gt \
    --out-dir exp/trajectory_deviation/data/deviate_scores \
    --M 20 \
    --num-workers 4 \
    --host <server-host> --port <server-port> \
    --floor 0.1 \
    --config-fail-results exp/trajectory_deviation/data/cache_eval_results_cache_fail.json \
    --resume
```

Per-cfg order:

```
load inference_{cfg}.yaml → Phase1Runner  (parallel, M × episodes)   → bg_{cfg}.jsonl
load {cfg}.yaml           → Phase2Runner  (parallel, 1 × episodes)   → cache_{cfg}.jsonl
offline Phase3 aggregate                                              → deviate_score_{cfg}.json
```

Flag reference & tuning:

| Flag | Default | Meaning | Tuning guidance |
|------|---------|---------|-----------------|
| `--configs` | — | Cache-config IDs (no `.yaml` suffix; driver resolves `inference_{cfg}.yaml` and `{cfg}.yaml`) | Pass all three for a meaningful comparison; single cfg for debugging |
| `--gt-dir` | — | Step 1b output root | Must match Step 1b `--out-dir` |
| `--out-dir` | — | Root for jsonl + state + `deviate_score_*.json` | `exp/trajectory_deviation/data/deviate_scores` |
| `--M` | 20 | Phase 1 stochastic-sample count used to estimate background L2 noise | **The SNR knob.** Larger M → more stable bg_l2 estimate, Phase 1 time grows linearly. Plan §10.2 reference is 20; 10 for smoke; 30 if you want tighter confidence |
| `--num-workers` | 4 | Within-cfg worker count (server needs `--concurrent`) | Bounded by server GPU capacity; 4 safe, 8 on bigger cards |
| `--host / --port` | `localhost:8000` | Server endpoint | Match Step 0 |
| `--floor` | 0.1 | Denominator floor: `max(bg_l2, floor)` — prevents division-by-tiny-bg | **Avoid changing** (plan §10.2 empirical value); if M is small and bg_l2 frequently drops below 0.1, experiment with 0.05 to inspect distribution |
| `--config-yaml-dir` | `exp/deviate_exp` | YAML lookup root | Do not change unless relocating the experiment |
| `--config-fail-results` | off | Optional Step-1a results JSON; when set, each cfg only runs `(task_id, orig_init_state_idx)` units that failed under that cfg | Recommended: `exp/trajectory_deviation/data/cache_eval_results_cache_fail.json`, so inits that already succeeded for the active cfg are not scored. Leave unset to evaluate all configs on the same successful GT set |
| `--skip-config-switch` | off | Do not call `load_cache_config`, assume the server is already on the right bundle | Use when sharding: one server per cfg (each on its own port), each client with this flag |
| `--include-failed-gt` | off | Keep `success=False` GT episodes | Default drops them (failed GTs carry no recovery signal); turn on for noise analysis |
| `--include-unknown-gt` | off | Keep legacy HDF5 without a `success` attr | Compatibility shim; rerun Step 1b instead |
| `--resume` | off | Resume from state file | Almost always on |

Common recipes:

- **Quick look**: `--M 10 --num-workers 4 --configs clip_w7_d4` — single cfg, half the samples, gets you a `deviate_score_clip_w7_d4.json` distribution fast.
- **Full comparison**: `--M 20` or higher, all three cfgs, `--num-workers` at whatever your GPU can take.
- **Looks wrong (all deviate ≈ 1)?** You almost certainly ran multiple drivers against one server and they stomped on each other's `load_cache_config`. Fix: single driver + serial cfgs, or one server per cfg with `--skip-config-switch`.

⚠️ **Never** run two `compute_deviate_scores` processes against the same server — Phase 1 may read the Phase 2 bundle and the reported deviate scores will cluster around 1.0 (useless).

---

## Step 3 — Per-cycle policy (recommended: 3 cfgs × 3 servers fully parallel)

> `run_step3_per_cycle_policy` lets the client decide — per inference cycle — whether to bypass the cache, by injecting a `__gate_decision__` signal consumed by `ClientControlledGate`. Full redesign notes: [`logs/trajectory_deviation_step3_redesign.log.md`](../../logs/trajectory_deviation_step3_redesign.log.md).
>
> Authoritative episode source: keys of `deviate_score_{cfg}.json` (`task_X/episode_Y`, where `Y` is the Step 1b `subset_init_state_idx`). Those keys already are the per-cfg Step 1a failure subset, so Step 3 no longer takes `--cache-eval-results` / `--config-fail-results`.

### Prerequisites

1. Three Step 2 outputs available: `deviate_score_clip_w7_d4.json` / `deviate_score_spatial16_w8_d4.json` / `deviate_score_max_pool_w3_d5.json` (merged into `exp/trajectory_deviation/data/deviate_scores/` per the [Step 2 parallel runbook](../../logs/trajectory_deviation_step2_parallel_commands.log.md)).
2. Step 1b pruned init states at `exp/trajectory_deviation/data/inits/`.
3. `exp/trajectory_deviation/config/step3_{cfg}.yaml` present for all three cfgs; each sets `checkpoints.cp1.gate.type: client_controlled`, other fields mirror the corresponding `{cfg}.yaml`.
4. Three servers reuse the Step 2 port map:

    | config | server local port | frp external port |
    |---|---:|---:|
    | `clip_w7_d4` | `7998` | `8998` |
    | `spatial16_w8_d4` | `7999` | `8999` |
    | `max_pool_w3_d5` | `8000` | `9000` |

    Public host: `155.98.36.32`

### Server commands

If the Step 2 servers are still running, **no restart needed** — `run_step3_per_cycle_policy` calls `send_load_cache_config` at startup to swap the bundle to `step3_{cfg}.yaml`. Otherwise launch them with the commands below (any `--cache-config` works; the client will switch it).

#### Server A: clip, local port 7998

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/trajectory_deviation/config/step3_clip_w7_d4.yaml \
    --env LIBERO \
    --port 7998 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

#### Server B: spatial16, local port 7999

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/trajectory_deviation/config/step3_spatial16_w8_d4.yaml \
    --env LIBERO \
    --port 7999 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

#### Server C: max_pool, local port 8000

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache-config exp/trajectory_deviation/config/step3_max_pool_w3_d5.yaml \
    --env LIBERO \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir "$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
```

Health checks (from the eval host):

```bash
curl http://155.98.36.32:8998/healthz
curl http://155.98.36.32:8999/healthz
curl http://155.98.36.32:9000/healthz
```

Each should print `OK`.

### Client commands

Launch three client processes on the LIBERO eval host, one per terminal. Each binds a single cfg and server; `--num-workers 5` is the MuJoCo EGL cap; `--tau-grid` / `--n-grid` are comma-separated.

#### Client 1: clip via frp port 8998

```bash
uv run python -m exp.trajectory_deviation.run_step3_per_cycle_policy \
    --cfg clip_w7_d4 \
    --host 155.98.36.32 --port 8998 \
    --yaml exp/trajectory_deviation/config/step3_clip_w7_d4.yaml \
    --deviate-score-json exp/trajectory_deviation/data/deviate_scores/deviate_score_clip_w7_d4.json \
    --init-states-dir exp/trajectory_deviation/data/inits \
    --out-dir exp/trajectory_deviation/data/step3/clip_w7_d4 \
    --task-suite-name libero_spatial \
    --tau-grid 3,5,7,10 \
    --n-grid 1,2,3,5,10 \
    --num-workers 5 \
    --resume
```

#### Client 2: spatial16 via frp port 8999

```bash
uv run python -m exp.trajectory_deviation.run_step3_per_cycle_policy \
    --cfg spatial16_w8_d4 \
    --host 155.98.36.32 --port 8999 \
    --yaml exp/trajectory_deviation/config/step3_spatial16_w8_d4.yaml \
    --deviate-score-json exp/trajectory_deviation/data/deviate_scores/deviate_score_spatial16_w8_d4.json \
    --init-states-dir exp/trajectory_deviation/data/inits \
    --out-dir exp/trajectory_deviation/data/step3/spatial16_w8_d4 \
    --task-suite-name libero_spatial \
    --tau-grid 3,5,7,10 \
    --n-grid 1,2,3,5,10 \
    --num-workers 5 \
    --resume
```

#### Client 3: max_pool via frp port 9000

```bash
uv run python -m exp.trajectory_deviation.run_step3_per_cycle_policy \
    --cfg max_pool_w3_d5 \
    --host 155.98.36.32 --port 9000 \
    --yaml exp/trajectory_deviation/config/step3_max_pool_w3_d5.yaml \
    --deviate-score-json exp/trajectory_deviation/data/deviate_scores/deviate_score_max_pool_w3_d5.json \
    --init-states-dir exp/trajectory_deviation/data/inits \
    --out-dir exp/trajectory_deviation/data/step3/max_pool_w3_d5 \
    --task-suite-name libero_spatial \
    --tau-grid 3,5,7,10 \
    --n-grid 1,2,3,5,10 \
    --num-workers 5 \
    --resume
```

Flag reference & tuning:

| Flag | Default | Meaning | Tuning guidance |
|------|---------|---------|-----------------|
| `--cfg` | — | Single cache-config id to run | Required; one client process, one cfg |
| `--host / --port` | — | Server endpoint (eval host uses the frp external port) | Required; three clients must hit three different servers to avoid `load_cache_config` races |
| `--yaml` | — | Step 3 cache YAML with `gate.type: client_controlled` | Required; the path must be resolvable on the server's filesystem |
| `--deviate-score-json` | — | Step 2 `deviate_score_{cfg}.json`; its keys drive the Step 3 episode list | Required; an empty JSON aborts with `has no episodes` |
| `--init-states-dir` | — | Step 1b-pre output dir (`{task}.init` / `{task}.init_map.json`) | Required |
| `--out-dir` | — | Output root: `run_state.json` + `results.jsonl` | `exp/trajectory_deviation/data/step3/{cfg}` |
| `--task-suite-name` | `libero_spatial` | Must equal earlier steps; drives `_MAX_STEPS_BY_SUITE` | Keep in sync with Step 1a/1b |
| `--tau-grid` | `3,5,7,10` | deviate_score threshold grid (scalars, not burst lengths) | **Core knob.** Smaller τ → search triggers more often; larger τ → more skips |
| `--n-grid` | `1,2,3,5,10` | Cycles of cache after a search fires (burst length) | **Second core knob.** Larger n lets cache carry a longer stretch after each search; n=1 re-evaluates every cycle |
| `--replan-steps` | 5 | env.steps actually executed per inference cycle | Matches `examples/libero/main.py`; lower = higher control resolution but more server calls |
| `--num-steps-wait` | 10 | No-op env.steps after reset (libero "object settle") | Matches libero main; rarely changed |
| `--resize-size` | 224 | Policy input image resolution | Fixed |
| `--num-workers` | 1 | Within-cfg LIBERO env concurrency | **Cap at 5** (MuJoCo EGL hard limit; runner aborts above that) |
| `--max-cycles-safety` | 5 | Extra cycles beyond `ceil(max_env_steps/replan_steps)` as a runaway guard | Default is fine |
| `--experiment-tag` | `trajectory_deviation_step3` | Forwarded to `client.episode_start(experiment=...)` | Useful to tag server-side logs |
| `--skip-load-cache-config` | off | Skip the startup `send_load_cache_config` | Only when the server is already on the right bundle |
| `--resume` | off | Resume from `run_state.json` | Almost always on |

`(τ, n)` combinations = `len(tau_grid) × len(n_grid)` = 4×5=20 by default; multiplied by episodes × 3 cfgs gives total unit count. Recipes:

- **Smoke**: `--tau-grid 5 --n-grid 1,5 --cfg clip_w7_d4 --num-workers 1` — 2 combos × single cfg × single worker; verifies gate signal injection and success semantics.
- **Full run**: default 4×5 + all 3 cfgs in parallel. Mirrors the Step 2 cadence of ~3h per server for ~150 episodes of inference-only feedback.
- **Tight on GPU?** Lower `--num-workers` first (≤5), then trim `--tau-grid`.

⚠️ Do not stack multiple `run_step3_per_cycle_policy` processes on one server — concurrent `load_cache_config` calls would corrupt the bundle write ordering and drop gate signals. Each cfg needs its own server.

### Merging cross-cfg results

After all three clients finish, aggregate the JSONLs into a `(cfg, τ, n)`-level CSV with `merge_step3_cfgs`. The script dedupes by `(cfg, ep, τ, n)` (last write wins on retries).

```bash
uv run python -m exp.trajectory_deviation.merge_step3_cfgs \
    --jsonl exp/trajectory_deviation/data/step3/clip_w7_d4/results.jsonl \
    --jsonl exp/trajectory_deviation/data/step3/spatial16_w8_d4/results.jsonl \
    --jsonl exp/trajectory_deviation/data/step3/max_pool_w3_d5/results.jsonl \
    --out exp/trajectory_deviation/data/step3/summary.csv
```

`summary.csv` fields: `cfg, tau, n, episodes, success_rate, mean_inference_ratio, std_inference_ratio` (population std, `ddof=0`).

### Quick verification

```bash
wc -l exp/trajectory_deviation/data/step3/*/results.jsonl
```

Per-cfg line count should be ≈ `len(episodes(cfg)) × |tau_grid| × |n_grid|` (e.g. clip_w7_d4 ≈ 159 × 4 × 5 = 3180; resume retries may inflate slightly — `merge_step3_cfgs` dedupes).

---

## Tuning cheat sheet (summary)

Ordered by "how core + how often you'd sweep", so you can decide what to vary during experiment design.

### Core experiment knobs

| Param | Step | Role | Typical values |
|-------|------|------|----------------|
| `--configs` | 2/3 | Cache configs to compare | `clip_w7_d4 spatial16_w8_d4 max_pool_w3_d5` (all) |
| `--M` (Phase 1) | 2 | Background L2 noise sample count | 10 / 20 / 30 |
| `--tau-grid` | 3 | deviate_score threshold triggering search (comma-separated) | `3,5,7,10` (default) |
| `--n-grid` | 3 | Burst length: cycles of cache after a search fires | `1,2,3,5,10` (default) |

### Concurrency / performance

| Param | Step | Role | Guidance |
|-------|------|------|----------|
| `--num-workers` | 1a/2 | Within-cfg worker count | Bounded by server GPU; 4–5 safe |
| `--num-workers` | 3 | Within-cfg LIBERO env concurrency | **Cap at 5** (MuJoCo EGL hard limit) |

### Plan-locked anchors (don't touch casually)

| Param | Value | Reason |
|-------|-------|--------|
| GT bundle | `inference_max_pool_w3_d5.yaml` | three cfgs share the same GT; any AlwaysSkip bundle is valid, and max-pool inference avoids CLIP |
| `--floor` (deviate_score) | 0.1 | plan §10.2 empirical floor, prevents denominator blowup |
| `--seed` (Step 1b) | 7 | plan §9.2 default; Phase 1 statistics are built on it |
| `--seed` (Step 1a) | 42 | Distinct from data-collection seed=7 to avoid overfitting |
| LIBERO max_steps | 220 | libero_spatial hard limit |

---

## Typical directory layout

```
exp/trajectory_deviation/data/
├── cache_eval_results.json                # Step 1a aggregate
├── inits/
│   ├── <task>.init
│   ├── <task>.init_map.json
│   └── step1b_filter.json
├── gt/task_{id}/episode_{subset_idx}.h5   # client-side GT
├── deviate_scores/
│   ├── bg_{cfg}.jsonl
│   ├── cache_{cfg}.jsonl
│   ├── phase1_state_{cfg}.json
│   ├── phase2_state_{cfg}.json
│   └── deviate_score_{cfg}.json
└── step3/
    ├── clip_w7_d4/
    │   ├── run_state.json
    │   └── results.jsonl
    ├── spatial16_w8_d4/
    │   ├── run_state.json
    │   └── results.jsonl
    ├── max_pool_w3_d5/
    │   ├── run_state.json
    │   └── results.jsonl
    └── summary.csv                     # produced by merge_step3_cfgs
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Step 3 aborts with `has no episodes` | `deviate_score_{cfg}.json` empty or wrong path | Confirm Step 2 outputs were merged into `exp/trajectory_deviation/data/deviate_scores/` and the filename `deviate_score_{cfg}.json` matches `--cfg` |
| Step 3 fails with `--num-workers=N exceeds MuJoCo EGL cap` | Per-host libero env concurrency capped at 5 | Lower `--num-workers` to ≤5; for more throughput add another eval host |
| Step 2 deviate scores all ≈ 1.0 | Phase 1 and Phase 2 read the same bundle | Concurrent drivers raced on `load_cache_config`. Kill parallel drivers, run serially, or shard (one server per cfg + `--skip-config-switch`) |
| Step 1b reports many `inference_failed=True` | Server dead or not on the selected AlwaysSkip GT bundle | Check server health; if you used `--skip-config-switch`, preload `inference_max_pool_w3_d5.yaml` or an equivalent AlwaysSkip bundle by hand |
| Step 1a `main.py` complains about `VIRTUAL_ENV` | uv env vars leaked into conda subprocess | `_build_subprocess_cmd` should strip them; audit any custom `--conda-env` wrapper |
| `load_cache_config` errors | Server missing `--concurrent`; or YAML `preload_path` does not exist on the server | Restart with `--concurrent`; verify artifact paths on the GPU server |

---

## Cross-references

- Server startup context, artifact building, CP1 experiment: [cp1_cache.md](cp1_cache.md)
- Cache-system components, YAML fields: [../cache/tutorial.md](../cache/tutorial.md)
- Remote inference topology: [../deployment/libero.md](../deployment/libero.md)
- Original plan and reviews: [../../logs/trajectory_deviation_experiment_plan.log.md](../../logs/trajectory_deviation_experiment_plan.log.md), [../../logs/trajectory_deviation_corrective_experiment.log.md](../../logs/trajectory_deviation_corrective_experiment.log.md)
