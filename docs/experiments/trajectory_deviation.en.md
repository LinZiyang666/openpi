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
Step 1b  run_step1b_gt.py               → GT HDF5 (AlwaysSkip, locked to clip_w7_d4 bundle)
Step 2   compute_deviate_scores         → deviate_score_{cfg}.json (Phase1 M× + Phase2 1× + Phase3 aggregate)
Step 3   run_spawn_experiment           → spawn_state_{cfg}.json + spawn_aggregate.csv
Step 4   analyze_deviation_results      → figures/*.png
```

## Topology (same as CP1 cache experiment)

```
┌─────────────────────────┐        frp tunnel / LAN         ┌────────────────────────┐
│  GPU server (no public) │ ◄──────────────────────────────│  LIBERO eval host       │
│  serve_policy.py         │   155.98.36.13:9000             │  exp/trajectory_        │
│  listens on :8000        │   → localhost:8000              │  deviation/*.py          │
│  --concurrent --collect  │                                  │  examples/libero/main   │
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
| Step 1b GT collection | **not parallel** | each unit spawns its own `main.py` + libero env; runner uses serial `run()` (not `parallel_run()`) | parallel subprocesses would fight over GPU/RAM; bundle only switched once (`inference_clip_w7_d4.yaml`) |
| Step 2 Phase 1 / Phase 2 | episode × sample (`--num-workers`) | configs serialise; inside one config Phase1 → Phase2 is sequential | driver issues `send_load_cache_config` between phases — concurrent drivers would race |
| Step 2 Phase 3 (aggregate) | — | single process, pure numpy | one output per cfg |
| Step 3 Spawn | episode × (s,n,k) (`--num-workers`) | configs serialise | SpawnRunner then BaselineRunner run sequentially inside the same cfg |
| Step 4 analysis | — | single process matplotlib (`Agg`) | no X server required |

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
5. All 6 YAMLs under `configs/cache_runs/deviate_exp/` present, with `preload_path` pointing to the server-side artifact locations.

---

## Step 0 — Start the GPU server (once, fully configured)

One server must survive the whole pipeline; pass every option up front. Each downstream driver calls `send_load_cache_config` to switch bundles, so no restarts are needed.

```bash
uv run scripts/serve_policy.py \
    --concurrent \
    --cache_config configs/cache_runs/deviate_exp/inference_clip_w7_d4.yaml \
    --collect \
    --collect-dir data/deviation_experiment/collected \
    --collect-images \
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
| `--cache_config` | Initial bundle at startup | Any of the 6 YAMLs works; preloading `inference_clip_w7_d4.yaml` lets Step 1b legally use `--skip-config-switch` |
| `--collect` + `--collect-dir` | Write per-episode intermediates (incl. `clean_action`) to HDF5 | **Yes.** Step 3 prefill only reads the server-side `clean_action`; missing `--collect` will make Step 3 raise `FileNotFoundError` |
| `--collect-images` | Also persist raw images (CLIP KeyBuilder needs them) | **Yes.** All 3 `deviate_exp/` bundles use `key_builder.type: clip`, so this must be on |
| `--env LIBERO` | Select env branch | Fixed |
| `--port 8000` | Listener | Match the client's `--port`; if you tunnel via frp, clients hit the remapped external port (e.g. `9000`) |
| `policy:checkpoint --policy.config pi05_libero --policy.dir <local>` | Use the local checkpoint instead of GCS | Strongly recommended |

Sanity check from the eval host:

```bash
curl http://<server-host>:<server-port>/healthz
# → OK
```

> Step 0 is the same as `cp1_cache.md` Step 4 plus the three `--collect*` flags. If you have already run CP1, just restart with those extras.

---

## Step 1a — Baseline cache evaluation

Goal: per-episode `success` flags, to identify episodes where the cache actually breaks.

```bash
uv run exp/cache_experiment/run_cache_experiments.py \
    --yaml-dir configs/cache_runs/deviate_exp \
    --task-suite libero_spatial \
    --host <server-host> --port <server-port> \
    --episodes-per-run 50 \
    --num-workers 4 \
    --seed 42 \
    --conda-env libero_sim \
    --state-path data/deviation_experiment/step1a_state.json
```

Flag reference & tuning:

| Flag | Meaning | Tuning guidance |
|------|---------|-----------------|
| `--yaml-dir` | Directory of YAMLs the driver iterates | `configs/cache_runs/deviate_exp` (6 files; the 3 `inference_*` are AlwaysSkip controls — fine to include, their success_rate will mirror pure inference) |
| `--episodes-per-run` | Episodes per task; LIBERO has 50 inits per task | 50 for a clean failure set; 10 for a quick smoke run |
| `--num-workers` | Task-level parallelism, workers share one server | 4 is safe with `--concurrent`; raise to 8 on bigger GPUs |
| `--seed` | Seed forwarded to `main.py`, controls libero env randomness | Fix at 42 for reproducibility; deliberately different from data-collection seed=7 |
| `--conda-env` | `main.py` must run inside the libero env | Fixed `libero_sim` |
| `--state-path` | Resume state file | Point into the experiment dir so `--resume` works across sessions |

Output: `<yaml-dir>/cache_eval_results.json`. Symlink/copy it into the experiment root so Step 1b-pre finds it:

```bash
cp configs/cache_runs/deviate_exp/cache_eval_results.json \
   data/deviation_experiment/cache_eval_results.json
```

---

## Step 1b-pre — Dump failed inits (offline single-process, seconds)

```bash
uv run python scripts/dump_step1a_failed_inits.py \
    --step1a-results data/deviation_experiment/cache_eval_results.json \
    --task-suite libero_spatial \
    --out-dir data/deviation_experiment/inits
```

Flag reference:

| Flag | Meaning | Tuning guidance |
|------|---------|-----------------|
| `--step1a-results` | Aggregated Step 1a JSON | Must be the deduped `cache_eval_results.json`, not a single-run log |
| `--task-suite` | LIBERO suite name; used to look up `task.name` and init states | Same as Step 1a |
| `--out-dir` | Artifact root | `data/deviation_experiment/inits`, referenced by later steps |
| `--no-torch` | Skip `.init` tensor write, emit only JSON | CI smoke only; don't pass in production runs |

Outputs:
- `<out>/<task_name>.init` — `(K, 92)` subset tensor of failed inits
- `<out>/<task_name>.init_map.json` — subset → orig index map
- `<out>/step1b_filter.json` — fed to the next stage's `--episode-filter`

---

## Step 1b — GT trajectory collection (**sequential, AlwaysSkip locked**)

Plan §18.B4.3 pins the GT bundle to `inference_clip_w7_d4.yaml` — all three Step 2 configs must share the same GT or the deviate-score denominator stops being comparable. The runner switches the bundle once at startup and then serialises every unit through `main.py` (each unit is its own libero env subprocess).

```bash
uv run python -m exp.trajectory_deviation.run_step1b_gt \
    --inits-dir data/deviation_experiment/inits \
    --out-dir data/deviation_experiment/gt \
    --task-suite libero_spatial \
    --host <server-host> --port <server-port> \
    --seed 7 \
    --conda-env libero_sim \
    --resume
```

Flag reference:

| Flag | Meaning | Tuning guidance |
|------|---------|-----------------|
| `--inits-dir` | Step 1b-pre output dir | Required; driver reads `step1b_filter.json` from here |
| `--out-dir` | GT HDF5 root (forwarded to `main.py --save-trajectory-dir`) | Must match Step 2's `--gt-dir`; typically `data/deviation_experiment/gt` |
| `--task-suite` | LIBERO suite | Same as earlier steps |
| `--host / --port` | Server endpoint | frp mapped port on eval host; `localhost:8000` locally |
| `--seed` | Seed forwarded to `main.py` | **Fix at 7** (plan §9.2 default); Phase 1 background noise estimation is built on this seed. Changing it would de-align the M samples from the GT distribution |
| `--conda-env` | libero env | Fixed `libero_sim` |
| `--inference-yaml` | GT bundle YAML path | **Do not change** from `inference_clip_w7_d4.yaml` — changing violates plan §18.B4.3 |
| `--state-path` | Runner state file | Default `<inits-dir>/step1b_state.json`; pair with `--resume` |
| `--resume` | Resume from state | Almost always on |
| `--skip-config-switch` | Skip `load_cache_config` | Only when sharding across servers and you switched bundles by hand |

Tuning notes:

- **Faster?** You can't — the runner is serial by design. Options are (a) raise `max_retries` only if units flake out, or (b) rewrite the runner around `parallel_run` + per-worker env, which plan doesn't authorise.
- **One unit keeps failing?** Raise `max_retries` (default 2) or reproduce it manually by invoking `examples/libero/main.py` with that single `(task_id, orig_init)` for debugging.
- Outputs: `gt/task_{id}/episode_{subset_idx}.h5`, mirrored to `collected/libero_spatial/task_{id}/episode_{subset_idx}.h5` on the server.

---

## Step 2 — Deviate score (parallel within a cfg, serial across cfgs)

Run all three cfgs at once (the driver switches bundles in order):

```bash
uv run python -m exp.trajectory_deviation.compute_deviate_scores \
    --configs clip_w7_d4 spatial16_w8_d4 max_pool_w3_d5 \
    --gt-dir data/deviation_experiment/gt \
    --out-dir data/deviation_experiment/deviate_scores \
    --M 20 \
    --num-workers 4 \
    --host <server-host> --port <server-port> \
    --floor 0.1 \
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
| `--out-dir` | — | Root for jsonl + state + `deviate_score_*.json` | `data/deviation_experiment/deviate_scores` |
| `--M` | 20 | Phase 1 stochastic-sample count used to estimate background L2 noise | **The SNR knob.** Larger M → more stable bg_l2 estimate, Phase 1 time grows linearly. Plan §10.2 reference is 20; 10 for smoke; 30 if you want tighter confidence |
| `--num-workers` | 4 | Within-cfg worker count (server needs `--concurrent`) | Bounded by server GPU capacity; 4 safe, 8 on bigger cards |
| `--host / --port` | `localhost:8000` | Server endpoint | Match Step 0 |
| `--floor` | 0.1 | Denominator floor: `max(bg_l2, floor)` — prevents division-by-tiny-bg | **Avoid changing** (plan §10.2 empirical value); if M is small and bg_l2 frequently drops below 0.1, experiment with 0.05 to inspect distribution |
| `--config-yaml-dir` | `configs/cache_runs/deviate_exp` | YAML lookup root | Do not change unless relocating the experiment |
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

## Step 3 — Spawn corrective experiment (parallel within a cfg, serial across cfgs)

```bash
uv run python -m exp.trajectory_deviation.run_spawn_experiment \
    --gt-dir data/deviation_experiment/gt \
    --collected-dir data/deviation_experiment/collected \
    --task-suite-name libero_spatial \
    --deviate-score-dir data/deviation_experiment/deviate_scores \
    --out-dir data/deviation_experiment/spawn \
    --configs clip_w7_d4 spatial16_w8_d4 max_pool_w3_d5 \
    --n-grid 1 3 5 10 20 \
    --k-grid 1 3 5 \
    --max-spawn-env-steps 300 \
    --num-workers 4 \
    --baselines random equidistant \
    --random-seed 0 \
    --host <server-host> --port <server-port> \
    --resume
```

Flag reference & tuning:

| Flag | Default | Meaning | Tuning guidance |
|------|---------|---------|-----------------|
| `--gt-dir` | — | Client-side GT HDF5 root (Step 1b output) | Required |
| `--collected-dir` | — | **Server-side** `--collect-dir`; driver reads `clean_action` for prefill | Required. If the eval host and server don't share a filesystem, sync `collected/` over, or run the driver on the server |
| `--task-suite-name` | — | Must equal Step 1a/1b value | Required |
| `--deviate-score-dir` | — | Step 2 output root; driver reads `deviate_score_{cfg}.json` | Required |
| `--out-dir` | — | Spawn state + aggregate CSV root | `data/deviation_experiment/spawn` |
| `--configs` | — | cfg list (same as Step 2) | Usually all three |
| `--D` | parsed from `_dN` in cfg name | Trajectory depth controlling how far back prefill reads | Leave unless you know the artifact depth diverges from the cfg name |
| `--n-grid` | `1 3 5 10 20` | Cycles of GT rolled out before teleport | **Core knob.** Small n = teleport close to the crisis (strictest recovery test); large n = let GT carry the episode then hand off to cache. Coarse exploration: `1 5 20` |
| `--k-grid` | `1 3 5` | Top-k highest-deviate points per episode (each is an independent spawn) | **Second core knob.** k=1 tests the single worst point; k=5 covers more. Ablation: `1 2 3 5 7` |
| `--max-spawn-env-steps` | 300 | Per-unit env.step budget after teleport | libero episodes ≤220 steps; 300 leaves ~30% margin. Tight-resource runs can drop to 220 at the cost of boundary-case failures |
| `--num-workers` | 4 | Within-cfg parallelism | Same as Step 2's rule of thumb |
| `--baselines` | `[]` | Extra baseline strategies after SpawnRunner: `random`, `equidistant` | At least `random` to argue "top-k deviate is useful"; both for the full figure set |
| `--random-seed` | 0 | Seed for the random baseline's k-selection | Keep fixed; sweep 0/1/2 for seed sensitivity if needed |
| `--skip-config-switch` | off | Skip `load_cache_config` | Only when sharding |
| `--config-yaml-dir` | `configs/cache_runs/deviate_exp` | YAML root | Leave as is |
| `--resume` | off | Resume | Almost always on |

`(n, k)` combinations = `len(n-grid) × len(k-grid)` = 5×3=15 by default; multiplied by episodes × cfgs this is the slowest step. Recipes:

- **Smoke**: `--n-grid 1 5 --k-grid 1 3 --configs clip_w7_d4 --baselines ""` — 4 combos × single cfg, validates teleport + prefill plumbing end-to-end.
- **Full run**: default 5×3 + all 3 cfgs + `random equidistant` baselines. Needed for plan §12's four figures.
- **Out of GPU?** Drop `--num-workers` first; only then trim `--n-grid` / `--k-grid`.

⚠️ Same rule: don't parallelise multiple `run_spawn_experiment` drivers against one server (bundle race, as in Step 2).

---

## Step 4 — Analysis and plots (single process, seconds)

```bash
uv run python -m exp.trajectory_deviation.analyze_deviation_results \
    --deviate-score-dir data/deviation_experiment/deviate_scores \
    --spawn-csv data/deviation_experiment/spawn/spawn_aggregate.csv \
    --out-dir data/deviation_experiment/figures \
    --configs clip_w7_d4 spatial16_w8_d4 max_pool_w3_d5 \
    --n-threshold 3
```

Flag reference:

| Flag | Default | Meaning | Tuning guidance |
|------|---------|---------|-----------------|
| `--deviate-score-dir` | — | Step 2 output root | Required |
| `--spawn-csv` | — | Step 3 aggregate CSV | Required |
| `--out-dir` | — | Figure output dir | `data/deviation_experiment/figures` |
| `--configs` | — | cfgs to plot (must exist in JSON/CSV) | Usually all three |
| `--n-threshold` | 3 | Close-in spawn cutoff defining a "true failure cycle" — only spawns with `n ≤ threshold` that fail count | **ROC sensitivity knob.** Smaller = stricter (only near-crisis failures); curve becomes sparser. Default 3; raise to 5 for a more permissive coverage curve |

Four figures per config: deviate-score histogram, top-k coverage (ROC-like), `(n, k_idx)` success-rate heatmap, and top-k vs random vs equidistant comparison at the best cell.

---

## Tuning cheat sheet (summary)

Ordered by "how core + how often you'd sweep", so you can decide what to vary during experiment design.

### Core experiment knobs

| Param | Step | Role | Typical values |
|-------|------|------|----------------|
| `--configs` | 2/3/4 | Cache configs to compare | `clip_w7_d4 spatial16_w8_d4 max_pool_w3_d5` (all) |
| `--M` (Phase 1) | 2 | Background L2 noise sample count | 10 / 20 / 30 |
| `--n-grid` | 3 | Cycles of GT before teleport | `1 3 5 10 20` (default) / `1 5 20` (coarse) |
| `--k-grid` | 3 | Top-k deviate points per episode | `1 3 5` (default) / `1 2 3 5 7` (ablation) |
| `--baselines` | 3 | Control strategies | `random equidistant` (full runs) |
| `--n-threshold` | 4 | ROC "true failure" cutoff | 3 (default) / 5 (looser) |

### Concurrency / performance

| Param | Step | Role | Guidance |
|-------|------|------|----------|
| `--num-workers` | 1a/2/3 | Within-cfg worker count | Bounded by server GPU; 4 safe, 8 possible |
| `--max-spawn-env-steps` | 3 | Per-unit env.step budget | 300 default; libero's 220 + 30% margin |

### Plan-locked anchors (don't touch casually)

| Param | Value | Reason |
|-------|-------|--------|
| GT bundle | `inference_clip_w7_d4.yaml` | plan §18.B4.3: three cfgs share the same GT |
| `--floor` (deviate_score) | 0.1 | plan §10.2 empirical floor, prevents denominator blowup |
| `--seed` (Step 1b) | 7 | plan §9.2 default; Phase 1 statistics are built on it |
| `--seed` (Step 1a) | 42 | Distinct from data-collection seed=7 to avoid overfitting |
| LIBERO max_steps | 220 | libero_spatial hard limit |

---

## Typical directory layout

```
data/deviation_experiment/
├── cache_eval_results.json                # Step 1a aggregate
├── inits/
│   ├── <task>.init
│   ├── <task>.init_map.json
│   └── step1b_filter.json
├── gt/task_{id}/episode_{subset_idx}.h5   # client-side GT
├── collected/libero_spatial/task_{id}/episode_{subset_idx}.h5  # server-side clean_action
├── deviate_scores/
│   ├── bg_{cfg}.jsonl
│   ├── cache_{cfg}.jsonl
│   ├── phase1_state_{cfg}.json
│   ├── phase2_state_{cfg}.json
│   └── deviate_score_{cfg}.json
├── spawn/
│   ├── spawn_state_{cfg}.json
│   ├── spawn_state_random_{cfg}.json
│   ├── spawn_state_equidistant_{cfg}.json
│   └── spawn_aggregate.csv
└── figures/{hist,coverage,heatmap,strategy}_{cfg}.png
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Step 3 `FileNotFoundError: collected/.../clean_action` | Server started without `--collect` / `--collect-images` | Restart server with the full Step 0 flags; rerun failed units via `--resume` |
| Step 2 deviate scores all ≈ 1.0 | Phase 1 and Phase 2 read the same bundle | Concurrent drivers raced on `load_cache_config`. Kill parallel drivers, run serially, or shard (one server per cfg + `--skip-config-switch`) |
| Step 1b reports many `inference_failed=True` | Server dead or not on `inference_clip_w7_d4.yaml` | Check server health; if you used `--skip-config-switch`, preload the GT bundle by hand |
| Step 4 coverage curve empty | No failure spawns within `n <= n_threshold` for this cfg | Raise `--n-threshold` or confirm Step 3 swept the full `n-grid` |
| Step 1a `main.py` complains about `VIRTUAL_ENV` | uv env vars leaked into conda subprocess | `_build_subprocess_cmd` should strip them; audit any custom `--conda-env` wrapper |
| `load_cache_config` errors | Server missing `--concurrent`; or YAML `preload_path` does not exist on the server | Restart with `--concurrent`; verify artifact paths on the GPU server |

---

## Cross-references

- Server startup context, artifact building, CP1 experiment: [cp1_cache.md](cp1_cache.md)
- Cache-system components, YAML fields: [../cache/tutorial.md](../cache/tutorial.md)
- Remote inference topology: [../deployment/libero.md](../deployment/libero.md)
- Original plan and reviews: [../../logs/trajectory_deviation_experiment_plan.log.md](../../logs/trajectory_deviation_experiment_plan.log.md), [../../logs/trajectory_deviation_corrective_experiment.log.md](../../logs/trajectory_deviation_corrective_experiment.log.md)
