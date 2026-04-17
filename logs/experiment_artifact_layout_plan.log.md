# Experiment Artifact Layout Plan

**Status**: `Plan`
**Level**: L2 (process + cross-directory reorganization)
**Authority**: Execution
**Author**: @LinZiyang666 / assistant
**Created**: 2026-04-17
**Goal**: Establish a single, enforced layout for experiment **scripts**, **configs**, **run artifacts** (logs / JSON / CSV / plots / reports), and **data sources** across the repository, and migrate the existing scattered files to conform.

---

## 1. Why

The repo has accumulated five distinct storage patterns for experiment-related files with no enforced convention:

1. `exp/` — python scripts for running/analyzing experiments (**clean**).
2. `configs/cache_runs/<exp>/` — YAML configs **mixed with** run logs, JSON state, JSON episode results, PNG/PDF plots, analysis `.md`, and ad-hoc analysis `.py`.
3. `data/<exp>/` — data sources **mixed with** evaluation JSONs, aggregate CSVs, plot PNGs, and timestamped `_pre_cleanup/` snapshots.
4. `exp_outputs/` — a separate output tree used by `qdrant_step_knn` only.
5. Repo root — `pi0.5vla.pdf`, `cache.yaml`, four ad-hoc scripts (`cmd.sh`, `convert.py`, `simple_pytorch_train.py`, `run_step4_tests.sh`).

Downstream consequences already observed:
- **~225 MB of ignored-but-on-disk** `.log` / `.png` / `.pdf` inside `configs/` (143 `.log`, 9 `.pdf`, 7 `.png`).
- Run-state files are scattered: e.g. 5 `cache_eval_results*.json` plus 4 `step1a_state*.json` at the root of `data/deviation_experiment/`, not under a per-run folder.
- `data/deviation_experiment/_pre_cleanup/20260414_115059/` is a manually-dated snapshot, indicating a previous ad-hoc reorg with no written rule.
- Root-level scripts and a 16 MB paper PDF make "where does new work go?" ambiguous for every new contributor.

**Out of scope for this plan** (will be addressed separately if at all):
- Repo root files (`cache.yaml`, `pi0.5vla.pdf`, `convert.py`, `cmd.sh`, `simple_pytorch_train.py`, `run_step4_tests.sh`) — owner directive.
- `assets/pi05_libero/` — model assets, ignored.
- Source code reorganization (`src/`, `tests/`, `packages/`).

**In scope (owner directive, 2026-04-17)**: the entire top-level `data/` tree migrates under `exp/`. Shared data → `exp/common/data/`; experiment-private data → `exp/<experiment>/data/`. This includes `data/db/`, `data/db_init/`, `data/cache_artifacts/`, `data/libero/`, `data/aloha_sim/`, `data/deviation_experiment/`, `data/warm_start_exp/`.

**Tracking-policy clarification (owner G1 R2 directive, 2026-04-17)**: the 24 currently-tracked experiment result JSONs under `configs/cache_runs/` (`experiment_state.json` × 3, `cache_eval_results.json` × 3, `*.episode_results.json` × 18) are **intentionally tracked** — these results are meant to be synced across machines via git. This plan only relocates them; it does not change tracking policy. All 24 use `git mv` to preserve history and stay tracked at their new paths.

---

## 2. Complete Inventory of Scattered Files

Snapshot taken 2026-04-17. "Tracked" = in `git ls-files`. "Ignored" = matched by `.gitignore` and not tracked.

### 2.1 Repo Root

| Path | Size | Status | Nature | Proposed home |
|------|------|--------|--------|---------------|
| `pi0.5vla.pdf` | 16 MB | tracked | Reference paper | `docs/papers/pi0.5vla.pdf` or git-LFS / gitignore + manual |
| `cache.yaml` | 2.4 KB | tracked | **Runtime** cache config for `serve_policy` | `configs/cache.yaml` |
| `cmd.sh` | 618 B | tracked | Ad-hoc training launch script | `scripts/` or delete if stale |
| `convert.py` | 1.8 KB | tracked | LIBERO HDF5 → LeRobot converter | `scripts/convert_libero_hdf5.py` |
| `simple_pytorch_train.py` | 6.6 KB | tracked | Minimal training example | `examples/` |
| `run_step4_tests.sh` | 1.1 KB | tracked | Step 4 test runner | `scripts/` or delete if superseded |

### 2.2 `configs/cache_runs/<exp>/` — configs mixed with run artifacts

| Extension | Count | Nature | Expected location |
|-----------|-------|--------|-------------------|
| `*.yaml` | 166 | Run configs (inputs) | **stay** in `configs/` |
| `*.log` | 143 | Stdout per run (phase1: 32, trajectory: 60, temporal_prune: 48, warm_start: 3) | **move** to run dir |
| `*.json` | 24 | `experiment_state.json`, `cache_eval_results.json`, `*.episode_results.json` | **move** to run dir |
| `*.pdf` | 9 | Analysis plots (`phase1_analysis.pdf`, `tp_heatmaps.pdf`, …) | **move** to run dir |
| `*.png` | 7 | Analysis plots (`tp_bars.png`, `trajectory_results.png`, …) | **move** to run dir |
| `*.md` | 4 | Inline analysis reports (`step1a_analysis.md`, `phase1_analysis.md`, `trajectory_analysis.md`, `tp_analysis.md`) | **move** to report dir |
| `*.py` | 4 | Inline analysis scripts (e.g. `plot_step1a_failure_venn.py`) | **move** to `exp/<pkg>/` |

Sub-tree breakdown:

| Sub-dir | YAML | LOG | JSON | PNG/PDF | MD | PY | Disk |
|---------|------|-----|------|---------|----|----|------|
| `deviate_exp/` | 9 | 0 | 0 | 1+1 | 1 | 1 | 432 KB |
| `phase1/` | 40 | 32 | 1 | 1+2 | 1 | 0 | 51 MB |
| `temporal_prune/` | 48 | 48 | 1 | 3+3 | 1 | 1 | 82 MB |
| `trajectory/` | 60 | 60 | 1 | 2+2 | 1 | 1 | 92 MB |
| `warm_start_exp/` | 9 | 3 | 21 | 0 | 0 | 0 | 2.0 MB |

### 2.3 `data/deviation_experiment/` — sources + evaluations + snapshots

| Path | Kind | Size | Notes |
|------|------|------|-------|
| `cache_eval_results.json` | eval output | root-level | flat; no run_id |
| `cache_eval_results_cache_fail.json` | eval output | root-level | ad-hoc suffix |
| `cache_eval_results_clip.json` | eval output | root-level | per-config slice |
| `cache_eval_results_maxpool.json` | eval output | root-level | per-config slice |
| `cache_eval_results_spatial16.json` | eval output | root-level | per-config slice |
| `step1a_state.json` | run state | root-level | no run_id |
| `step1a_state_clip.json` | run state | root-level | per-config |
| `step1a_state_max_pool.json` | run state | root-level | per-config |
| `step1a_state_spatial16.json` | run state | root-level | per-config |
| `step1a/` | **empty** | 4 KB | dead dir |
| `step1a_runs/` | run dir (6 files: yaml, log, 2 episode jsons, state, eval) | 1.5 MB | **only one per-run dir in whole tree** |
| `inits/` | shared Step-1a init pickles + filter jsons (**tracked** `*.init`) | — | cross-run shared, intentionally tracked |
| `step1b_inits/` | shared Step-1b init maps | — | shared |
| `collected/libero_spatial/` | raw trajectories | 703 MB | data source |
| `gt/task_{0..9}/` | ground-truth trajectories | 1.8 GB | data source |
| `gt_trajectories/task_{0,1}/` | ground-truth trajectories (partial) | 16 MB | **duplicate namespace** vs `gt/` |
| `deviate_scores/` | step-2 outputs | — | eval output |
| `step2_deviate_scores/` | step-2 outputs (full pipeline) | — | eval output, **overlaps** `deviate_scores/` |
| `step3/{clip_w7_d4, max_pool_w3_d5, spatial16_w8_d4, plots}/` | step-3 outputs | 7.3 MB | eval output |
| `spawn_dry_run/` | step-2 dry run csv + state | 16 KB | eval output |
| `_pre_cleanup/20260414_115059/` | manual snapshot | 28 KB | archived by hand |

### 2.4 `data/warm_start_exp/`

| Path | Kind | Notes |
|------|------|-------|
| `baseline_failures.json` | input list | shared across sweep |
| `state_full_{clip,max_pool,spatial16}.json` | run state | per-keybuilder |
| `results/success_rate_sweep.png` | eval output | only 1 plot; rest of plots live in `configs/cache_runs/warm_start_exp/` — **split across trees** |
| `timing/` | **empty** | dead dir |

### 2.5 `exp_outputs/`

| Path | Notes |
|------|-------|
| `exp_outputs/qdrant_step_knn_example/{config.json,details.csv,summary.csv,report.md}` | Example output for one experiment only; parallel convention to `data/*/` — redundant. |

### 2.6 Ambiguous Scripts

| Path | Nature | Signal |
|------|--------|--------|
| `scripts/verify_step2.py` | Step-2 one-shot verification | tied to one experiment; docstring references `logs/` |
| `scripts/verify_env_save_restore.py` | Phase 0 smoke for deviation experiment | explicitly references `logs/trajectory_deviation_corrective_implementation.log.md` |
| `scripts/verify_restore_obs_equivalence.py` | Phase 0 smoke for deviation experiment | same |
| `scripts/dump_step1a_failed_inits.py` | Step-1b input generator | docstring calls out cache experiment pipeline |
| `scripts/dump_libero_init_images.py` | LIBERO helper | likely general-purpose |

### 2.7 `.gitignore` Inconsistencies

| Pattern | Status | Issue |
|---------|--------|-------|
| `data/**` with `!data/*.json`, `!data/**/*.json` | active | **permits all JSON under `data/`**, including large result dumps |
| `configs/**/*.log`, `*.png`, `*.pdf` | active | **no rule for `*.json`** — result JSONs get committed |
| `exp_outputs` | active | directory is ignored but directory exists and is used |
| `*.log` with `!logs/`, `!logs/**` | active | correct |

---

## 3. Proposed Layout (owner-specified rules, §3.1–§3.2; mapping §3.3 under discussion)

### 3.1 Canonical Tree (owner-decided, final)

Each experiment is a self-contained unit. `common/` is treated as an experiment too — same sub-directory layout, but hosts cross-experiment shared assets.

```
exp/
  <experiment>/
    __init__.py
    <runner>.py, <helper>.py, ...    # experiment code directly at root (no code/ subdir)
    config/                          # all YAML configs for this experiment
    data/                            # run-produced artifacts: json, jsonl, log, h5
      cache_artifacts/               # pkl / DB artifacts
    analysis/                        # plot scripts + analysis tools + generated png/pdf/analysis.md
  common/                            # shared "experiment"
    <shared *.py>                    # e.g. _subprocess.py, _unit_key.py at root
    config/
    data/
      cache_artifacts/
    analysis/

tests/<exp>/test_*.py                # framework-integrated tests — UNCHANGED, stay here
```

### 3.2 File-Kind Rules (owner-decided, final)

| Kind | Location |
|------|----------|
| Experiment runner / helper / builder `.py` | `exp/<exp>/` root |
| Experiment yaml config | `exp/<exp>/config/` |
| Run-time produced data (`*.json`, `*.jsonl`, `*.log`, `*.h5`) | `exp/<exp>/data/` |
| Pkl / DB / index artifacts | `exp/<exp>/data/cache_artifacts/` |
| Plot scripts, analysis tools, `*.png`, `*.pdf`, analysis `*.md` | `exp/<exp>/analysis/` |
| Cross-experiment shared assets | `exp/common/` (same sub-structure) |
| Smoke / verification scripts integrated with pytest | `tests/<exp>/` (unchanged) |
| Smoke / verification scripts **not** integrated with pytest | alongside experiment code in `exp/<exp>/` root |

### 3.3 Scope

**In scope** (owner directive, 2026-04-17):
- Entire top-level `data/` tree migrates under `exp/`: shared → `exp/common/data/`, private → `exp/<exp>/data/`.
- All of `configs/cache_runs/` migrates into per-experiment `config/` / `data/` / `analysis/` subdirs.
- `exp_outputs/` folded into `exp/<exp>/`.
- `scripts/` entries that docstring-declare experiment ownership migrate per Q4.

**Out of scope**:
- Repo root files (§1): `cache.yaml`, `pi0.5vla.pdf`, `convert.py`, `cmd.sh`, `simple_pytorch_train.py`, `run_step4_tests.sh`.
- `configs/` top-level directory after `cache_runs/` is emptied — fate TBD (§5 F2).
- `assets/`, `src/`, `tests/`, `packages/`.

### 3.4 File-by-File Mapping (FINAL)

#### 3.4.1 `exp/cache_experiment/` dissolution (package removed)

| Source | Target |
|--------|--------|
| `run_cache_experiments.py` | `exp/common/run_cache_experiments.py` |
| `build_in_memory_cache_artifact.py` | `exp/common/build_in_memory_cache_artifact.py` |
| `build_clip_cache_artifact.py` | `exp/common/build_clip_cache_artifact.py` |
| `calibrate_score_sum_stats.py` | `exp/common/calibrate_score_sum_stats.py` |
| `calibrate_robot_state_tau.py` | `exp/common/calibrate_robot_state_tau.py` |
| `generate_cache_run_yamls.py` | `exp/common/generate_cache_run_yamls.py` |
| `analyze_cache_results.py` | `exp/common/analyze_cache_results.py` |
| `analyze_warm_sweep.py` | `exp/warm_start/analyze_warm_sweep.py` |
| `__init__.py` | removed |

Existing `exp/common/` files (`_subprocess.py`, `_unit_key.py`, `_cache_config_rpc.py`, `_run_state_base.py`, `__init__.py`) stay unchanged.

#### 3.4.2 `configs/cache_runs/` dispersal

Column legend: **T** = tracked (use `git mv`), **U** = untracked (use plain `mv`). All tracked items — including the 24 result JSONs — stay tracked after the move; no `git rm` / `git rm --cached` for any of them.

| Source | Target | T/U |
|--------|--------|-----|
| `deviate_exp/*.yaml` (9) | `exp/trajectory_deviation/config/` | T |
| `deviate_exp/plot_step1a_failure_venn.py` | `exp/trajectory_deviation/analysis/plot_step1a_failure_venn.py` | T |
| `deviate_exp/step1a_analysis.md` | `exp/trajectory_deviation/analysis/step1a_analysis.md` | T |
| `deviate_exp/step1a_analysis.pdf`, `step1a_failure_venn.png` | `exp/trajectory_deviation/analysis/` | U |
| `phase1/*.yaml` (40) | `exp/common/config/phase1/` | T |
| `phase1/*.log` (32) | `exp/common/data/phase1/` | U |
| `phase1/experiment_state.json` | `exp/common/data/phase1/experiment_state.json` | T |
| `phase1/plot_results.py` | `exp/common/analysis/phase1/plot_results.py` | T |
| `phase1/phase1_analysis.md` | `exp/common/analysis/phase1/phase1_analysis.md` | T |
| `phase1/phase1_analysis.pdf`, `phase1_results.{pdf,png}` | `exp/common/analysis/phase1/` | U |
| `trajectory/*.yaml` (60) | `exp/common/config/trajectory/` | T |
| `trajectory/*.log` (60) | `exp/common/data/trajectory/` | U |
| `trajectory/experiment_state.json` | `exp/common/data/trajectory/experiment_state.json` | T |
| `trajectory/plot_results.py` | `exp/common/analysis/trajectory/plot_results.py` | T |
| `trajectory/trajectory_analysis.md` | `exp/common/analysis/trajectory/trajectory_analysis.md` | T |
| `trajectory/trajectory_analysis.pdf`, `trajectory_results{,_facets}.{pdf,png}` | `exp/common/analysis/trajectory/` | U |
| `temporal_prune/*.yaml` (48) | `exp/temporal_prune/config/` | T |
| `temporal_prune/*.log` (48) | `exp/temporal_prune/data/` | U |
| `temporal_prune/experiment_state.json` | `exp/temporal_prune/data/experiment_state.json` | T |
| `temporal_prune/plot_results.py` | `exp/temporal_prune/analysis/plot_results.py` | T |
| `temporal_prune/temporal_prune_analysis.md` | `exp/temporal_prune/analysis/temporal_prune_analysis.md` | T |
| `temporal_prune/tp_{bars,heatmaps,lines}.{png,pdf}` | `exp/temporal_prune/analysis/` | U |
| `warm_start_exp/{max_pool,spatial16,clip}/*.yaml` (9) | `exp/warm_start/config/{max_pool,spatial16,clip}/` | T |
| `warm_start_exp/**/*.log` (3) | `exp/warm_start/data/{max_pool,spatial16,clip}/` | U |
| `warm_start_exp/**/cache_eval_results.json` (3) | `exp/warm_start/data/{max_pool,spatial16,clip}/` | T |
| `warm_start_exp/**/*.episode_results.json` (18) | `exp/warm_start/data/{max_pool,spatial16,clip}/` | T |
| `warm_start_exp/max_pool/retry/*.log` (2) | `exp/warm_start/data/max_pool/retry/` | U |

#### 3.4.3 `data/` migration

Analysis outputs (PNG / PDF / analysis `.md`) are routed to `analysis/`; pure run-produced data (JSON / JSONL / CSV / per-run subdirs) stays in `data/` per §3.2.

| Source | Target |
|--------|--------|
| `data/deviation_experiment/cache_eval_results*.json` (5) | `exp/trajectory_deviation/data/` |
| `data/deviation_experiment/step1a_state*.json` (4) | `exp/trajectory_deviation/data/` |
| `data/deviation_experiment/step1a_runs/`, `spawn_dry_run/` | `exp/trajectory_deviation/data/` |
| `data/deviation_experiment/step2_deviate_scores/` (pure JSONL/JSON) | `exp/trajectory_deviation/data/step2_deviate_scores/` |
| `data/deviation_experiment/deviate_scores/deviate_score_*.json` (3) | `exp/trajectory_deviation/data/deviate_scores/` |
| `data/deviation_experiment/deviate_scores/step2_deviate_score_analysis*.md` (2) | `exp/trajectory_deviation/analysis/` |
| `data/deviation_experiment/deviate_scores/plots/*.png`, `plots/*.md`, `plots/*.json` | `exp/trajectory_deviation/analysis/deviate_scores_plots/` |
| `data/deviation_experiment/step3/summary.csv`, `clip_w7_d4/`, `max_pool_w3_d5/`, `spatial16_w8_d4/` | `exp/trajectory_deviation/data/step3/` |
| `data/deviation_experiment/step3/plots/*.png` | `exp/trajectory_deviation/analysis/step3_plots/` |
| `data/deviation_experiment/gt/`, `gt_trajectories/`, `collected/`, `inits/`, `step1b_inits/` | `exp/trajectory_deviation/data/` |
| `data/deviation_experiment/_pre_cleanup/20260414_115059/` | `exp/trajectory_deviation/data/_pre_cleanup/20260414_115059/` |
| `data/deviation_experiment/step1a/` (empty) | delete |
| `data/cache_artifacts/libero_spatial/` | `exp/common/data/cache_artifacts/libero_spatial/` |
| `data/cache_artifacts/libero_spatial_warm/` | `exp/common/data/cache_artifacts/libero_spatial_warm/` |
| `data/warm_start_exp/baseline_failures.json`, `state_full_*.json` | `exp/warm_start/data/` |
| `data/warm_start_exp/results/success_rate_sweep.png` | `exp/warm_start/analysis/` |
| `data/warm_start_exp/timing/` (empty) | delete |
| `data/db/libero_cache/` (3.5 GB) | `exp/common/data/db/libero_cache/` |
| `data/db_init/{libero, libero_cache, libero_cache_image, libero_image}/` | `exp/common/data/db_init/` |
| `data/db_init/sample_cache.py` | `exp/common/data/db_init/sample_cache.py` (stays with its data; no code edits) |
| `data/libero/videos/` | `exp/common/data/libero/videos/` |
| `data/aloha_sim/videos/` | `exp/common/data/aloha_sim/videos/` |
| top-level `data/` directory | **delete** after above moves |

#### 3.4.4 `exp/<package>/` code reshuffle

| Source | Target |
|--------|--------|
| `exp/trajectory_deviation/plot_step3_tradeoff.py`, `plot_step3_tradeoff_total.py`, `plot_step3_heatmaps.py`, `plot_deviate_score_distribution.py` | `exp/trajectory_deviation/analysis/` |
| `exp/trajectory_deviation/analyze_deviation_results.py`, `analyze_step2_deviate_scores.py` | `exp/trajectory_deviation/analysis/` |
| `exp/trajectory_deviation/run_*.py`, `compute_deviate_scores.py`, `merge_step3_cfgs.py`, `_libero_env.py`, `__init__.py` | stay at `exp/trajectory_deviation/` root |
| `exp/temporal_prune/generate_temporal_prune_yamls.py`, `__init__.py` | stay at `exp/temporal_prune/` root |
| `exp/qdrant_step_knn/*.py` | stay at `exp/qdrant_step_knn/` root |
| `exp/qdrant_step_knn/qdrant_step_knn_experiment_config.example.json` | `exp/qdrant_step_knn/config/qdrant_step_knn_experiment_config.example.json` |

#### 3.4.5 `exp_outputs/` migration

| Source | Target |
|--------|--------|
| `exp_outputs/qdrant_step_knn_example/{config.json, details.csv, summary.csv}` | `exp/qdrant_step_knn/data/example/` |
| `exp_outputs/qdrant_step_knn_example/report.md` | `exp/qdrant_step_knn/analysis/example_report.md` |
| top-level `exp_outputs/` directory | delete |

#### 3.4.6 `scripts/` migration

| Source | Target |
|--------|--------|
| `scripts/verify_env_save_restore.py` | `exp/trajectory_deviation/verify_env_save_restore.py` |
| `scripts/verify_restore_obs_equivalence.py` | `exp/trajectory_deviation/verify_restore_obs_equivalence.py` |
| `scripts/verify_step2.py` | `exp/trajectory_deviation/verify_step2.py` |
| `scripts/dump_step1a_failed_inits.py` | `exp/trajectory_deviation/dump_step1a_failed_inits.py` |
| `scripts/dump_libero_init_images.py`, `compute_norm_stats.py`, `serve_policy.py`, `train.py`, `train_pytorch.py`, `train_test.py`, `__init__.py` | stay at `scripts/` |
| `tests/scripts/test_verify_smoke_scripts.py`, `test_dump_step1a_failed_inits.py` | stay at `tests/scripts/` (update `sys.path` / `importlib` target paths to new script locations) |

#### 3.4.7 Out of scope (not moved this round)

- Repo-root files: `cache.yaml`, `pi0.5vla.pdf`, `convert.py`, `cmd.sh`, `simple_pytorch_train.py`, `run_step4_tests.sh`.
- `assets/`, `src/`, `tests/` (except import-path updates in §3.4.6), `packages/`.
- `configs/` directory itself — once `cache_runs/` is emptied, `configs/` becomes empty; delete at end of migration.

### 3.5 `.gitignore` Strategy (FINAL — G1 R2)

Post-migration rules. Old `data/**` and `configs/**/*.{log,png,pdf}` / `exp_outputs` blocks are removed (those trees disappear). New rules:

```diff
- # Data directories.
- assets/
- checkpoints/
- data/**
- !data/**/
- !data/*.json
- !data/**/*.json
- # Step 1a-dump initial states (small torch pickles, ~25KB each, 10 tasks).
- # Tracked so secondary machines can pull them via git instead of scp;
- # they're the index source-of-truth shared by Step 1b/2/3.
- !data/deviation_experiment/inits/*.init
- !data/deviation_experiment/inits/*.pruned_init
+ # Data directories.
+ assets/
+ checkpoints/
+
+ # Experiment run-produced artifacts: ignored by default.
+ exp/**/data/**
+ !exp/**/data/                                            # keep directory stubs
+
+ # Tracked source-of-truth exceptions (cross-machine scp avoidance, shared across step1a/1b/2/3).
+ !exp/trajectory_deviation/data/inits/
+ !exp/trajectory_deviation/data/inits/*.init
+ !exp/trajectory_deviation/data/inits/*.pruned_init
+ !exp/trajectory_deviation/data/inits/per_unit_filters/
+ !exp/trajectory_deviation/data/inits/per_unit_filters/*.json
+
+ # Tracked cache calibration metadata (small JSON, needed for reproducible runs).
+ !exp/common/data/cache_artifacts/
+ !exp/common/data/cache_artifacts/**/
+ !exp/common/data/cache_artifacts/**/calibration.json
+
+ # Tracked experiment result JSONs (owner G1 R2 directive: synced across machines via git).
+ !exp/common/data/phase1/
+ !exp/common/data/phase1/experiment_state.json
+ !exp/common/data/trajectory/
+ !exp/common/data/trajectory/experiment_state.json
+ !exp/temporal_prune/data/
+ !exp/temporal_prune/data/experiment_state.json
+ !exp/warm_start/data/
+ !exp/warm_start/data/clip/
+ !exp/warm_start/data/clip/cache_eval_results.json
+ !exp/warm_start/data/clip/*.episode_results.json
+ !exp/warm_start/data/max_pool/
+ !exp/warm_start/data/max_pool/cache_eval_results.json
+ !exp/warm_start/data/max_pool/*.episode_results.json
+ !exp/warm_start/data/spatial16/
+ !exp/warm_start/data/spatial16/cache_eval_results.json
+ !exp/warm_start/data/spatial16/*.episode_results.json
- ...
- # Ignore ad-hoc logs everywhere, but keep curated project logs under logs/.
- *.log
- !logs/
- !logs/**
+ # Ignore ad-hoc logs everywhere, but keep curated project logs under logs/.
+ *.log
+ !logs/
+ !logs/**
- configs/**/*.log
- configs/**/*.png
- configs/**/*.pdf
+ # configs/ rules removed — tree empties after cache_runs/ migration and the configs/ directory is deleted in Phase 4.
- exp_outputs
+ # exp_outputs removed — tree deleted in Phase 6.
```

**Cache-artifact pkls policy** (resolves F3 in §5): bulk `.pkl` / `.bin` / `.safetensors` remain ignored via `exp/**/data/**`. Only per-directory `calibration.json` is whitelisted above. If future metadata needs tracking, add explicit `!` lines alongside `calibration.json`.

**Result JSON semantics (G1 R2 owner directive)**: the 24 currently-tracked experiment result JSONs stay tracked after the migration. Phase 4 uses `git mv` so git history is preserved and the files continue to appear in `git ls-files`. The `.gitignore` block above whitelists each exact path so that:
- (a) If a future run recreates a deleted file, `git add` will pick it up (not masked by the `exp/**/data/**` ignore).
- (b) No new JSONs created in the same directories are auto-tracked — only the 24 explicit paths are whitelisted.

This is a pure relocation; tracking policy is unchanged from pre-migration.

---

## 4. Migration Steps (executable)

Execution happens in a dedicated branch (off `Ziyang`) and lands as a single large reorg PR after G1 approval. Order is chosen so `git mv` preserves history and tests don't break mid-way.

### Phase 0 — Safety

1. `git tag pre-artifact-reorg-2026-04-17` on current HEAD so nothing is unrecoverable.
2. Create branch `reorg/artifact-layout` off `Ziyang`.
3. Backup `data/db/` (3.5 GB) + `data/cache_artifacts/` (2.8 GB) out-of-tree (these are ignored but large + regenerable is slow).

### Phase 1 — Skeleton

4. Create empty package skeletons:
   - `exp/common/{config,data,analysis}/__init__.py` (only for dirs, not data)
   - `exp/trajectory_deviation/{config,data,analysis}/`
   - `exp/temporal_prune/{config,data,analysis}/`
   - `exp/qdrant_step_knn/{config,data,analysis}/`
   - `exp/warm_start/{config,data,analysis}/__init__.py` (new package)

### Phase 2 — (removed, G1 R2)

Previously untracked the 24 result JSONs. Owner directive (G1 R2 2026-04-17): those files stay tracked; this plan only relocates them via `git mv` in Phase 4. No Phase 2 steps remain.

### Phase 3 — Code moves (per §3.4.1, §3.4.4, §3.4.6)

Each step 5–10 moves code; steps 11a–11h update every path/import/docstring/runbook that references the moved module **before** running the pytest gate in step 13. Nothing is batched for "later sweep" — if it breaks `uv run pytest`, fix it here.

5. `git mv` `exp/cache_experiment/{run_cache_experiments,build_*,calibrate_*,generate_cache_run_yamls,analyze_cache_results}.py` → `exp/common/`.
6. `git mv exp/cache_experiment/analyze_warm_sweep.py exp/warm_start/`.
7. `git rm exp/cache_experiment/__init__.py`; `rmdir exp/cache_experiment/` (owner G1 R2: this single 1-line docstring file may be deleted; empty dir is then removed per existing rule).
8. `git mv exp/trajectory_deviation/{plot_step3_tradeoff,plot_step3_tradeoff_total,plot_step3_heatmaps,plot_deviate_score_distribution,analyze_deviation_results,analyze_step2_deviate_scores}.py exp/trajectory_deviation/analysis/`.
9. `git mv scripts/verify_env_save_restore.py scripts/verify_restore_obs_equivalence.py scripts/verify_step2.py scripts/dump_step1a_failed_inits.py exp/trajectory_deviation/`.
10. `git mv exp/qdrant_step_knn/qdrant_step_knn_experiment_config.example.json exp/qdrant_step_knn/config/`.

**11a — `exp.cache_experiment.*` → `exp.common.*` / `exp.warm_start.*`**: rewrite imports of `run_cache_experiments`, `build_in_memory_cache_artifact`, `build_clip_cache_artifact`, `calibrate_score_sum_stats`, `calibrate_robot_state_tau`, `generate_cache_run_yamls`, `analyze_cache_results` (→ `exp.common.*`) and `analyze_warm_sweep` (→ `exp.warm_start.analyze_warm_sweep`) in every `.py` under `exp/`, `tests/`, `scripts/` (plus docs-mentioned CLI invocations).

**11b — `exp.trajectory_deviation.<analyzer>` → `exp.trajectory_deviation.analysis.<analyzer>`**: rewrite imports/invocations of `analyze_deviation_results`, `analyze_step2_deviate_scores`, `plot_step3_tradeoff`, `plot_step3_tradeoff_total`, `plot_step3_heatmaps`, `plot_deviate_score_distribution` in:
- `tests/exp/test_analyze_deviation_results.py` (currently `import exp.trajectory_deviation.analyze_deviation_results as ana`)
- docstrings inside the moved modules (self-referential `uv run python -m exp.trajectory_deviation.<name>` lines — 5+ sites in `analyze_step2_deviate_scores.py`, 1 each in `plot_step3_*.py`, `analyze_deviation_results.py`)
- `docs/experiments/trajectory_deviation.md` and `.en.md` command blocks if any use these module paths.

**11c — `scripts/verify_*` / `dump_step1a_failed_inits` bootstrap fix**: `scripts/verify_env_save_restore.py:69` (`sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`) and `scripts/verify_restore_obs_equivalence.py:77` (`repo_root = Path(__file__).resolve().parent.parent`) assume parent-of-parent is repo root. After the move to `exp/trajectory_deviation/`, parent-of-parent is `exp/`, not repo root. Patch both to `Path(__file__).resolve().parents[2]` or use `pathlib` with an explicit marker file.

**11d — `tests/scripts/test_verify_smoke_scripts.py`**: update the `sys.path` / `importlib.util.spec_from_file_location` target to `exp/trajectory_deviation/verify_*.py`.

**11e — `tests/scripts/test_dump_step1a_failed_inits.py`**: update `importlib.util.spec_from_file_location` path to `exp/trajectory_deviation/dump_step1a_failed_inits.py`.

**11f — Self-docstring `python -m scripts.*` references**: verify none of the moved `scripts/verify_*.py` / `dump_step1a_failed_inits.py` docstrings or `__doc__` blocks say `python -m scripts.<name>`; rewrite any occurrence to `python -m exp.trajectory_deviation.<name>`.

**11g — `exp/qdrant_step_knn/qdrant_step_knn_experiment_config.example.json`**: grep any path reference (likely in `docs/experiments/*.md` and `exp/qdrant_step_knn/*.py`) and rewrite to `exp/qdrant_step_knn/config/qdrant_step_knn_experiment_config.example.json`.

**11h — Final audit command before pytest**:
```
rg -n '\b(exp\.cache_experiment|scripts\.verify_|scripts\.dump_step1a_failed_inits|exp\.trajectory_deviation\.(analyze_deviation_results|analyze_step2_deviate_scores|plot_step3_tradeoff|plot_step3_tradeoff_total|plot_step3_heatmaps|plot_deviate_score_distribution))\b' -- exp/ tests/ scripts/ docs/ logs/ configs/ cache.yaml
```
Must return zero hits (except inside this plan itself and archived logs).

12. (Renumbered — rolled into 11d/11e above.)

13. Run `uv run pytest` — must pass end of Phase 3. If it fails, go back to 11a–11h; do not advance to Phase 4.

### Phase 4 — configs/ dispersal (per §3.4.2)

Tracked files use `git mv`; untracked files use plain `mv`; T/U flag per the §3.4.2 table. The 24 result JSONs are tracked (T) per owner G1 R2 directive; they stay tracked at their new paths.

14. `git mv configs/cache_runs/deviate_exp/*.yaml exp/trajectory_deviation/config/` (T, 9 files).
15. `git mv configs/cache_runs/deviate_exp/plot_step1a_failure_venn.py exp/trajectory_deviation/analysis/` (T).
16a. `git mv configs/cache_runs/deviate_exp/step1a_analysis.md exp/trajectory_deviation/analysis/` (T).
16b. Plain `mv configs/cache_runs/deviate_exp/step1a_analysis.pdf configs/cache_runs/deviate_exp/step1a_failure_venn.png exp/trajectory_deviation/analysis/` (U).
17. `git mv configs/cache_runs/phase1/*.yaml exp/common/config/phase1/` (T, 40 files).
18a. Plain `mv configs/cache_runs/phase1/*.log exp/common/data/phase1/` (U, 32 files).
18b. `git mv configs/cache_runs/phase1/experiment_state.json exp/common/data/phase1/` (T — tracked result JSON, preserve history).
19. `git mv configs/cache_runs/phase1/plot_results.py exp/common/analysis/phase1/` (T).
20a. `git mv configs/cache_runs/phase1/phase1_analysis.md exp/common/analysis/phase1/` (T).
20b. Plain `mv configs/cache_runs/phase1/phase1_analysis.pdf configs/cache_runs/phase1/phase1_results.pdf configs/cache_runs/phase1/phase1_results.png exp/common/analysis/phase1/` (U).
21. Repeat 17–20 for `trajectory/` → `exp/common/{config,data,analysis}/trajectory/`:
   - 21a. `git mv` 60 `*.yaml` → `exp/common/config/trajectory/` (T).
   - 21b. Plain `mv` 60 `*.log` → `exp/common/data/trajectory/` (U).
   - 21c. `git mv configs/cache_runs/trajectory/experiment_state.json exp/common/data/trajectory/` (T).
   - 21d. `git mv plot_results.py` + `git mv trajectory_analysis.md` → `exp/common/analysis/trajectory/` (T).
   - 21e. Plain `mv trajectory_analysis.pdf trajectory_results.pdf trajectory_results.png trajectory_results_facets.pdf trajectory_results_facets.png` → `exp/common/analysis/trajectory/` (U).
22. Repeat 17–20 for `temporal_prune/` → `exp/temporal_prune/{config,data,analysis}/`:
   - 22a. `git mv` 48 `*.yaml` → `exp/temporal_prune/config/` (T).
   - 22b. Plain `mv` 48 `*.log` → `exp/temporal_prune/data/` (U).
   - 22c. `git mv configs/cache_runs/temporal_prune/experiment_state.json exp/temporal_prune/data/` (T).
   - 22d. `git mv plot_results.py` + `git mv temporal_prune_analysis.md` → `exp/temporal_prune/analysis/` (T).
   - 22e. Plain `mv tp_{bars,heatmaps,lines}.{png,pdf}` → `exp/temporal_prune/analysis/` (U).
23. For `warm_start_exp/{max_pool,spatial16,clip}/` → `exp/warm_start/{config,data}/{max_pool,spatial16,clip}/`:
   - 23a. `git mv` 9 `*.yaml` → `exp/warm_start/config/<variant>/` (T).
   - 23b. Plain `mv` 3 `*.log` → `exp/warm_start/data/<variant>/` (U).
   - 23c. `git mv` 3 `cache_eval_results.json` → `exp/warm_start/data/<variant>/` (T — tracked result JSONs, preserve history).
   - 23d. `git mv` 18 `*.episode_results.json` → `exp/warm_start/data/<variant>/` (T).
   - 23e. Plain `mv configs/cache_runs/warm_start_exp/max_pool/retry/*.log exp/warm_start/data/max_pool/retry/` (U).
24. `rmdir configs/cache_runs/*` once empty; `rmdir configs/cache_runs`; if `configs/` empty, `rmdir configs/`.

### Phase 5 — data/ migration (per §3.4.3)

Analysis-kind files (`*.png`, `*.pdf`, analysis `*.md`) are split out to `analysis/` per §3.2, **not** bulk-moved under `data/`.

25. `git mv data/deviation_experiment/inits exp/trajectory_deviation/data/inits` (preserves tracked `*.init` + `per_unit_filters/*.json`).
26a. Move untracked pure-data trees under `data/deviation_experiment/` → `exp/trajectory_deviation/data/`:
   - `gt/`, `gt_trajectories/`, `collected/`, `step1a_runs/`, `step1b_inits/`, `step2_deviate_scores/`, `spawn_dry_run/`, `_pre_cleanup/20260414_115059/`.
   - Root-level JSONs: `cache_eval_results*.json` (5), `step1a_state*.json` (4).
26b. Split `data/deviation_experiment/deviate_scores/`:
   - `deviate_score_*.json` (3 files) → `exp/trajectory_deviation/data/deviate_scores/` (plain `mv`).
   - `step2_deviate_score_analysis.md`, `step2_deviate_score_analysis_zh.md` → `exp/trajectory_deviation/analysis/` (plain `mv`).
   - `plots/*.png`, `plots/*.md`, `plots/deviate_score_distribution_summary.json` → `exp/trajectory_deviation/analysis/deviate_scores_plots/` (plain `mv`).
   - `rmdir data/deviation_experiment/deviate_scores/plots`; `rmdir data/deviation_experiment/deviate_scores` once empty.
26c. Split `data/deviation_experiment/step3/`:
   - `summary.csv`, `clip_w7_d4/`, `max_pool_w3_d5/`, `spatial16_w8_d4/` → `exp/trajectory_deviation/data/step3/` (plain `mv`).
   - `plots/*.png` → `exp/trajectory_deviation/analysis/step3_plots/` (plain `mv`).
   - `rmdir data/deviation_experiment/step3/plots`; `rmdir data/deviation_experiment/step3` once empty.
27. `rmdir data/deviation_experiment/step1a` (empty).
28. `git mv data/cache_artifacts/libero_spatial exp/common/data/cache_artifacts/libero_spatial` (preserves tracked `calibration.json`); likewise `libero_spatial_warm`. Bulk `.pkl` files inside are untracked and ride along.
29. Plain `mv data/warm_start_exp/baseline_failures.json data/warm_start_exp/state_full_*.json exp/warm_start/data/`.
30. Plain `mv data/warm_start_exp/results/success_rate_sweep.png exp/warm_start/analysis/`.
31. `rmdir data/warm_start_exp/timing` (empty); `rmdir data/warm_start_exp/results`; `rmdir data/warm_start_exp`.
32. Plain `mv data/db exp/common/data/db`.
33. Plain `mv data/db_init exp/common/data/db_init` (keeps `sample_cache.py` bundled with its data, per Q5).
34. Plain `mv data/libero/videos exp/common/data/libero/videos`; `rmdir data/libero` once empty.
35. Plain `mv data/aloha_sim/videos exp/common/data/aloha_sim/videos`; `rmdir data/aloha_sim` once empty.
36. `rmdir data/` once empty.

### Phase 6 — exp_outputs/ migration (per §3.4.5)

37. Move `exp_outputs/qdrant_step_knn_example/{config.json, details.csv, summary.csv}` → `exp/qdrant_step_knn/data/example/`.
38. Move `exp_outputs/qdrant_step_knn_example/report.md` → `exp/qdrant_step_knn/analysis/example_report.md`.
39. `rmdir exp_outputs/`.

### Phase 7 — Path reference sweep

Phase 3 fixed dotted-module imports for the code moves; Phase 7 covers **filesystem-path** references (CLI args, YAML paths, docstring examples, runbook commands) plus any dotted-module references not caught during Phase 3 (e.g. because they live inside moved configs or doc files).

40. Filesystem-path sweep (ripgrep, multiple patterns — not just `data/`):
   ```
   rg -n -e '\bdata/(cache_artifacts|deviation_experiment|warm_start_exp|db|db_init|libero|aloha_sim)/' \
         -e '\bconfigs/cache_runs/' \
         -e '\bexp_outputs/' \
         -e 'scripts/(verify_env_save_restore|verify_restore_obs_equivalence|verify_step2|dump_step1a_failed_inits)\.py' \
         -- src/ exp/ scripts/ tests/ docs/ logs/ configs/ cache.yaml pyproject.toml
   ```
41. Rewrite each match:
   - `data/cache_artifacts/` → `exp/common/data/cache_artifacts/`
   - `data/deviation_experiment/` → `exp/trajectory_deviation/data/` (splitting out analysis-kind sub-paths per §3.4.3 where they appear)
   - `data/warm_start_exp/` → `exp/warm_start/data/`
   - `data/db/`, `data/db_init/`, `data/libero/`, `data/aloha_sim/` → `exp/common/data/…`
   - `configs/cache_runs/<sub>/…` → per §3.4.2 mapping
   - `exp_outputs/qdrant_step_knn_example/` → `exp/qdrant_step_knn/data/example/` (report.md → `exp/qdrant_step_knn/analysis/example_report.md`)
   - `scripts/verify_*.py` / `scripts/dump_step1a_failed_inits.py` CLI invocations → `python -m exp.trajectory_deviation.<name>`
42. Rewrite `docs/experiments/*.md` runbook command blocks (especially `warm_start_sweep.md`/`.en.md` lines 205–231 with literal `configs/cache_runs/warm_start_exp/<variant>/` and `data/warm_start_exp/state_full_*.json`, and `trajectory_deviation.md`/`.en.md` step commands).
43. Update `cache.yaml` if it references `data/cache_artifacts/...` (grep first; path is consumed by `serve_policy` at startup).
44. Sanity-check script bootstraps: the `Path(__file__).resolve().parent.parent` fixes in Phase 3 step 11c must match the actual new depth (`exp/trajectory_deviation/<file>.py` → repo root is `parents[2]`).
45. Replace `.gitignore` per §3.5 (drop `data/**` / `configs/**/*.{log,png,pdf}` / `exp_outputs` blocks; add `exp/**/data/**` with the init/calibration + 24-result-JSON exceptions).
46. Final audit: re-run the Phase 3 step 11h command + the step 40 command; both must return zero hits outside `logs/` plan/archive files.

### Phase 8 — Rules doc

47. Create `docs/experiments/artifact_layout.md` describing §3.1 + §3.2 as the canonical rule; register in `docs/experiments/README.md`; register in WORKING_AGREEMENT §8 subsystem table.
48. Update `logs/README.md` index entry for this log to `Implemented`.

### Phase 9 — Verify

49. `uv run pytest` — must be green.
50. Dry-run one small experiment per package and confirm outputs land under `exp/<exp>/data/` and not at the old paths.
51. `git status` — should show no stray files; `git log --stat` — confirm `git mv`s preserved history for tracked files, including the 24 result JSONs.

**Estimated magnitude**: ~474 files touched via `git mv` (includes 24 tracked result JSONs now moved with history), ~30 via plain `mv` (untracked), 1 `git rm` (`exp/cache_experiment/__init__.py` only), ~5.2 GB relocated on disk (most of it ignored). Phases 3–6 produce one commit each; Phase 7 produces one commit; Phase 8 produces one commit.

---

## 5. Open Decisions (discuss one-by-one)

### Already decided by owner (locked)

| # | Topic | Decision |
|---|-------|----------|
| L1 | Layout per experiment | `config/`, `data/` (with `data/cache_artifacts/`), `analysis/` subdirs; runner code directly at `exp/<exp>/` root; `common/` is an "experiment" with the same sub-structure |
| L2 | Repo root cleanup (`cache.yaml`, `pi0.5vla.pdf`, `convert.py`, `cmd.sh`, `simple_pytorch_train.py`, `run_step4_tests.sh`) | **Out of scope this round** — don't touch |
| L3 | Analysis tool definition | Plot scripts, analyzers, and their outputs all go to `analysis/` |
| L4 | Test scripts | Pytest-integrated → `tests/` (unchanged); non-integrated smoke/verify scripts → `exp/<exp>/` root |

### Open — resolve in order

**Q1 — Where do `calibrate_*.py` and `generate_*_yamls.py` live?** **[RESOLVED 2026-04-17]**

Owner decision: all such scripts stay at `exp/<exp>/` root (treated as experiment code, not analysis).

Affected files:
- `exp/cache_experiment/calibrate_score_sum_stats.py` → stay at root
- `exp/cache_experiment/calibrate_robot_state_tau.py` → stay at root
- `exp/cache_experiment/generate_cache_run_yamls.py` → stay at root
- `exp/temporal_prune/generate_temporal_prune_yamls.py` → stay at root

---

**Q2 — How do cross-experiment vs per-experiment data sources split?** **[RESOLVED 2026-04-17]**

| Source | Target | Rationale |
|--------|--------|-----------|
| `data/cache_artifacts/libero_spatial/` (7 pkls + calibration.json, 2.8 GB) | `exp/common/data/cache_artifacts/libero_spatial/` | Shared by phase1 / trajectory / temporal_prune / warm_start / deviate_exp via `cache.yaml` preload |
| `data/cache_artifacts/libero_spatial_warm/` (3 warm-variant pkls) | `exp/common/data/cache_artifacts/libero_spatial_warm/` | Same kind of artifact; kept in `common/` alongside the regular variant |
| `data/deviation_experiment/gt/` (1.8 GB) | `exp/trajectory_deviation/data/gt/` | Deviation-private GT |
| `data/deviation_experiment/gt_trajectories/` (16 MB) | `exp/trajectory_deviation/data/gt_trajectories/` | Deviation-private; naming duplication vs `gt/` flagged as cleanup item (see below) |
| `data/deviation_experiment/collected/libero_spatial/` (703 MB) | `exp/trajectory_deviation/data/collected/libero_spatial/` | Deviation-private collected trajectories |
| `data/deviation_experiment/inits/` (tracked `*.init` + `per_unit_filters/*.json`) | `exp/trajectory_deviation/data/inits/` | Deviation-pipeline internal (shared across step1a/1b/2/3) |
| `data/deviation_experiment/step1b_inits/` | `exp/trajectory_deviation/data/step1b_inits/` | Deviation step-1a filtered output for step-1b/2/3 |

**Deferred cleanup**: `gt/` vs `gt_trajectories/` naming overlap — both live under `exp/trajectory_deviation/data/` after migration; consolidating into one name is a separate cleanup task, not required by this plan.

---

**Q3 — Experiment boundaries** **[RESOLVED 2026-04-17]**

Owner rule: only experiments currently standing out as `exp/` packages stay distinguished; all "basic sweep" experiments (phase1, trajectory) fold into `exp/common/`.

**Final experiment roster**:

| Package | Status |
|---------|--------|
| `exp/common/` | hosts shared infrastructure + **basic sweeps (phase1, trajectory)** |
| `exp/trajectory_deviation/` | existing, distinguished |
| `exp/temporal_prune/` | existing, distinguished |
| `exp/qdrant_step_knn/` | existing, distinguished |
| `exp/warm_start/` | **new** — distinguished (has own `analyze_warm_sweep.py`) |
| `exp/cache_experiment/` | **dissolved** — contents distributed below |

**`exp/cache_experiment/` dissolution**:

| Source file | Target |
|-------------|--------|
| `run_cache_experiments.py` | `exp/common/` |
| `build_in_memory_cache_artifact.py` | `exp/common/` |
| `build_clip_cache_artifact.py` | `exp/common/` |
| `calibrate_score_sum_stats.py` | `exp/common/` |
| `calibrate_robot_state_tau.py` | `exp/common/` |
| `generate_cache_run_yamls.py` | `exp/common/` |
| `analyze_cache_results.py` | `exp/common/` |
| `analyze_warm_sweep.py` | `exp/warm_start/` |
| `__init__.py` | removed (`git rm`; owner G1 R2 2026-04-17: single 1-line docstring file, no functional impact) |

**`exp/common/` internal layout (owner-decided)**:

- **Code**: flat at `exp/common/` root (scripts live as top-level modules, same as today's `_subprocess.py` etc.).
- **Config / data / analysis**: **preserve original sub-grouping** — don't collapse phase1 and trajectory into a flat heap.

```
exp/common/
  __init__.py
  _subprocess.py, _unit_key.py, _cache_config_rpc.py, _run_state_base.py   # existing
  run_cache_experiments.py, build_*.py, calibrate_*.py, generate_cache_run_yamls.py, analyze_cache_results.py   # moved

  config/
    phase1/*.yaml                          # 40 files (was configs/cache_runs/phase1/*.yaml)
    trajectory/*.yaml                      # 60 files (was configs/cache_runs/trajectory/*.yaml)

  data/
    phase1/                                # was configs/cache_runs/phase1/*.log + experiment_state.json
    trajectory/                            # was configs/cache_runs/trajectory/*.log + experiment_state.json
    cache_artifacts/
      libero_spatial/                      # 7 pkl + calibration.json
      libero_spatial_warm/                 # 3 pkl

  analysis/
    phase1/                                # plot_results.py, phase1_analysis.md, phase1_results.{pdf,png}, phase1_analysis.pdf
    trajectory/                            # plot_results.py, trajectory_analysis.md, trajectory_results{,_facets}.{pdf,png}, trajectory_analysis.pdf
```

---

**Q4 — `scripts/verify_*.py` and `dump_step1a_failed_inits.py` relocation** **[RESOLVED 2026-04-17]**

Verification:
- All four scripts (`verify_env_save_restore.py`, `verify_restore_obs_equivalence.py`, `verify_step2.py`, `dump_step1a_failed_inits.py`) are **CLI tools** (`def main()` + `if __name__ == "__main__":`), not pytest test files.
- They are **covered by** pytest tests in `tests/scripts/`, which ARE real pytest files (`def test_*`, `import pytest`, `importlib` loading).

Owner rule: non-framework-integrated scripts go with experiment code; framework-integrated tests stay in `tests/`.

| File | Target |
|------|--------|
| `scripts/verify_env_save_restore.py` | `exp/trajectory_deviation/verify_env_save_restore.py` |
| `scripts/verify_restore_obs_equivalence.py` | `exp/trajectory_deviation/verify_restore_obs_equivalence.py` |
| `scripts/verify_step2.py` | `exp/trajectory_deviation/verify_step2.py` |
| `scripts/dump_step1a_failed_inits.py` | `exp/trajectory_deviation/dump_step1a_failed_inits.py` |
| `tests/scripts/test_verify_smoke_scripts.py` | stays in `tests/` (update `sys.path` / `importlib` to new script path) |
| `tests/scripts/test_dump_step1a_failed_inits.py` | stays in `tests/` (update `importlib.util.spec_from_file_location` path) |

Remaining `scripts/` entries (`dump_libero_init_images.py`, `compute_norm_stats.py`, `serve_policy.py`, `train.py`, `train_pytorch.py`, `train_test.py`) are general-purpose → stay at `scripts/`.

---

**Q5 — Whole-`data/` migration details** **[RESOLVED 2026-04-17]**

| Source | Target |
|--------|--------|
| `data/db/libero_cache/` (3.5 GB runtime Qdrant DB) | `exp/common/data/db/libero_cache/` |
| `data/db_init/{libero, libero_cache, libero_cache_image, libero_image}/` (369 MB) | `exp/common/data/db_init/{libero, libero_cache, libero_cache_image, libero_image}/` |
| `data/db_init/sample_cache.py` | **stay with its data** at `exp/common/data/db_init/sample_cache.py` (uses `SCRIPT_DIR` to locate sibling data dirs; zero code change) |
| `data/libero/videos/` (4 MB) | `exp/common/data/libero/videos/` |
| `data/aloha_sim/videos/` (2 MB) | `exp/common/data/aloha_sim/videos/` |
| Top-level `data/` directory | **delete after migration** — grep & update all hard-coded `data/...` paths across `src/`, `exp/`, `scripts/`, `docs/experiments/*.md` |

---

### Future / deferred

| # | Topic | Note |
|---|-------|------|
| F2 | Repo-root files (§3.3) | Separate task |
| ~~F3~~ | ~~`exp/<exp>/data/cache_artifacts/` gitignore policy (large pkls)~~ | **Resolved 2026-04-17** (G1 R1 revision): pkls stay ignored via `exp/**/data/**`; `calibration.json` explicitly whitelisted in §3.5 |

---

## 6. Risks & Rollback

- **Risk**: runbook commands in `docs/experiments/*.md` (including the warm_start lines 205–231 currently open in the IDE) embed literal paths like `data/warm_start_exp/state_full_*.json` and `configs/cache_runs/warm_start_exp/<variant>/`. Any path change without runbook update breaks reproducibility for the next run.
- **Risk**: `src/openpi/cache/` or `serve_policy.py` may hard-code `cache.yaml` path at repo root. Must grep before moving.
- **Risk**: `data/deviation_experiment/inits/*.init` are **tracked on purpose** (per `.gitignore` comment) as cross-machine source-of-truth. Migration must not break that exception.
- **Rollback**: pre-migration tag lets a full revert work with `git reset --hard pre-artifact-reorg-2026-04-17` + manual `data/` restore from backup (coordinate before starting).

---

## 7. Progress Tracking

| Phase | Status | Notes |
|-------|--------|-------|
| Understand (this doc) | Done | Inventory complete in §2 |
| Layout rules (§3.1–§3.2) | Locked 2026-04-17 | — |
| File-by-file mapping (§3.4) | **Revised 2026-04-17 (G1 R2)** | R1: split analysis/data + temporal_prune filename + T/U column; R2: TR column collapsed to T (24 result JSONs stay tracked) |
| Executable migration (§4) | **Revised 2026-04-17 (G1 R2)** | 8 phases, 51 steps; Phase 2 removed (no untracking); Phase 4 uses `git mv` for 24 result JSONs; Phase 7 broadened sweep |
| Decisions Q1–Q5 (§5) | **All resolved 2026-04-17** | See Decision Log |
| `.gitignore` (§3.5) | **Revised 2026-04-17 (G1 R2)** | 24 result JSONs added as explicit whitelist entries; policy is pure relocation (no tracking change) |
| **G1 review** | **APPROVED 2026-04-17 (Round 3)** | R1: 7 concerns → R2 revision. R2: 4 concerns → R3 revision. R3: APPROVED; one non-blocking doc fix (experiment_state.json ×4→×3 in §1) applied post-approval. |
| Code (migrations) | Ready to start | G1 approved; can proceed to Phase 0 |
| G2 review | Not started | |
| Verify (`uv run pytest` + dry run) | Not started | |

### Decision Log

| Date | Decision | Source |
|------|----------|--------|
| 2026-04-17 | Per-experiment layout: `config/` + `data/` (+`data/cache_artifacts/`) + `analysis/`; code at experiment root | owner |
| 2026-04-17 | `common/` is an "experiment" with identical sub-structure | owner |
| 2026-04-17 | Repo-root files out of scope this round | owner |
| 2026-04-17 | Analysis tools + outputs all go to `analysis/` | owner |
| 2026-04-17 | Pytest-integrated tests stay in `tests/`; non-integrated smoke scripts live with experiment code | owner |
| 2026-04-17 | **Entire `data/` tree migrates under `exp/`**; shared→`exp/common/data/`, private→`exp/<exp>/data/` | owner |
| 2026-04-17 | Q1 resolved — `calibrate_*.py`, `generate_*_yamls.py` stay at `exp/<exp>/` root | owner |
| 2026-04-17 | Q2 resolved — `libero_spatial[_warm]/` → `exp/common/data/cache_artifacts/`; all `deviation_experiment/` data → `exp/trajectory_deviation/data/`; `gt` vs `gt_trajectories` rename deferred | owner |
| 2026-04-17 | Q3 resolved — `exp/cache_experiment/` dissolved; shared scripts → `exp/common/` root (flat); phase1 + trajectory live in `exp/common/{config,data,analysis}/` with **preserved sub-grouping**; distinguished packages = `trajectory_deviation`, `temporal_prune`, `qdrant_step_knn`, `warm_start` (new) | owner |
| 2026-04-17 | Q4 resolved — 4 CLI scripts (`verify_*.py`, `dump_step1a_failed_inits.py`) → `exp/trajectory_deviation/` root; their pytest files stay in `tests/` with updated import paths | owner |
| 2026-04-17 | Q5 resolved — `db/`+`db_init/`+`libero/videos/`+`aloha_sim/videos/` → `exp/common/data/`; `sample_cache.py` stays with its data; top-level `data/` directory deleted after migration | owner |
| 2026-04-17 | G1 R1 revision — §3.4.2 T/U column + temporal_prune filename; §3.4.3 analysis/data split; §3.5 finalized; Phase 2 `git rm --cached` semantics; Phase 3 expanded import/path sweep (13a–13h); Phase 7 broadened sweep patterns; F3 resolved | assistant (executing G1 R1 feedback) |
| 2026-04-17 | **Owner G1 R2 directive**: this task is a file-location refactor only; no files may be deleted; empty directories may be removed; the 24 tracked result JSONs are moved via `git mv` and remain tracked. Exception: `exp/cache_experiment/__init__.py` (1-line docstring, no functional impact) may be deleted per subsequent owner confirmation. | owner (via G1 R2 reviewer relay + subsequent owner confirmation for `__init__.py`) |
| 2026-04-17 | G1 R2 revision — §1 "Downstream consequences" bullet about 24 tracked JSONs removed; §1 adds "Tracking-policy clarification" paragraph; §3.4.2 TR→T (all tracked, `git mv`); Phase 2 removed; Phase 4 uses `git mv` for 24 JSONs; Phase 3/4/5/6/7/8/9 renumbered (steps 5–51); §3.5 adds 24 explicit `!` whitelist entries; §3.5 "Result JSON semantics" rewritten (pure relocation) | assistant (executing G1 R2 feedback) |

---

## 8. Appendix — Raw Numbers

- Git-tracked files by extension: 643 JSON, 209 PY, 168 YAML, 95 MD, 13 LOG, 10 INIT, 7 YML, 5 TXT, 4 SH, 1 PDF.
- Tracked artifacts outside `src/tests/docs/examples/packages/`: 207 (dominated by `configs/cache_runs/**.yaml` and `data/**/*.json`).
- `data/` disk usage: 9.3 GB total (3.5 GB `db/`, 2.8 GB `cache_artifacts/`, 2.5 GB `deviation_experiment/`, 369 MB `db_init/`, rest small).
- `configs/` disk usage: 227 MB (51 MB `phase1/`, 82 MB `temporal_prune/`, 92 MB `trajectory/`, 2 MB `warm_start_exp/`, 432 KB `deviate_exp/`).

---

## Review Log

### G1 Round 1 — Reviewer

- [Concern] Finalize the `.gitignore` policy before G1 approval instead of leaving it tentative/open — reasoning: the plan marks mapping and executable migration as locked, but §3.5 still says "Tentative" and leaves `exp/<exp>/data/cache_artifacts/` unresolved. The proposed `exp/**/data/**` ignore also lacks explicit exceptions for intentionally tracked source-of-truth artifacts such as moved `*.init` files and `calibration.json`, so the migration cannot be enforced safely.
- [Concern] Split analysis outputs out of the `data/deviation_experiment/` directory moves — reasoning: §3.2 says plot scripts, analysis tools, generated `*.png`, `*.pdf`, and analysis `*.md` belong under `analysis/`, but §3.4.3/Phase 5 move whole `deviate_scores/` and `step3/` subtrees into `exp/trajectory_deviation/data/`, including `deviate_scores/plots/*.png`, `deviate_scores/plots/*.md`, `step2_deviate_score_analysis*.md`, and `step3/plots/*.png`.
- [Concern] Correct tracked/untracked and filename mismatches in the executable move steps — reasoning: several files described as plain/untracked moves are tracked (`configs/cache_runs/deviate_exp/step1a_analysis.md`, `phase1_analysis.md`, `trajectory_analysis.md`, `temporal_prune_analysis.md`), so the executable plan should use `git mv` where history preservation matters. The temporal-prune mapping names `tp_analysis.md`, but the actual file is `temporal_prune_analysis.md`.
- [Concern] Expand Phase 3 path/import updates before the Phase 3 pytest gate — reasoning: Phase 3 moves `analyze_deviation_results.py` and other analysis modules to `exp/trajectory_deviation/analysis/`, but only specifies `exp.cache_experiment.*` and `tests/scripts` import updates before running `uv run pytest`. Existing tests and docs import or invoke `exp.trajectory_deviation.analyze_deviation_results`, `exp.trajectory_deviation.analyze_step2_deviate_scores`, and plot modules, so the intermediate test gate will fail unless those imports/module paths are explicitly migrated or the move is delayed.
- [Concern] Resolve the Phase 2 `git rm` versus "restored under exp/" semantics — reasoning: the plan removes tracked result JSONs first, then says they will be restored under `exp/`, while §3.5 proposes ignoring all `exp/**/data/**`. The plan needs to state whether these JSONs become untracked runtime artifacts or remain tracked history-preserved files, and the commands must match that choice.
- [Concern] Broaden the path reference sweep beyond `grep -rn "data/"` — reasoning: the migration also changes `configs/cache_runs`, `exp_outputs`, dotted module paths, and script path bootstraps such as `Path(__file__).resolve().parent.parent` in scripts that move from `scripts/` to `exp/trajectory_deviation/`. These will not be found by the current Phase 7 command, yet they can break CLI execution and runbooks.
- [Concern] Bring `logs/README.md` into sync with the current plan state before approval — reasoning: the index entry still describes unresolved owner decisions and "待 owner 选方向后进入 G1", while the plan itself says Q1-Q5 are resolved and G1 is ready. Working Agreement §4 requires the logs index to stay synchronized with log changes.

### G1 Round 1 — Author response (2026-04-17)

Each concern addressed in-place; summary of where:

| # | Reviewer concern | Resolution location |
|---|------------------|---------------------|
| 1 | `.gitignore` tentative, missing `*.init` / `calibration.json` exceptions, cache_artifacts policy unresolved | §3.5 rewritten from "Tentative" → "FINAL"; diff-style replacement with explicit `!exp/trajectory_deviation/data/inits/*.{init,pruned_init}`, `!exp/common/data/cache_artifacts/**/calibration.json`; F3 marked resolved in §5 deferred table |
| 2 | `data/deviation_experiment/deviate_scores/` and `step3/` contain analysis outputs (PNG/MD/plots) that violate §3.2 | §3.4.3 table split into per-subdir rows (deviate_scores JSON→data, MD→analysis, plots→analysis; step3 data→data, plots→analysis); Phase 5 step 28b/28c encode the split as concrete commands |
| 3 | Tracked/untracked status wrong in Phase 4; `tp_analysis.md` vs actual `temporal_prune_analysis.md` | §3.4.2 gains **T/U/TR** column; filename corrected; Phase 4 steps 18a/18b, 22a/22b, 23c/23d, 24c/24d, 25a–25d split `git mv` (tracked) vs plain `mv` (untracked) |
| 4 | Phase 3 import updates too narrow (misses `exp.trajectory_deviation.analyze_deviation_results` etc. and script bootstrap `Path(__file__).resolve().parent.parent`) | Phase 3 expanded to 13a–13h: 13b explicitly covers `tests/exp/test_analyze_deviation_results.py` + docstring `-m exp.trajectory_deviation.<name>` self-references + doc runbook paths; 13c fixes the `parents[2]` bootstrap in `scripts/verify_env_save_restore.py:69` and `scripts/verify_restore_obs_equivalence.py:77`; 13h is a zero-hit audit command run before the pytest gate |
| 5 | Phase 2 `git rm` + "restored under exp/" semantics ambiguous vs §3.5 ignoring `exp/**/data/**` | Phase 2 header renamed to "Untrack result JSONs"; uses `git rm --cached` (preserve working-tree file); §3.5 adds explicit "Result JSON semantics" paragraph stating they become untracked runtime artifacts, not restored tracked files |
| 6 | Phase 7 sweep only looks for `data/` — misses `configs/cache_runs`, `exp_outputs`, module paths, script bootstraps | Phase 7 step 42 rewritten with ripgrep `-e` alternation covering `data/<subpath>`, `configs/cache_runs/`, `exp_outputs/`, `scripts/verify_*` / `dump_step1a_failed_inits`; step 46 adds bootstrap sanity check; step 48 final zero-hit audit across both Phase-3 and Phase-7 patterns |
| 7 | `logs/README.md` index out of sync | Being updated in the same commit as this revision (description rewritten to reflect G1 R1 state) |

### G1 Round 2 — Reviewer

Owner directive supplied before this round (2026-04-17): this task is a file-location refactor only; do not delete any file; empty directories may be removed; the 24 tracked result JSONs must be moved to their destination directories and must not be removed from git tracking. These directives are owner-specified and override the R1 revision's "untrack runtime artifacts" approach.

- [Concern] Replace every `git rm --cached` / untrack step for the 24 tracked result JSONs with `git mv` to the final data directories — reasoning: §3.4.2 still marks those files as `TR`; §3.5 still says they become untracked runtime artifacts; Phase 2 still says `git rm --cached`; Phase 4 then uses plain `mv`. This contradicts the owner directive that no files are deleted/untracked in this location-only refactor. The plan must keep these files tracked and move them directly to `exp/common/data/{phase1,trajectory}/`, `exp/temporal_prune/data/`, and `exp/warm_start/data/{clip,max_pool,spatial16}/`.
- [Concern] Remove the planned deletion of `exp/cache_experiment/__init__.py` and convert it to a move/reuse/compatibility-shim step — reasoning: §3.4.1 and §5 Q3 still say `__init__.py` is `removed`, and Phase 3 step 9 still says `git rm exp/cache_experiment/__init__.py`. The file currently contains a package docstring, so deleting it violates the owner directive. The plan should either `git mv` it to a new package initializer, merge its content into the appropriate existing initializer, or keep `exp/cache_experiment/__init__.py` as a compatibility shim.
- [Concern] Update `.gitignore` strategy to match tracked JSON preservation — reasoning: §3.5 currently ignores `exp/**/data/**` and only whitelists init/calibration files. If the 24 JSONs remain tracked via `git mv`, existing tracking survives, but future edits remain tracked only because they are already in the index. The plan should state this explicitly and avoid describing those JSONs as ignored/regenerable untracked artifacts; optionally add explicit `!` exceptions if future add/re-add should be supported.
- [Concern] Update progress/decision logs to record the new owner directive — reasoning: the Decision Log currently records the R1 revision as using `git rm --cached` semantics, but Round 2 owner input reverses that choice. The plan needs a new owner decision row documenting "file-location refactor only; no file deletion; empty directories may be removed; 24 tracked result JSONs use `git mv` and remain tracked."

### G1 Round 2 — Author response (2026-04-17)

Before executing, the author performed an explicit legitimacy check (per process gap surfaced in the previous turn):
- **Concerns 1, 3, 4** are clear, consistent with the owner directive, and internally non-contradictory — accept.
- **Concern 2 (`__init__.py`)** required owner clarification — the file is a 1-line docstring with no functional content; owner confirmed deletion is acceptable (see Decision Log row 2026-04-17 G1 R2 directive). Remaining three concerns then applied without conflict.
- No concern was overridden; all four executed.

| # | Reviewer concern | Resolution location |
|---|------------------|---------------------|
| 1 | Replace `git rm --cached` + plain `mv` with `git mv` for 24 tracked JSONs | §3.4.2 `TR` column removed; legend now only T/U; §1 adds Tracking-policy clarification paragraph; Phase 2 removed entirely; Phase 4 steps 18b, 21c, 22c, 23c/23d use `git mv` for the 24 JSONs |
| 2 | Don't delete `exp/cache_experiment/__init__.py` (owner file-deletion prohibition) | Raised with owner; owner confirmed deletion allowed for this specific 1-line docstring file. Decision Log row records both the general "no file deletion" rule and the exception. Phase 3 step 7 keeps `git rm` with an explicit in-line citation of the owner exception |
| 3 | `.gitignore` semantics must match continued-tracked status | §3.5 title updated to "FINAL — G1 R2"; "Result JSON semantics" rewritten to explain pure relocation; added 24 explicit `!` whitelist entries (per-path for `experiment_state.json` / `cache_eval_results.json` and globs for `*.episode_results.json`). Rationale (a)/(b) included so future maintainers know the whitelist is intentional, not defensive |
| 4 | Record new owner directive in Decision Log | New Decision Log row added 2026-04-17 for the owner G1 R2 directive (no deletion / empty-dir exception / `__init__.py` exception / 24 JSONs tracked via `git mv`); separate assistant-authored row records the mechanical R2 revisions. Progress Tracking updated to reflect R2 status on affected sections |

### G1 Round 3 — Reviewer

Owner directive carried into this round (2026-04-17): this is a file-location refactor only; no files may be deleted; empty directories may be removed; the 24 tracked result JSONs must be moved with `git mv` and remain tracked. The only file-deletion exception is `exp/cache_experiment/__init__.py`, which owner confirmed may be removed because it is a single docstring-only package marker with no functional impact.

- No blocking concerns found. R2 concerns 1/3/4 are resolved in the executable plan, `.gitignore` policy, and Decision Log. R2 concern 2 is resolved by the explicit owner exception for `exp/cache_experiment/__init__.py`.
- [Suggestion] Correct the §1 tracking-policy parenthetical from `experiment_state.json` × 4 to `experiment_state.json` × 3 — reasoning: an independent `git ls-files` check shows the 24 tracked result JSONs are 3 `experiment_state.json`, 3 `cache_eval_results.json`, and 18 `*.episode_results.json`. The executable mapping and Phase 4 `git mv` commands already cover the correct 24 files, so this is a non-blocking documentation count fix.
