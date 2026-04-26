# docs/

Project architecture and usage guides. For implementation logs, see [`logs/README.md`](../logs/README.md).

> **AGENT: READ FIRST** — Project rules are in [`WORKING_AGREEMENT.md`](../WORKING_AGREEMENT.md) (workflow §2, documentation §4, subsystem rules §8). This file is a navigation index only — it does not define rules.

Bilingual companions use `.en.md` (English) or `.zh.md` (Chinese); they are folded under the primary entry in each subdirectory's index and do not occupy their own row here.

---

## Directory Layout

```
docs/
├── reference/        # Project reference (structure, architecture, deployment)
├── architecture/     # Cache system specs and workflow diagrams
├── cache/            # Cache system user guides (tutorial, migration, components)
├── experiments/      # Experiment run-books (CP1, temporal prune, trajectory deviation)
├── data_collection/  # Data collection (HDF5 schema, --collect flag)
├── deployment/       # Deployment / simulator setup (ALOHA, LIBERO)
├── papers/           # Related-work bibliographies (inference cache literature, etc.)
└── upstream/         # Original upstream openpi docs (remote inference, docker, norm stats)
```

Each subdirectory has its own `README.md` index listing the docs inside.

## Reading Paths

### First time in this fork

1. Read [`../WORKING_AGREEMENT.md`](../WORKING_AGREEMENT.md) for project rules and development workflow.
2. Read [reference/openpi.md](reference/openpi.md) for project structure, model architecture, code paths, and deployment options.

### Working on the inference pipeline or cache system

1. [reference/openpi.md](reference/openpi.md) — locate the PyTorch model and policy code paths.
2. [architecture/cache_system.md](architecture/cache_system.md) — CP1/CP2/CP3 design and integration boundary.
3. [cache/tutorial.md](cache/tutorial.md) — hands-on component guide, YAML config, testing patterns.
4. [`../logs/archive/step1.log`](../logs/archive/step1.log) and [`../logs/archive/step2.log`](../logs/archive/step2.log) — staged API and timing implementation history.

### Working on data collection

1. [data_collection/guide.md](data_collection/guide.md) — user-facing workflow and HDF5 layout.
2. [`../logs/archive/step3_data_collection.log`](../logs/archive/step3_data_collection.log) — hook points, wrapper ordering, compatibility decisions.

### Working on cache experiments

1. [experiments/cp1_cache.md](experiments/cp1_cache.md) — full experiment pipeline (artifact building, YAML generation, experiment execution, result analysis).
2. [`../logs/archive/cache_experiment_plan.log.md`](../logs/archive/cache_experiment_plan.log.md) — experiment design rationale.

### Working on training, deployment, or environment setup

1. [reference/openpi.md](reference/openpi.md) — configs, transforms, and deployment modes.
2. Then the relevant deployment guide under [deployment/](deployment/) or [upstream/](upstream/).

---

## Section Indexes

### [reference/](reference/)

| File | Description |
|------|-------------|
| [reference/openpi.md](reference/openpi.md) | Project structure, model architecture, Pi0 vs Pi0.5 differences, code paths, training configs, deployment, hardware |

### [architecture/](architecture/)

| File | Description |
|------|-------------|
| [architecture/cache_system.md](architecture/cache_system.md) \[[ZH](architecture/cache_system.zh.md)\] | Cache system spec: 3-stage pipeline, CP1/CP2/CP3 checkpoints, interceptor pattern, component design; §5.10 Search Session — Cross-Step Score Memo (opt-in per-episode score memoization, mutation contract, lock-free derivation). Chinese companion frozen at 2026-04-03 |
| [architecture/cache_workflow.md](architecture/cache_workflow.md) | End-to-end workflow diagrams: startup, single inference with CP1/CP3, episode lifecycle, storage layer, YAML mapping, design principles |

### [cache/](cache/)

| File | Description |
|------|-------------|
| [cache/tutorial.md](cache/tutorial.md) | Complete tutorial: glossary, all components (KeyBuilder/Gate/Judge/SearchStrategy/Backend), YAML config, registration, testing; Search Session score-memo usage (lifecycle through interceptor → orchestrator, mutation contract, manual usage example, `force_legacy_path()` parity escape hatch) |
| [cache/migration.md](cache/migration.md) \[[EN](cache/migration.en.md)\] | Cache framework migration guide: how to adapt the cache system for non-Pi0.5 models |
| [cache/temporal_prune.md](cache/temporal_prune.md) | Temporal Prune KeyBuilder 使用指南：两步架构、参数配置、Reducer 选择、离线 Artifact 构建、生命周期 |
| [cache/llm_layer_extract.md](cache/llm_layer_extract.md) | CP1 LLM Layer Extract KeyBuilder 使用指南：两步架构（LayerExtractor + PrefixReducer）、attach_model 注入、离线 Stage 1 重建契约（重 tokenize + tokenizer self-check）、在线/离线 parity test |

### [experiments/](experiments/)

| File | Description |
|------|-------------|
| [experiments/artifact_layout.md](experiments/artifact_layout.md) | Canonical `exp/<experiment>/{config,data,analysis}/` layout rules — where new files go, tracking policy, `.gitignore` exceptions |
| [experiments/cp1_cache.md](experiments/cp1_cache.md) | CP1 Cache experiment guide: artifact building, calibration, YAML generation, 3-phase experiment execution, result analysis |
| [experiments/temporal_prune.md](experiments/temporal_prune.md) | Temporal Prune experiment pipeline |
| [experiments/llm_layer_extract.md](experiments/llm_layer_extract.md) | CP1 LLM Layer Extract 端到端 runbook：数据采集 → Step 2 build pkl（带 tokenizer self-check）→ YAML 模板（A/B 两种 reducer）→ run_cache_experiments → 结果分析 → manual parity verify |
| [experiments/trajectory_deviation.md](experiments/trajectory_deviation.md) \[[EN](experiments/trajectory_deviation.en.md)\] | Trajectory Deviation experiment runbook: Step 1a→1b→2→3→4 pipeline, parallelism rules, tunables |
| [experiments/warm_start_sweep.md](experiments/warm_start_sweep.md) \[[EN](experiments/warm_start_sweep.en.md)\] | Warm Start sweep runbook: 3 keybuilder × 3 start_t under always-hit + always_warm_start, artifact rebuild, 3-server parallel run, recovery/loss analysis |

### [data_collection/](data_collection/)

| File | Description |
|------|-------------|
| [data_collection/guide.md](data_collection/guide.md) | HDF5 data collection via `--collect` flag, schema and directory layout |

### [deployment/](deployment/)

| File | Description |
|------|-------------|
| [deployment/aloha_sim.md](deployment/aloha_sim.md) | ALOHA Sim remote inference (WSL2 client + remote GPU) |
| [deployment/libero.md](deployment/libero.md) | LIBERO remote inference and simulator environment setup (WSL2 client + remote GPU) |

### [papers/](papers/)

| File | Description |
|------|-------------|
| [papers/inference_cache_related_work.md](papers/inference_cache_related_work.md) | Related-work bibliography for inference caching / retrieval-augmented control in robotics, organized by proximity to our cache system (RT-Cache, VINN, VLA-Cache, BAC, RTC, Behavior Retrieval, etc.) |
| [papers/cloud_edge_deployment.md](papers/cloud_edge_deployment.md) | Cloud/edge deployment, brain-cerebellum split, fleet serving, compute/energy efficiency — deployment-context motivation for inference cache |
| [papers/paper_workbench.md](papers/paper_workbench.md) | Paper workbench: idea → method → story → experiments, living document |

### [upstream/](upstream/) — Original upstream openpi docs

| File | Description |
|------|-------------|
| [upstream/remote_inference.md](upstream/remote_inference.md) | General WebSocket remote inference setup |
| [upstream/docker.md](upstream/docker.md) | Docker installation and container usage |
| [upstream/norm_stats.md](upstream/norm_stats.md) | Normalization statistics: reuse, recompute, asset_id mapping |
