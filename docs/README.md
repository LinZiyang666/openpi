# docs/

Project architecture and usage guides. For implementation logs, see [`logs/README.md`](../logs/README.md).

> **AGENT: READ FIRST** — Project rules are in [`WORKING_AGREEMENT.md`](../WORKING_AGREEMENT.md) (workflow §2, documentation §4, subsystem rules §8). This file is a navigation index only — it does not define rules.

Bilingual companions (`*.en.md`, `*_chinese.md`) are folded under the primary entry as `[EN]` / `[ZH]` links and do not occupy their own row.

## Reading Paths

### First time in this fork

1. Read [`../WORKING_AGREEMENT.md`](../WORKING_AGREEMENT.md) for project rules and development workflow.
2. Read [openpi_reference.md](openpi_reference.md) for project structure, model architecture, code paths, and deployment options.

### Working on the inference pipeline or cache system

1. [openpi_reference.md](openpi_reference.md) — locate the PyTorch model and policy code paths.
2. [cache_system_architecture.md](cache_system_architecture.md) — CP1/CP2/CP3 design and integration boundary.
3. [cache_system_tutorial.md](cache_system_tutorial.md) — hands-on component guide, YAML config, testing patterns.
4. [`../logs/archive/step1.log`](../logs/archive/step1.log) and [`../logs/archive/step2.log`](../logs/archive/step2.log) — staged API and timing implementation history.

### Working on data collection

1. [data_collection_guide.md](data_collection_guide.md) — user-facing workflow and HDF5 layout.
2. [`../logs/archive/step3_data_collection.log`](../logs/archive/step3_data_collection.log) — hook points, wrapper ordering, compatibility decisions.

### Working on cache experiments

1. [run_cp1_cache_experiment.md](run_cp1_cache_experiment.md) — full experiment pipeline (artifact building, YAML generation, experiment execution, result analysis).
2. [`../logs/archive/cache_experiment_plan.log.md`](../logs/archive/cache_experiment_plan.log.md) — experiment design rationale.

### Working on training, deployment, or environment setup

1. [openpi_reference.md](openpi_reference.md) — configs, transforms, and deployment modes.
2. Then the relevant deployment guide in the table below.

---

## Architecture & Reference (Fork)

| File | Description |
|------|-------------|
| [openpi_reference.md](openpi_reference.md) | Project structure, model architecture, Pi0 vs Pi0.5 differences, code paths, training configs, deployment, hardware |
| [cache_system_architecture.md](cache_system_architecture.md) \[[ZH](cache_system_architecture_chinese.md)\] | Cache system spec: 3-stage pipeline, CP1/CP2/CP3 checkpoints, interceptor pattern, component design. Chinese companion frozen at 2026-04-03 |
| [cache_system_workflow.md](cache_system_workflow.md) | End-to-end workflow diagrams: startup, single inference with CP1/CP3, episode lifecycle, storage layer, YAML mapping, design principles |

## Cache System Guides (Fork)

| File | Description |
|------|-------------|
| [cache_system_tutorial.md](cache_system_tutorial.md) | Complete tutorial: glossary, all components (KeyBuilder/Gate/Judge/SearchStrategy/Backend), YAML config, registration, testing |
| [cache_migration_guide.md](cache_migration_guide.md) \[[EN](cache_migration_guide.en.md)\] | Cache framework migration guide: how to adapt the cache system for non-Pi0.5 models |

## Experiment Guides (Fork)

| File | Description |
|------|-------------|
| [run_cp1_cache_experiment.md](run_cp1_cache_experiment.md) | CP1 Cache experiment guide: artifact building, calibration, YAML generation, 3-phase experiment execution, result analysis |
| [run_temporal_prune_experiment.md](run_temporal_prune_experiment.md) | Temporal Prune experiment pipeline |
| [temporal_prune_guide.md](temporal_prune_guide.md) | Temporal Prune KeyBuilder 使用指南：两步架构、参数配置、Reducer 选择、离线 Artifact 构建、生命周期 |

## Data Collection (Fork)

| File | Description |
|------|-------------|
| [data_collection_guide.md](data_collection_guide.md) | HDF5 data collection via `--collect` flag, schema and directory layout |

## Deployment & Simulator (Fork)

| File | Description |
|------|-------------|
| [aloha_sim_remote.md](aloha_sim_remote.md) | ALOHA Sim remote inference (WSL2 client + remote GPU) |
| [libero_remote.md](libero_remote.md) | LIBERO remote inference and simulator environment setup (WSL2 client + remote GPU) |

## Upstream Guides (from original openpi)

| File | Description |
|------|-------------|
| [remote_inference.md](remote_inference.md) | General WebSocket remote inference setup |
| [docker.md](docker.md) | Docker installation and container usage |
| [norm_stats.md](norm_stats.md) | Normalization statistics: reuse, recompute, asset_id mapping |
