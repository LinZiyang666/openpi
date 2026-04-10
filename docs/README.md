# docs/

Project architecture and usage guides. For implementation logs, see [`logs/README.md`](../logs/README.md).

> **AGENT: READ FIRST** — Project rules are in [`constitution.md`](../constitution.md) (workflow §2, documentation §4, subsystem rules §8). This file is a navigation index only — it does not define rules.

## Reading Paths

### First time in this fork

1. Read [`../constitution.md`](../constitution.md) for project rules and development workflow.
2. Read [openpi_reference.md](openpi_reference.md) for project structure, model architecture, code paths, and deployment options.

### Working on the inference pipeline or cache system

1. Read [openpi_reference.md](openpi_reference.md) to locate the PyTorch model and policy code paths.
2. Read [cache_system_architecture.md](cache_system_architecture.md) for the CP1/CP2/CP3 design and integration boundary.
3. Read [cache_system_tutorial.md](cache_system_tutorial.md) for hands-on component guide, YAML config, and testing patterns.
4. Then read [`../logs/archive/step1.log`](../logs/archive/step1.log) and [`../logs/archive/step2.log`](../logs/archive/step2.log) for the actual staged API and timing implementation history.

### Working on data collection

1. Read [data_collection_guide.md](data_collection_guide.md) for the user-facing workflow and HDF5 layout.
2. Then read [`../logs/archive/step3_data_collection.log`](../logs/archive/step3_data_collection.log) for hook points, wrapper ordering, and compatibility decisions.

### Working on cache experiments

1. Read [run_cp1_cache_experiment.md](run_cp1_cache_experiment.md) for the full experiment pipeline (artifact building, YAML generation, experiment execution, result analysis).
2. Read [`../logs/cache_experiment_plan.log.md`](../logs/cache_experiment_plan.log.md) for the experiment design rationale.

### Working on training, deployment, or environment setup

1. Read [openpi_reference.md](openpi_reference.md) for configs, transforms, and deployment modes.
2. Then read the relevant guide:
   - [remote_inference.md](remote_inference.md)
   - [aloha_sim_remote.md](aloha_sim_remote.md)
   - [docker.md](docker.md)
   - [norm_stats.md](norm_stats.md)

## Architecture (Fork)

| File | Description |
|------|-------------|
| [cache_system_architecture.md](cache_system_architecture.md) | Cache system spec: 3-stage pipeline, CP1/CP2/CP3 checkpoints, interceptor pattern, component design with design-vs-implementation annotations |
| [cache_system_architecture_chinese.md](cache_system_architecture_chinese.md) | Chinese version — **frozen at 2026-04-03**, not synced with English version |
| [cache_system_tutorial.md](cache_system_tutorial.md) | Complete tutorial: glossary, all components (KeyBuilder/Gate/Judge/SearchStrategy/Backend), YAML config, registration, testing |
| [cache_system_workflow.md](cache_system_workflow.md) | End-to-end workflow diagrams: startup, single inference with CP1/CP3, episode lifecycle, storage layer, YAML mapping, design principles |
| [openpi_reference.md](openpi_reference.md) | Project structure, model architecture, Pi0 vs Pi0.5 differences, code paths, training configs, deployment, hardware |

## Usage Guides (Fork)

| File | Description |
|------|-------------|
| [cache_migration_guide.md](cache_migration_guide.md) | Cache framework migration guide (Chinese): how to adapt the cache system for non-Pi0.5 models |
| [cache_migration_guide.en.md](cache_migration_guide.en.md) | Cache framework migration guide (English) |
| [data_collection_guide.md](data_collection_guide.md) | HDF5 data collection via `--collect` flag, schema and directory layout |
| [run_cp1_cache_experiment.md](run_cp1_cache_experiment.md) | CP1 Cache experiment guide: artifact building, calibration, YAML generation, 3-phase experiment execution, result analysis |
| [aloha_sim_remote.md](aloha_sim_remote.md) | ALOHA Sim remote inference (WSL2 client + remote GPU) |

## Upstream Guides (from original openpi)

| File | Description |
|------|-------------|
| [remote_inference.md](remote_inference.md) | General WebSocket remote inference setup |
| [docker.md](docker.md) | Docker installation and container usage |
| [norm_stats.md](norm_stats.md) | Normalization statistics: reuse, recompute, asset_id mapping |
