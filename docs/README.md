# docs/

Project architecture and usage guides. For implementation logs, see [`claude_log/README.md`](../claude_log/README.md).

## Reading Paths

### First time in this fork

1. Read [`../README.md`](../README.md) for fork scope and entry points.
2. Read [`../CLAUDE.md`](../CLAUDE.md) for current development workflow, constraints, and caveats.
3. Read [openpi_reference.md](openpi_reference.md) for project structure, model architecture, code paths, and deployment options.

### Working on the inference pipeline or cache system

1. Read [openpi_reference.md](openpi_reference.md) to locate the PyTorch model and policy code paths.
2. Read [cache_system_architecture.md](cache_system_architecture.md) for the CP1/CP2/CP3 design and integration boundary.
3. Then read [`../claude_log/step1.log`](../claude_log/step1.log) and [`../claude_log/step2.log`](../claude_log/step2.log) for the actual staged API and timing implementation history.

### Working on data collection

1. Read [data_collection_guide.md](data_collection_guide.md) for the user-facing workflow and HDF5 layout.
2. Then read [`../claude_log/step3_data_collection.log`](../claude_log/step3_data_collection.log) for hook points, wrapper ordering, and compatibility decisions.

### Working on training, deployment, or environment setup

1. Read [openpi_reference.md](openpi_reference.md) for configs, transforms, and deployment modes.
2. Then read the relevant guide:
   - [remote_inference.md](remote_inference.md)
   - [aloha_sim_remote.md](aloha_sim_remote.md)
   - [libero_remote.md](libero_remote.md)
   - [docker.md](docker.md)
   - [norm_stats.md](norm_stats.md)

## Architecture

| File | Description |
|------|-------------|
| [cache_system_architecture.md](cache_system_architecture.md) | Cache system spec: 3-stage pipeline, CP1/CP2/CP3 checkpoints, interceptor pattern (English) |
| [cache_system_architecture_chinese.md](cache_system_architecture_chinese.md) | Same as above, Chinese version |
| [cache_system_tutorial.md](cache_system_tutorial.md) | Complete tutorial: glossary, all components (KeyBuilder/Gate/Judge/SearchStrategy/Backend), YAML config, registration, testing |
| [openpi_reference.md](openpi_reference.md) | Project structure, model architecture, Pi0 vs Pi0.5 differences, code paths, training configs, deployment, hardware |

## Usage Guides

| File | Description |
|------|-------------|
| [data_collection_guide.md](data_collection_guide.md) | HDF5 data collection via `--collect` flag, schema and directory layout |
| [remote_inference.md](remote_inference.md) | General WebSocket remote inference setup (upstream) |
| [aloha_sim_remote.md](aloha_sim_remote.md) | ALOHA Sim remote inference (WSL2 client + remote GPU) |
| [run_cp1_cache_experiment.md](run_cp1_cache_experiment.md) | CP1 Cache 实验完整运行指南：artifact 构建、校准、YAML 生成、3-phase 实验执行、结果分析 |
| [libero_remote.md](libero_remote.md) | LIBERO remote inference with video recording |
| [docker.md](docker.md) | Docker installation and container usage (upstream) |
| [norm_stats.md](norm_stats.md) | Normalization statistics: reuse, recompute, asset_id mapping (upstream) |
