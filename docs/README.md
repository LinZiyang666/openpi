# docs/

Project architecture and usage guides. For implementation logs, see [`claude_log/README.md`](../claude_log/README.md).

## Architecture

| File | Description |
|------|-------------|
| [cache_system_architecture.md](cache_system_architecture.md) | Cache system spec: 3-stage pipeline, CP1/CP2/CP3 checkpoints, interceptor pattern (English) |
| [cache_system_architecture_chinese.md](cache_system_architecture_chinese.md) | Same as above, Chinese version |
| [openpi_reference.md](openpi_reference.md) | Project structure, model architecture, Pi0 vs Pi0.5 differences, code paths, training configs, deployment, hardware |

## Usage Guides

| File | Description |
|------|-------------|
| [data_collection_guide.md](data_collection_guide.md) | HDF5 data collection via `--collect` flag, schema and directory layout |
| [remote_inference.md](remote_inference.md) | General WebSocket remote inference setup (upstream) |
| [aloha_sim_remote.md](aloha_sim_remote.md) | ALOHA Sim remote inference (WSL2 client + remote GPU) |
| [libero_remote.md](libero_remote.md) | LIBERO remote inference with video recording |
| [docker.md](docker.md) | Docker installation and container usage (upstream) |
| [norm_stats.md](norm_stats.md) | Normalization statistics: reuse, recompute, asset_id mapping (upstream) |
