# openpi Fork Overview

This repository is a fork of Physical Intelligence's `openpi` with additional work on staged PyTorch inference, caching, data collection, and retrieval experiments.

## Start Here

> **AGENT: READ FIRST** — Load [`CLAUDE.md`](CLAUDE.md) for Agent Directive, then [`constitution.md`](constitution.md) for full project rules before any work.

- [`constitution.md`](constitution.md): **Project constitution** — the single authoritative source of all rules
- [`CLAUDE.md`](CLAUDE.md): AI agent entry point + Agent Directive
- [`docs/README.md`](docs/README.md): Architecture docs and usage guide index
- [`logs/README.md`](logs/README.md): Implementation logs and design records index
- [`UPSTREAM_README.md`](UPSTREAM_README.md): Preserved upstream README (background reference, may be outdated)

## Current Fork Scope

- Active development targets **PyTorch + Pi0.5**
- The fork adds:
  - staged inference APIs in `src/openpi/models_pytorch/`
  - cache interception and timing in `src/openpi/cache/`
  - HDF5 data collection in `src/openpi/collect/`
  - retrieval experiments in `exp/`
- JAX inference is disabled in the current fork path

## Quick Start

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv run scripts/serve_policy.py --env <ENV_NAME>
uv run pytest
```

For task-specific setup and architecture details, follow the indexed docs above.
