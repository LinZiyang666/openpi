# openpi Fork Overview

This repository is a fork of Physical Intelligence's `openpi` with additional work on staged PyTorch inference, caching, data collection, and retrieval experiments.

The most important point: the original upstream documentation does **not** fully describe the current state of this fork. Treat [`CLAUDE.md`](CLAUDE.md) as the authoritative entry point for this repository.

## Start Here

- [`CLAUDE.md`](CLAUDE.md): fork scope, development workflow, quick reference, and current caveats
- [`docs/README.md`](docs/README.md): architecture specs and usage guides
- [`claude_log/README.md`](claude_log/README.md): implementation logs and design records
- [`UPSTREAM_README.md`](UPSTREAM_README.md): preserved upstream-style README inherited from the original project; useful for broad background, but some details may be outdated for this fork

## Current Fork Scope

- Active development targets **PyTorch + Pi0.5**
- The fork adds:
  - staged inference APIs in `src/openpi/models_pytorch/`
  - cache interception and timing in `src/openpi/cache/`
  - HDF5 data collection in `src/openpi/collect/`
  - retrieval experiments in `exp/`
- JAX inference is disabled in the current fork path; see [`CLAUDE.md`](CLAUDE.md) and [`src/openpi/policies/policy_config.py`](src/openpi/policies/policy_config.py)

## Quick Start

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv run scripts/serve_policy.py --env <ENV_NAME>
uv run pytest
```

For task-specific setup and architecture details, follow the indexed docs above rather than reading the preserved upstream README first.
