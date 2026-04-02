# OpenPI - Pi0 / Pi0.5 VLA Model Codebase

> Forked from Physical Intelligence's [openpi](https://github.com/Physical-Intelligence/openpi) (base commit: `54cbaee6`).
> Paper: *pi0.5: a Vision-Language-Action Model with Open-World Generalization* (arXiv:2504.16054)
> For detailed model theory, training configs, and deployment options, see [`docs/openpi_reference.md`](docs/openpi_reference.md).
>
> **Scope of this fork:** All new development (cache system, data collection, retrieval) targets **PyTorch + Pi0.5 only**. The JAX inference path has been disabled and will error. Upstream Pi0 / Pi0-FAST code is preserved but not actively maintained.

---

## Development Workflow

### 1. Understand before coding

Before any work, **read the indexes first**:
- [`docs/README.md`](docs/README.md) — architecture specs and usage guides index
- [`claude_log/README.md`](claude_log/README.md) — implementation logs and design records index

Then read the specific docs relevant to the task. Never start coding without understanding the existing architecture and prior decisions.

### 2. Plan before implementation

1. **Requirement plan**: Review architecture docs (especially `docs/cache_system_architecture.md`) and discuss requirements with the user.
2. **Code plan**: Discuss the specific code changes — which files, what interfaces, how it integrates with existing modules.
3. **Alignment**: Get user confirmation on the plan before writing code.

### 3. Code with decoupling in mind

- New features should be **decoupled from the inference pipeline** — use interceptor/wrapper/hook patterns (see `src/openpi/cache/`, `src/openpi/collect/`).
- Code must be **backward-compatible** with the upstream openpi codebase and **forward-compatible** for future extensions.
- Prefer composable wrappers over modifying existing inference internals.

### 4. Verify

Run `uv run pytest` after changes. For inference-path changes, verify with the existing staged API tests.

---

## Documentation Guidelines

### Where to write

| Location | Purpose | Audience |
|----------|---------|----------|
| `docs/` | Project architecture and usage guides | Anyone learning or using the project |
| `claude_log/` | Implementation plans, design decisions, code change records | Developers and agents working on the code |
| `exp/` | Experiment scripts and configs only — no design docs | — |

### When to create docs

- **New feature or system**: Write an architecture doc in `docs/` if it introduces a new subsystem.
- **Implementation decisions**: Write a log in `claude_log/` when making non-obvious design choices.
- **Don't create docs** for trivial changes or things derivable from reading the code.

### After creating/modifying docs

Always update the relevant index:
- `docs/README.md` for docs
- `claude_log/README.md` for claude_log

---

## Quick Reference

```bash
# Environment setup (requires Python 3.11+, uv package manager)
GIT_LFS_SKIP_SMUDGE=1 uv sync

# PyTorch training (single / multi-GPU)
uv run scripts/train_pytorch.py <config_name> --exp_name <name>
torchrun --nproc_per_node=N scripts/train_pytorch.py <config_name> --exp_name <name>

# Serve policy — default checkpoint for environment
uv run scripts/serve_policy.py --env <ENV_NAME>

# Serve policy — custom checkpoint (tyro union branch)
uv run scripts/serve_policy.py --env <ENV_NAME> policy:checkpoint --policy.config <config_name> --policy.dir <checkpoint_dir>

# Serve with data collection
uv run scripts/serve_policy.py --env <ENV_NAME> --collect

# Run tests
uv run pytest
```

## Project Structure & Architecture

For full project tree, model architecture, Pi0 vs Pi0.5 differences, code paths, and transform pipeline, see [`docs/openpi_reference.md`](docs/openpi_reference.md).

## Fork Additions (beyond upstream openpi)

| Module | Purpose |
|--------|---------|
| `src/openpi/cache/` | Multi-level inference cache with interceptor pattern (CP1/CP2/CP3) |
| `src/openpi/collect/` | Forward-hook data collection, writes per-episode HDF5 |
| `src/openpi/models_pytorch/` | Complete PyTorch model port with staged API |
| `scripts/train_pytorch.py` | PyTorch DDP training entrypoint |
| `exp/` | Qdrant-based retrieval experiments (ingest, KNN benchmark, toy servers) |
| `convert.py` | HDF5 demo data → LeRobot format |
| `simple_pytorch_train.py` | Minimal standalone MLP training for quick experiments |

## Caveats

- **JAX path is disabled** — only PyTorch inference works. Do not attempt JAX inference.
- **Pi0.5 only** — all new code assumes `pi05=True`. The `pi05` flag on `Pi0Config` controls many conditional branches.
- `adaRMSNorm` is only in the Action Expert, not PaliGemma backbone. The PyTorch port uses modified HF modules in `transformers_replace/`.
- `action_dim=32` is padded for the largest robot; smaller robots zero-pad.
- Image keys: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` (defined in `model.py`).
