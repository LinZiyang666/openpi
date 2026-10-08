# CPU reproduction

Run from the repository root. Every script and output is in this directory. Prefix every Python command with:

```bash
taskset -c 10-21,54-65 env PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python
```

Pass these script paths in dependency order, with the common directory prefix `exp/offline_search/rounds/r14/astra/`:

1. `inventory.py`
2. `early_grid.py`, then `pilot_selection.py`
3. `dev_extract.py`
4. `partition.py` (threaded CPU), `dev_partition.py`, `continuity.py`
5. `allocation.py`, then `task_study.py`
6. `selfcheck.py`, `controller_check.py`, then `summarize.py`
7. `freeze.py`, then `audit.py`

The final freeze records timestamps and hashes, so rerendering it is an explicit new freeze. Do not run the inherited production CLIs: they own other directories. No command here launches an environment, contacts a remote machine or uses a GPU. The blocked-metric implementation bounds working memory; all source arrays and fit provenance come from B-library artifacts or the explicit B-development root allowlist.
