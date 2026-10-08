# Reproduction

From the repository root, prefix every invocation with:

```bash
taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src CUDA_VISIBLE_DEVICES= .venv/bin/python
```

Run these scripts in this directory in order: `inventory.py`, `extract.py`, `replay_study.py`, `robust_study.py`, `static_study.py`, `signal_quality.py`, `periodic_and_power.py`, `selfcheck.py`, `freeze.py`, `validation_selection.py`, `write_docs.py`, `audit.py`. Early `early.py` was run once before calibration and must not be rerun over the final handoff. `freeze.py` refuses a changed payload at an existing freeze timestamp. No command starts a rollout or policy model. Imported R11 library code is read-only; the local audit hook fences writes and disallows A/non-library inputs.

The exploratory off-path replay preceded the robust multi-path scalar fit. Keep both evidence tables. Development holdout results were used for global width selection and are not a final independent validation claim. Repeating extraction uses the immutable completed root summaries and cached numerical traces; source hashes in the trace metadata and audit remain authoritative.
