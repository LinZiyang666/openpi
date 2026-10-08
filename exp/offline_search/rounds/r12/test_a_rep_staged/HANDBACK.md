Six staged test-A replicate roots, each containing only `<root>_off`:

- `r12_test_rep_pi05_l10_50`
- `r12_test_rep_pi05_spatial_50`
- `r12_test_rep_groot_spatial_50`
- `r12_test_rep_groot_l10_50`
- `r12_test_rep_pi05_spatial_200`
- `r12_test_rep_pi05_l10_50_periodic_ir32`

CPU verification: **41 passed** (`CPU_TESTS.xml`): 6 definition/hash checks,
24 real plan + sync comparisons with transport mocked (six cells × local/h100
× staged/scratch-installed), and 11 installer/freshness/drift checks. Final server
argv matches each deployed source off arm except root paths/tags; eval500 bytes,
frozen artifacts, served payload hashes and library subsets match. Off selects
`examples.libero.worker_entry` with replan_steps=5.

Install from the repository on a host permitted to write trace_runs:

```bash
cd /home/weiland/projects/openpi
taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m exp.offline_search.rounds.r12.prepare_test_a_rep \
  --install --destination /home/weiland/trace_runs/os_closed_loop/
```

The installer preflights all six roots, refuses existing destinations (including
dangling symlinks), verifies staged definitions against deployed off sources,
and copies definitions only. Installation to trace_runs was not attempted;
there were no servers, drivers, live syncs, chains or remote operations.
