# R14 per-task descent handback

Four launch-ready B roots are under `/home/weiland/trace_runs/os_closed_loop/`:

| Root | Fleet | Arms / episodes per arm |
| --- | --- | --- |
| `r14_descent_pi05_l10_50` | @h100 | 8 / 300 |
| `r14_descent_pi05_spatial_50` | @h100 | 8 / 300 |
| `r14_descent_groot_l10_50` | @local | 8 / 300 |
| `r14_descent_groot_spatial_50` | @local | 8 / 300 |

For each cell, the arm prefix is `r14d_<cell>_`. Queue order is `pure_c10`, `periodic_ir45`, `periodic_ir40`, `task_periodic_ir40`, `periodic_ir32`, `task_periodic_ir32`, `periodic_ir25`, `off`. The full runner lines are in [queue_dev.txt](queue_dev.txt). All 32 arms reuse SHA-verified R11/R13 periodic/off/C10 or frozen R14 allocation artifacts; no refit. Total: 9,600 episode runs. Recording is off; existing `--os-r14-audit` decision logging is on. No fleet is bound inside a root.

Each suite has one common list of 300 B identities, 30 per task, paired across every arm and both models. Seed: `2026101401` (spatial selection RNG adds one). Every identity is outside both actual served subsets. [selection.json](selection.json) contains every pair, both subset exclusions and exact intersections with each prior B manifest. In each suite, 213 selected pairs appeared in prior B selections/reservations; this is development reuse, not independent validation. Each root's `dev_selection.json`, `dev_contracts.json` and `ROOT.json` bind the selection and actual fits.

CPU verification passed: 12 tests covering 3,360 paired plugin decisions across all six uniform levels and mixed tables, exact action bytes, keyed coins/caps, interleaved connections, accepted-history retries, off and C10 endpoints, staging and scratch installation. A follow-up loader/install test passed for all four emitted artifacts through actual `PluginRuntime`. Both fleet plans passed for all four dev roots and all four example test roots with remote, process-launch and server operations blocked. See [verification.json](verification.json), [tests.log](tests.log), [loader_tests.log](loader_tests.log), [plans_dev/audit.json](plans_dev/audit.json), and [emitter_smoke/plans/audit.json](emitter_smoke/plans/audit.json). Protected serving sources and frozen catalogs are unchanged. [cap_tables.json](cap_tables.json) preserves periodic caps and Astra's frozen task caps/priorities for comparison with the measured task boundaries.

The CPU test-A emitter accepts two JSON files:

* `levels.json`: `{ "pi05_l10_50": { "0": "off", "1": "ir32", ... "9": "c10" }, ... }`; explicitly supply all ten task IDs for each included cell. Allowed levels: `off`, `ir25`, `ir32`, `ir40`, `ir45`, `c10`.
* `uniform.json`: `{ "pi05_l10_50": "ir32", ... }`, with exactly the same cells.

Run from `/home/weiland/projects/openpi`:

```bash
CPU=(taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python)
D=exp/offline_search/rounds/r14/descent
"${CPU[@]}" -m exp.offline_search.rounds.r14.descent.prepare emit-test-a --levels "$D/levels.json" --uniform "$D/uniform.json" --out "$D/test_a_selected"
"${CPU[@]}" -m exp.offline_search.rounds.r14.descent.verify verify-test-a --out "$D/test_a_selected"
"${CPU[@]}" -m exp.offline_search.rounds.r14.descent.verify plan-test-a --out "$D/test_a_selected"
# Later installation, after the task table and uniform levels are chosen:
"${CPU[@]}" -m exp.offline_search.rounds.r14.descent.prepare install-test-a --out "$D/test_a_selected"
```

This stages `r14_descent_test_<cell>` roots with `pure_c10`, `uniform_<level>`, and `task_descent` arms, each using the byte-preserved R11 eval500 A template, 500 episodes and recording off. `<out>/queue_test_a.txt` already contains final-path runner lines. Installation verifies definition/fit hashes and refuses existing destinations. The mixed tables in `levels.example.json` / `uniform.example.json` and `emitter_smoke/` are CPU examples, not selected results. No test-A root has been installed in the trace directory.

Shared runtime changes: **none**. Dev roots need no code deployment. Test-A uses only the new opt-in `descent.recipe:TaskDescentKnob`; deploy `descent/__init__.py` and `descent/recipe.py` alongside the existing R14 runtime before launching it. Exact hashes are in [TEST_A_RUNTIME_SOURCES.sha256](TEST_A_RUNTIME_SOURCES.sha256). Enabled tasks delegate to unchanged R14 `task_periodic`; off tasks bypass the knob with an explicit mask, and C10 uses cap zero. The table never uses a large surrogate cap for off.

The coordinator still chooses the SR tolerance and freezes the task/uniform level tables from B descent outcomes, using pure C10 as the reference and stopping at the first failing lower level. No new outcomes were used here. No git, remote/tether operation, sync/chain, server, GPU or launch was run. Python used CPU affinity `22-37,66-81`, CUDA hidden and one BLAS/OpenMP thread; pytest scratch was outside the repository with its cache disabled. Writes were confined to `rounds/r14/descent/`, the four allowed dev roots and `/tmp` test scratch.
