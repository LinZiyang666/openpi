<task>
R16 follow-up (owner, 2026-10-08 10:0x): use the finished per-task test runs to raise the knob for tasks whose success rate fell short, and rerun only those tasks. Owner: "成功率低的任务你把IR提升上去，没必要一定重跑更细的IR下降". Build the tooling; I run everything.
</task>

<input>
A changes file (JSON list) that I write: `[{"cell": "pi05_l10_100_k1", "task": 4, "from": "ir32", "to": "c10"}, ...]`. `to` is one of the existing levels (ir45 / ir40 / ir32 / ir25 / off) or `c10` = the k-step pure-inference fallback of that cell (same k). Cells and tasks refer to the frozen R16 per-task tables in `exp/offline_search/rounds/r16/dev_results_conductor/levels.json` and the finished test groups `r16_conductor_test_{pi05,groot}_k{1,2}_v1`.
</input>

<deliverables>
1. `retest` emitter: from the frozen tables + changes file, build per-cell adjusted tables (only the listed tasks change) and resident conductor groups that run ONLY the changed (cell, task) pairs on their original 50 test-A identities each (same UIDs/inits, seeds, horizon, k, recording off, same pool), as new arms with new names (e.g. `r16rt_<cell>_v1`) under new roots `r16_conductor_retest_{model}_k{k}_v1`, plus a queue file in the format `cond_loop2.sh` reads. Reuse the resident multi-method server / episode dispatch path exactly as the test groups do. Must work for any subset of the 32 cells and both models.
2. Merge analysis: for each cell, replace the changed tasks' 50 outcomes (and their decision-log V/M counts) with the retest outcomes, keep the other tasks from the original test run, and report per cell and pooled per size × k: SR and IR (D5 pricing) before vs after, the paired comparison against full-step pure inference (R14 test C10) and against the full-step per-task descent of the same size (R14 size 50, R15 sizes 100/200/500), same estimators as `analyze test --conductor`. Also list per changed task: old level/SR → new level/SR.
3. CPU tests on synthetic outcomes; a short `HANDBACK_RETEST.md` with the exact commands (emit, deploy if any worker bytes change, run via `/home/weiland/projects/openpi_ext/scratch/r16_ops/cond_loop2.sh`-style queue, analyze).
</deliverables>

<constraints>
CPU only: taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python, in /home/weiland/projects/openpi. The GR00T test groups are running now (tmux r16_cond_test): do not touch their processes, roots, journals or logs; do not edit any file the running/next test servers import (resident server, plugin, methods, shared runtime). New code goes in new R16 modules. No servers, GPU, remote ops, git, pkill, rm -rf. Do not rewrite frozen tables or existing results.
</constraints>
