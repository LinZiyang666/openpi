# R16 owner-selected task retests

New modules: `retest.py` emits adjusted frozen-fit wrappers and sparse groups; `retest_runner.py` runs them through the existing resident/conductor/worker stack; `retest_analysis.py` merges accepted retest cohorts with original A. `retest_loop.sh` accepts the same `@local /absolute/group.json` lines as `cond_loop2.sh`. Use this new loop: the original loop selects dev/test queues and calls the original runner, which would assign new UIDs. No existing runtime, server, worker, protocol, table, result or conductor lock was edited. `RETEST_LOCK.json` separately anchors the unchanged measurement code and the new tooling.

Each requested `(cell,task)` gets its original 50 A inits **and exact original task UIDs**. New `r16rt_<cell>_v1` names select the adjusted method and label journals/decision logs. The local controller changes only the sparse plan and UID-to-arm join; existing attempt fences, endpoint dispatch, worker attestation, simulator/rollout seeds, live K, checkpoint, horizon and recording-off behavior remain. Missing suites get one dormant frozen method, with zero dispatched episodes, to satisfy the unchanged launcher. Fits only call the existing `set_table`; no refit. Resident fits add zero GPU bytes; server counts/memory remain the existing defaults (three π0.5 processes; two GR00T processes per suite).

Choices: reject duplicate/unknown tasks, stale `from`, equal levels and unknown levels. Your file is authoritative: every different existing `to` is accepted, including `off`; the tool does not infer improvements or redo dev descent. `c10` is the same-k pure fallback, with cap zero. Only listed tasks change in the 32-cell `levels.json` snapshot. New roots are `r16_conductor_retest_{model}_k{k}_v1`; only selected model/k groups are emitted. Original 500-identity manifests stay byte-identical as pairing templates; dispatch manifests contain only selected cohorts. The new `PLAN_FREEZE.json` seals the owner selection, timestamp, adjusted tables, source hashes and group hashes before running, separately from the original dev freeze.

Commands below are for the coordinator; none were run here. Write your list to `/home/weiland/projects/openpi_ext/scratch/r16_ops/retest_changes_v1.json`. Wait for the original test line to finish before running retests; existing fleet locks refuse overlap. Emission reads only immutable definitions for selected groups, with zero outcome reads. Analysis requires all four original groups and all emitted retest groups to have `DONE.json` and complete error-free accepted cohorts.

```bash
cd /home/weiland/projects/openpi
CPU=(taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python)
"${CPU[@]}" -m exp.offline_search.rounds.r16.retest verify
"${CPU[@]}" -m exp.offline_search.rounds.r16.retest emit \
  --changes /home/weiland/projects/openpi_ext/scratch/r16_ops/retest_changes_v1.json \
  --generation v1
```

No worker deployment/sync is needed: worker bytes, protocol payloads and imported method classes are unchanged. The existing `/scratch/zixuans8/openpi_r16_conductor_v1` island works; the local tree already contains the new controller modules and freshly emitted fits. Do not regenerate the old deployment or replace the running dev/test queues.

CPU proof: **236 tests pass** (41 new retest tests plus the existing 195), with synthetic outcomes only. Tests cover arbitrary cell/model/k subsets, exact original UIDs/seeds, immutable lower-fit arrays, sparse resume with a changed server count, stale-attempt fencing, C10/off behavior, full cohort replacement, unchanged task arrays, original estimators and tamper rejection. The real table freeze and source locks also verify without outcome reads.

At a tmux shell, run the new, independently logged queue:

```bash
cd /home/weiland/projects/openpi
GROOT_SERVERS_PER_SUITE=2 bash exp/offline_search/rounds/r16/retest_loop.sh \
  /home/weiland/projects/openpi/exp/offline_search/rounds/r16/retest_v1/queue_retest.txt \
  /scratch/zixuans8/openpi_r16_conductor_v1
```

This uses the `cond_loop2.sh` queue/retry pattern and existing runner lifecycle. Log: `retest_v1/retest_loop.log`. Ctrl-C stops the loop; repeat the same command to resume, keeping completed episodes and reissuing fenced in-flight work. If separately allocated memory permits six GR00T processes, change the environment value to `3`; use `2` to return to four. Omitting it uses the coordinator's existing `r16_ops/groot_sps` value, falling back to two. The runner authenticates only its own new roots/receipts.

Checks after the first episodes: journal `yaml_id=r16rt_<cell>_v1`, `task_uid=r16t_<cell>_task_descent:eval:<task>:<init>`; only changed tasks/inits 0..49. Decision `arm/yaml_id/bundle` all equal the new arm; `uid` keeps the original prefix; MISS `miss_k` equals the cell k. Vision rows have the adjusted `r14.descent_level` and new wrapper `r14.fit_sha256`; blind/tail rows have fit identity but may omit the level. Tail rows remain HIT/no vision/no policy call. Worker probe must still attest timan107/A pools/seed template/horizon; launch has local/timan107 and `STAGE1_ONLY=0`.

After completion:

```bash
cd /home/weiland/projects/openpi
CPU=(taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python)
"${CPU[@]}" -m exp.offline_search.rounds.r16.retest_analysis \
  --plan exp/offline_search/rounds/r16/retest_v1/plan.json \
  --output exp/offline_search/rounds/r16/retest_v1/results
```

`results/RETEST.md` and `results/retest.json` report before/after per cell and size×k, same paired comparisons against R14 full C10 and same-size R14/R15 descent, shared bootstrap CIs, after-vs-before comparisons, changed task old/new level/SR, input hashes and every merged Y/N/V/M/cost array. Exact original estimators and source-derived D5 prices are reused; only the selected whole task cohorts are replaced. Other tasks retain original arrays byte-for-byte. Use a fresh generation/output for a new selection; existing selections/results are never overwritten. A second generation still validates `from` against the original frozen dev table and merges against original test A; it is not a cumulative incremental edit.

These adjusted results are **exploratory, selected after viewing test A**. Their descriptive paired CIs use the existing estimator and do not account for that selection. Original frozen test results remain unchanged. Policy process RNG initialization is unchanged; draw order remains concurrency-dependent as in the original conductor, so identical seeds do not promise identical sampled actions across reruns. No real-environment run was performed; CPU validation uses synthetic outcomes only. Validation counts/checksums are in `retest_validation.json` and `RETEST.sha256`.

## Additive v2 boundary follow-up

Read-only preflight verified the supplied **7 disjoint tasks / 350 new episodes / 4 model×k groups**, all `to: ir45`. Together with v1's 46 tasks, the combined result replaces 53 tasks. No v2 roots or outputs were emitted here. The selection is the explicit JSON list; no floating SR threshold is reapplied.

`retest_combined.py` adds validation and multi-plan analysis under a separate `RETEST_COMBINED_LOCK.json`. All sealed v1 code/locks/roots/results remain unchanged; no worker sync is needed. v2 uses the original emitter/runner/queue. The combined analyzer retains v1 outcomes and counts, adds v2's cohorts (including different tasks in the same cell), and retains original A everywhere else. It rejects overlapping tasks, duplicate generations/groups/plans, inconsistent freezes/original pairing, and incomplete cohorts. The original estimators/prices and exploratory interpretation remain.

Run these commands yourself; the first is read-only, the second emits only v2:

```bash
cd /home/weiland/projects/openpi
CPU=(taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python)
"${CPU[@]}" -m exp.offline_search.rounds.r16.retest_combined validate \
  --changes /home/weiland/projects/openpi_ext/scratch/r16_ops/retest_changes_v2_boundary.json \
  --against-plan exp/offline_search/rounds/r16/retest_v1/plan.json --generation v2
"${CPU[@]}" -m exp.offline_search.rounds.r16.retest emit \
  --changes /home/weiland/projects/openpi_ext/scratch/r16_ops/retest_changes_v2_boundary.json \
  --generation v2
GROOT_SERVERS_PER_SUITE=2 bash exp/offline_search/rounds/r16/retest_loop.sh \
  /home/weiland/projects/openpi/exp/offline_search/rounds/r16/retest_v2/queue_retest.txt \
  /scratch/zixuans8/openpi_r16_conductor_v1
"${CPU[@]}" -m exp.offline_search.rounds.r16.retest_combined analyze \
  --plan exp/offline_search/rounds/r16/retest_v1/plan.json \
  --plan exp/offline_search/rounds/r16/retest_v2/plan.json \
  --output exp/offline_search/rounds/r16/retest_v1_v2_results
```

The new output contains `levels.json`, `retest.json`, and `RETEST.md`: combined tables, per-cell/size×k/per-k SR and IR, both paired reference comparisons/CIs, after-vs-before comparisons, changed-task generation labels, task provenance and merged Y/N/V/M/cost arrays. Existing output directories are refused. Use both plans for the combined result; the old single-plan analyzer still treats each generation separately. New validation evidence/checksums are `retest_combined_validation.json` / `RETEST_COMBINED.sha256`; the prior `RETEST.sha256` retains the handback's pre-addendum hash.

CPU validation: **253 tests passed**, including 17 new combination tests. Synthetic checks cover seven-pair-only v2 dispatch, exact original UIDs/seeds, unchanged v1 bytes, same-cell accumulation, accepted-attempt V/M/N counts, order independence, overlap/conflict rejection, the exact three-of-50 boundary, and equality with the sealed single-generation analysis/estimators. No real outcomes were read or analyzed here.

## Explicit v3 c10 correction superseding seven v1 pairs

`retest_superseding.py` adds the seven-pair correction under its own `RETEST_SUPERSEDING_LOCK.json`; all prior tooling, locks, frozen tables, v1/v2 plans, roots and outputs remain unchanged. The owner file is hash-pinned. Read-only preflight verified all seven original frozen `from` levels and each pair's v1 `to: ir45`, with no overlap in v2. The new dispatch is **7 tasks / 350 episodes / 4 model×k groups**, using only the supplied pairs, new `r16rt_<cell>_v3` arms and `r16_conductor_retest_<model>_k<k>_v3` roots. It preserves original identities, UIDs, inits, seeds, A pool, horizon, manifests, lower fit bytes and the existing runner/worker path. `c10` uses cap zero and the cell's same-k pure fallback. No floating SR threshold is reapplied.

Emission and analysis require the explicit previous plans. v3's legacy-compatible `changes.json` contains cell/task/from/to; `owner_changes.json` retains the exact supplied bytes including `supersedes: v1`. The new plan and `PLAN_FREEZE.json` seal both selections, supersession metadata, previous plan hashes, new source lock and adjusted levels. The unchanged `retest_loop.sh` and `retest_runner.py` accept these groups. No worker sync is needed.

The combined analyzer requires complete, accepted error-free cohorts for v1, v2 and v3, including the superseded v1 measurements. It permits only the listed **v3 c10 → v1 ir45** supersessions; every other overlap is rejected before outcome reads. Final reports replace the seven v1 task cohorts with v3 and keep every other v1/v2/original cohort, independent of plan argument order. The final combined selection still has **53 unique changed tasks**: 39 from v1, seven from v2 and seven from v3. `retest.json` includes final levels, changed-task generation labels, provenance, all final Y/N/V/M/cost arrays and each superseded v1 task's level, SR/IR/counts, arrays and arm/group identity. `RETEST.md` shows final levels/SRs and a separate superseded-measurement reference table. Both original paired reference comparisons, pooled estimators, CIs, D5 pricing and after-vs-before comparisons are unchanged. These remain exploratory post-test selections.

Run the following commands yourself. Validation is read-only; **v3 emission and the GPU run were left to the coordinator**. Use a fresh analysis directory. The existing v1, v2 and v1+v2 results must remain intact.

```bash
cd /home/weiland/projects/openpi
CPU=(taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python)
"${CPU[@]}" -m exp.offline_search.rounds.r16.retest_superseding validate \
  --changes /home/weiland/projects/openpi_ext/scratch/r16_ops/retest_changes_v3_c10.json \
  --against-plan exp/offline_search/rounds/r16/retest_v1/plan.json \
  --against-plan exp/offline_search/rounds/r16/retest_v2/plan.json \
  --generation v3
"${CPU[@]}" -m exp.offline_search.rounds.r16.retest_superseding emit \
  --changes /home/weiland/projects/openpi_ext/scratch/r16_ops/retest_changes_v3_c10.json \
  --against-plan exp/offline_search/rounds/r16/retest_v1/plan.json \
  --against-plan exp/offline_search/rounds/r16/retest_v2/plan.json \
  --generation v3
GROOT_SERVERS_PER_SUITE=2 bash exp/offline_search/rounds/r16/retest_loop.sh \
  /home/weiland/projects/openpi/exp/offline_search/rounds/r16/retest_v3/queue_retest.txt \
  /scratch/zixuans8/openpi_r16_conductor_v1
"${CPU[@]}" -m exp.offline_search.rounds.r16.retest_superseding analyze \
  --plan exp/offline_search/rounds/r16/retest_v1/plan.json \
  --plan exp/offline_search/rounds/r16/retest_v2/plan.json \
  --plan exp/offline_search/rounds/r16/retest_v3/plan.json \
  --output exp/offline_search/rounds/r16/retest_v1_v2_v3_results
```

The focused CPU check is rerunnable with the same prefix:

```bash
"${CPU[@]}" -m pytest -q -p no:cacheprovider \
  exp/offline_search/rounds/r16/test_retest_superseding.py
```

CPU validation: **278 tests passed**, including **28 new supersession tests** and 250 existing regressions. New evidence is recorded in `retest_superseding_tests.xml`, `retest_superseding_validation.json` and `RETEST_SUPERSEDING.sha256`. The new tests cover exact sparse dispatch/UIDs/seeds, immutable prior bytes and lower fits, original frozen `from` validation, source/target/authority rejection, mixed v1 cells, final arrays and retained v1 counts, five-of-50 outcomes, order independence, missing cohorts, all existing reference estimators and v1+v2 numerical parity. The three TCP server integration tests are excluded from this CPU-only regression check. All **732 protected existing files** were byte-checked unchanged; the handback retains its entire prior prefix.

The complete CPU regression command used was:

```bash
"${CPU[@]}" -m pytest -q -p no:cacheprovider \
  exp/offline_search/rounds/r16/test_r16.py \
  exp/offline_search/rounds/r16/test_conductor.py \
  exp/offline_search/rounds/r16/test_groot_servers.py \
  exp/offline_search/rounds/r16/test_retest.py \
  exp/offline_search/rounds/r16/test_retest_combined.py \
  exp/offline_search/rounds/r16/test_retest_superseding.py \
  tests/conductor tests/libero_groot/test_dynamic_bundle_guards.py \
  tests/serving/test_router_bundle_rebind.py \
  --ignore=tests/conductor/test_integration.py
```
