# Fast-WAM filtered Cache continuation — 2026-10-16

New run: `/data/mw_finetune/eval/ff1015_fastwam_li_18c1`.
Source: `/data/mw_finetune/eval/ff1015_fastwam_li_18`, FAIL
`worker_restart_budget_exhausted`, `all_owned_closed=true`.

Native `cache_resume.prepare/select` retains 3,787 error-free accepted outcomes
(including unsuccessful trials) and dispatches only the missing 1,213. The new
run has a 5,000-task FULL_EPISODE_PLAN, a 1,213-task EPISODE_PLAN and the native
CACHE_RESUME selection/pins. Source task records and seeds are unchanged.
Runtime, server Python, model/serving inputs, two replicas, B16, MPS, retrieval
rows/fits, YAML bytes and original driver orchestration are inherited unchanged.
No old run, study, fit, campaign or command was rebuilt/re-staged. No reset was
used. The coordinator's launch.py SHARE and reset.py extensions were preserved.
No server, worker, driver, GPU workload, remote write or transfer was run here.

The native clone normally provisions remote nodes. `filter_failed.resume`
substitutes its local copy step only, leaving native resume selection and plan
construction intact. LOCAL_PLAN_SEAL.json is deliberately not PREPARED.json:
coordinator prepare must provision fresh worker/server paths first. The stage
wrapper now preserves inherited driver/control pins for continuations instead
of rebuilding the driver from the checkout. Other staging paths are unchanged.

## Worker choice still required

The draft retains the explicitly requested timan1×40 + timan107×45 (85 total).
It is not currently launchable at that count: a read-only CPU probe ran the
actual frozen worker admission code, lifting only the same foreign-user gate
as SHARE. It did not override per-worker VRAM admission or acquire a reservation.
GPU3: 49,140 MiB total, 26,797 used, 4,914 headroom, zero pending reservations;
1,024 MiB per renderer. The native calculation fits 17: 40/24/20 refused;
16 admitted. Snapshot: `ff1015_fastwam_li_18c1_ADMISSION.json`.

Choose one before prepare (no topology change was silently applied):

* timan1×16 + timan107×45: 61 workers; the 16-worker timan1 probe passed.
* timan107×45 only: avoids timan1; keeps the source timan107 fleet size.
* keep 40+45 and wait for capacity. With no pending reservations, GPU3 usage
  must fall to at most 3,266 MiB for the existing 40×1,024 MiB admission budget.

The existing `worker_topology.redistribute` knob changes execution counts and
records WORKER_REDISTRIBUTION; canonical batch treatment, max batch, runtime,
cache rows, fits, seeds and task identities stay fixed. Reduced concurrency
changes throughput and may change batch composition. The B16 native numerical
policy is not certified byte-identical across batch compositions, so this is an
explicit operational deviation, not a claim of bitwise-identical trajectories.
Native physical VRAM admission remains enabled. Capacity must still hold when
launched; schedule around other fleets on both worker nodes.

## Coordinator commands

The local continuation already exists. Do not run filter_failed.build or reset
the source. Do not rebuild this continuation or re-stage the source.

```bash
cd /home/weiland/projects/openpi
D=exp/offline_search/rounds/r17/filter_failed
M=exp.offline_search.rounds.r17.filter_failed
RUN=/data/mw_finetune/eval/ff1015_fastwam_li_18c1
CPU=(env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python)

# Run exactly ONE override only after choosing it; omit both to retain 40+45:
"${CPU[@]}" -m "$M.resume" --run "$RUN" --worker-counts timan1=16,timan107=45
# OR:
"${CPU[@]}" -m "$M.resume" --run "$RUN" --worker-counts timan107=45

# Coordinator only, after choosing topology / waiting for admission:
bash "$D/commands/prepare_ff1015_fastwam_li_18c1.sh"
bash "$D/commands/launch_ff1015_fastwam_li_18c1.sh"

# Analysis follows the registered continuation automatically. An explicit
# --r17-run is also supported, and supplying it twice does not duplicate data:
"${CPU[@]}" -m "$M.analyze" --r17-run "$RUN" --output "$D/analysis"

# Repeat the independent CPU coverage/merge proof if desired:
"${CPU[@]}" -m "$M.prove_resume" --run "$RUN"
```

Use the provided filtered launcher; it retains the coordinator's remote worker
command rewrite and the frozen per-worker VRAM admission. No worker source
file changed, so there is no worker-source rsync command to hand over. Prepare
uses the existing public-network rsync transport and CPU staging.

`CONTINUATIONS.json` registers this new segment without editing CAMPAIGNS.json.
`filter_failed.analyze` follows native ancestors, merges by task UID and scores
one outcome per model/suite/size/task/init; original and copied ancestor rows
cannot add extra episodes. It also discovers registered descendants by default.
`filter_failed_knob` does not read ff1015 outcomes; it uses ffk knob lanes, whose
native load_knob reader already follows resume ancestry. No knob analysis change
was needed.

## CPU proof

* Frozen original driver planner: exactly 1,213 records, identical to the
  selected original task records (`ff1015_fastwam_li_18c1_PLANNER.json`).
* 3,787 retained + 1,213 missing = 5,000 unique planned UIDs; zero overlap.
* 65 copied assets/archive/orchestration files byte-identical; filter metadata,
  runtime, server interpreter, serving/batching and all source pins match.
* Private disposable merge fixture adds 1,213 explicitly synthetic outcomes and
  copies all 3,787 ancestor rows: the actual collect path reports exactly 5,000
  unique cell/task/init outcomes, 500 per cell, both when reading the child alone
  and when supplying source plus child. Synthetic data never entered a real run.
* Actual merged outcomes before continuation execution: 3,787, not 5,000.
* Full analyzer completed: 84 cells, 70 currently complete; its new output is
  `ff1015_fastwam_li_18c1_analysis_before_run/`, without writing live run files.
* Continuation and relevant staging tests: see FOCUSED_TESTS.log. The broader
  staging suite has four existing reset-test failures: two fixtures omit the
  now-required campaign.json, and two expect refusals relaxed by the current
  coordinator reset code. These are independent of the continuation branch;
  reset.py and those expectations were not changed. Full log: TESTS.log.

| Suite | Size | Retained | Missing | Unique final coverage |
|---|---:|---:|---:|---:|
| libero_10 | 50 | 500 | 0 | 500 |
| libero_10 | 100 | 500 | 0 | 500 |
| libero_10 | 200 | 500 | 0 | 500 |
| libero_10 | 300 | 287 | 213 | 500 |
| libero_10 | 400 | 0 | 500 | 500 |
| libero_spatial | 50 | 500 | 0 | 500 |
| libero_spatial | 100 | 500 | 0 | 500 |
| libero_spatial | 200 | 500 | 0 | 500 |
| libero_spatial | 300 | 500 | 0 | 500 |
| libero_spatial | 400 | 0 | 500 | 500 |
