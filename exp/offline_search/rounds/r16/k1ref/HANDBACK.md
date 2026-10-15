# R16 k=1 reference rerun — coordinator handback

CPU implementation only. No policy server, worker, chain, monitor, GPU operation,
remote command, transfer, credential read, or commit was performed. All new
experiment definitions and results are under this directory. R16's frozen roots,
tables, outputs, code and worker island were not changed.

The corrected owner reference governs: size 50 uses
`r16d_<model>_<suite>_50_k1_pure_c10`; sizes 100/200/500 use
`r16d_<model>_<suite>_100plus_k1_pure_c10`, with identical B task/init manifests.
Test references are the four R16 phase-1 k1 pure arms, each with the same 500 A
task/init identities as its cells. Phase-1 reference UIDs remain phase-1 UIDs;
they are never substituted for an unmeasured task-descent UID cohort.

`BASELINE_AUDIT.json` reproduces **31 changed resolved selections / 19 tasks with
an unmeasured next dev level**, out of 160 tasks. The 19 are unresolved, not
claimed final selections. The source is the hash-verified frozen
`dev_results_conductor/dev.json`; its original input ledger is retained.

| cell | task → next level |
|---|---|
| pi05_l10_50_k1 | 2 → ir40 |
| pi05_l10_100_k1 | 5 → ir25 |
| pi05_l10_200_k1 | 2 → ir40; 5 → ir25 |
| pi05_l10_500_k1 | 2 → ir40; 4 → off |
| pi05_spatial_100_k1 | 0 → ir40 |
| pi05_spatial_200_k1 | 0 → ir32; 8 → ir25 |
| pi05_spatial_500_k1 | 0 → ir40; 8 → off |
| groot_l10_50_k1 | 7 → ir40 |
| groot_l10_200_k1 | 0 → ir40 |
| groot_l10_500_k1 | 0 → ir32 |
| groot_spatial_100_k1 | 0 → ir40 |
| groot_spatial_200_k1 | 0 → ir40 |
| groot_spatial_500_k1 | 0 → ir32; 7 → ir25; 9 → ir40 |

The sealed inventory contains **802 measured dev cell/task/level cohorts**, **197
measured A cell/task/level cohorts**, and four pure A references. Original A and
all k1 retest v1/v2/v3 cohorts were loaded through R16's strict accepted-attempt
loaders, including superseded v1 measurements. About 4.50 GB of actual source
inputs were audited. Fits, K, manifests, UIDs, launch records, decision counts,
LOEO evidence, tails and D5 prices use the existing validators. The inventory is
an immutable reuse cache; do not rerun `bootstrap` over it.

| phase | new episodes |
|---|---:|
| smoke | 16: eight arms × two B episodes; four workers |
| knob-off A | **3,550**: 71 missing tasks × 50; 89 tasks reused |
| dev supplement | **up to 570** in the first wave; **up to 1,770** over at most four waves |
| freeze | 0 |
| frozen-level A | outcome-dependent; **at most 2,350** |
| each raise generation | 50 × newly missing raised task/level cohorts; zero for reused levels |
| final analysis | 0 |

Dev bounds are before pruning cells that pass the full-system rule. Such cells
are not tuned: all their final levels are `off`, with their measured off SR/IR.
There are 28 known resolved frozen levels missing A, plus at most 19 unresolved
dev selections; this gives the 47 × 50 initial-test upper bound. Exact adaptive
counts are printed and sealed in each `runs/<generation>/plan.json` before any
launch. No exact outcome-dependent total is asserted in advance. Raises can
require at most five generations of new measurements; a sixth emit can be an
analysis-only completion check. `EPISODES.json` records these counts and bounds.

Dev visits `c10 → ir45 → ir40 → ir32 → ir25 → off`, stopping at the first failure.
`100*(successes-reference_successes) >= -150` passes. Each emitted wave contains
only one currently missing level per task; the next wave is emitted only after
all prior cohorts complete. Already measured levels are consumed without dispatch.
Test deficits use integers: ≤2 passes, 3–4 raises one level, ≥5 selects c10.
Measured intermediate levels can satisfy successive raises without new episodes.
An error or incomplete cohort cannot count as a success-rate loss.

## 1. Build and verify the immutable local server snapshot

Run these commands in a coordinator shell. The destination below must be fresh.
The separate `/home/weiland/projects/openpi_r16_k1ref_cpu` worktree was used only
for CPU validation; the commands below build the serving snapshot from final code.

```bash
cd /home/weiland/projects/openpi
export R16_K1REF_OUTPUT=/home/weiland/projects/openpi/exp/offline_search/rounds/r16/k1ref
export R16_K1REF_SNAPSHOT=/home/weiland/projects/openpi_r16_k1ref_server
K1_CPU=(taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python)
"${K1_CPU[@]}" -m exp.offline_search.rounds.r16.k1ref.snapshot build \
  --destination "$R16_K1REF_SNAPSHOT"

# Frozen B pool YAMLs are ignored by git and absent from the old local manifest.
# Complete construction before using the snapshot; copy and hash both verbatim.
for K1_SUITE in libero_10 libero_spatial; do
  K1_POOL_REL="exp/offline_search/closed_loop/ops/h100/bpool_${K1_SUITE}.yaml"
  install -D -m 444 "$K1_POOL_REL" "$R16_K1REF_SNAPSHOT/$K1_POOL_REL"
  sha256sum "$R16_K1REF_SNAPSHOT/$K1_POOL_REL" >> "$R16_K1REF_SNAPSHOT/R16_SNAPSHOT.sha256"
done
chmod a-w "$R16_K1REF_SNAPSHOT/R16_SNAPSHOT.sha256"

cd "$R16_K1REF_SNAPSHOT"
"${K1_CPU[@]}" -m exp.offline_search.rounds.r16.k1ref.snapshot verify \
  --destination "$R16_K1REF_SNAPSHOT"
"${K1_CPU[@]}" -m exp.offline_search.rounds.r16.k1ref.workflow verify
```

The builder creates a detached worktree at HEAD, copies ignored R16 artifacts and
the legacy YAML templates, and restores only these serving files:

| file | R16 source |
|---|---|
| packages/openpi-client/src/openpi_client/msgpack_numpy.py | 231a1cf7 |
| src/openpi/cache/groot/batcher.py | 30505063 |
| src/openpi/cache/groot/staged.py | 30505063 |

Every entry of the original `conductor_deployment/local.sha256` is checked.
Repository paths are rebased to the snapshot; absolute external frozen fits and
groups retain their original paths and hashes. The original manifest is not
edited. `R16_SNAPSHOT.sha256` also pins the new code and CODE_LOCK. Python source
checks in R16's absolute-path locks read the snapshot's serving bytes; outcome,
fit and identity paths retain their original bindings. Measured snapshot files
are read-only. Outputs continue to the original k1ref tree through
`R16_K1REF_OUTPUT`.

No worker files need changing or rsync. The worker imports only the existing R16
conductor/worker/client modules; the new strategy and UID join run in the local
controller. To recheck the existing island, the coordinator may run:

```bash
tether exec timan107 -- bash -c 'cd /scratch/zixuans8/openpi_r16_conductor_v1 && sha256sum -c worker.sha256'
```

There is no tether push, pull, expose, or hidden source sync. Existing
`control.remote` exec calls manage the existing agent lease and fleet lock.

## 2. Smoke — two episodes per arm, four workers

The sealed smoke and off queues are already emitted. Do not re-emit them.

```bash
bash "$R16_K1REF_OUTPUT/run_queue.sh" "$R16_K1REF_OUTPUT/runs/smoke/queue.txt" 4
"${K1_CPU[@]}" -m exp.offline_search.rounds.r16.k1ref.workflow check-smoke
```

Smoke has k1 c10 and ir45 in each suite/model, two original B identities per arm.
It is excluded from every official analysis. π0.5 smoke uses one process, GR00T
two (one per checkpoint); ordinary queues keep **π0.5 three processes / GR00T
four processes**. No different layout was measured. The whole local RTX 4090 is
the assigned device. The old free-memory gates and sequential starts remain;
inspect actual first-inference memory during smoke before the full queues.

The fixed driver is **23159**. Servers are 23150–23152 for ordinary π0.5 and
23150–23153 for ordinary GR00T, within the authorized 23150–23155 allocation.
The adapter explicitly refuses **23179**, reserved by the live Fast-WAM driver.
Requested busy ports fail; there is no alternate-port selection.

Inspect `resident/<port>.json`, startup and resident-binding rows: correct model,
suite checkpoint, method/fit, `miss_steps=1`, no added fit GPU bytes. Actual calls
must have k1; policy tails are HIT/no vision/no policy call. The journals use
original UIDs and new groups' own log directories. `progress.json` must attest
timan107, the exact A/B pools, seed 2026101401, replan=5, ten settling steps,
224/256 resize, and 520/220 limits. MISS commitment remains ten controls through
the existing blind tail. No source tree or pool is written remotely.

## 3. Knob-off A — 3,550 new episodes

```bash
bash "$R16_K1REF_OUTPUT/run_queue.sh" "$R16_K1REF_OUTPUT/runs/off/queue.txt" 64
```

All 16 cells have a sparse off definition; every task's serving table is off.
Only the 71 missing cohorts are dispatched. The next preparation command refuses
until this queue is DONE and all selected cohorts validate.

## 4. Dev supplement — emit, run, analyze one wave at a time

The first command computes the full-system decision from complete knob-off A,
then omits all tuning for qualifying cells. Do not pre-emit all four waves.

```bash
for K1_DEV_GEN in 1 2 3 4; do
  K1_DEV_NAME=$(printf 'dev%02d' "$K1_DEV_GEN")
  K1_DEV_QUEUE="$R16_K1REF_OUTPUT/runs/$K1_DEV_NAME/queue.txt"
  if [[ ! -f "$K1_DEV_QUEUE" ]]; then
    "${K1_CPU[@]}" -m exp.offline_search.rounds.r16.k1ref.workflow dev \
      --generation "$K1_DEV_GEN"
  fi
  [[ -f "$K1_DEV_QUEUE" ]] || break
  bash "$R16_K1REF_OUTPUT/run_queue.sh" "$K1_DEV_QUEUE" 64
done
```

Each following emit strictly analyzes the completed previous wave and reuses the
frozen original outcomes. A failed level keeps its predecessor; lower levels are
never emitted for that task. `DEV_COMPLETE` means proceed to freeze.

## 5. Freeze all 16 operational tables

```bash
"${K1_CPU[@]}" -m exp.offline_search.rounds.r16.k1ref.workflow freeze
```

This refuses with any unresolved task. `freeze/levels.json`, `dev.json`,
`TABLE_FREEZE.json`, and `SEAL.json` bind the new selection, code, inventory and
actual supplement inputs. Full-system cells have all-off operational tables and
are labeled untuned. No old R16 table or freeze is changed. On resume, skip this
command if `freeze/TABLE_FREEZE.json` already exists.

## 6. Test the frozen levels — missing cohorts only

```bash
if [[ ! -f "$R16_K1REF_OUTPUT/runs/test00/queue.txt" ]]; then
  "${K1_CPU[@]}" -m exp.offline_search.rounds.r16.k1ref.workflow test --generation 0
fi
bash "$R16_K1REF_OUTPUT/run_queue.sh" "$R16_K1REF_OUTPUT/runs/test00/queue.txt" 64
```

Generation zero dispatches only missing frozen-level A cohorts, with original
task-descent UIDs and init indices 0–49. Full-system cells reuse off throughout.
It does not preemptively replace an unmeasured frozen-level cohort with a raise.

## 7. Each raise generation — emit from the previous analysis

```bash
for K1_TEST_GEN in 1 2 3 4 5 6; do
  K1_PREVIOUS=$(printf '%s/analyses/g%02d/analysis.json' "$R16_K1REF_OUTPUT" "$((K1_TEST_GEN-1))")
  if "${K1_CPU[@]}" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["selection"]["complete"] else 1)' "$K1_PREVIOUS"; then
    break
  fi
  K1_TEST_NAME=$(printf 'test%02d' "$K1_TEST_GEN")
  K1_TEST_QUEUE="$R16_K1REF_OUTPUT/runs/$K1_TEST_NAME/queue.txt"
  if [[ ! -f "$K1_TEST_QUEUE" ]]; then
    "${K1_CPU[@]}" -m exp.offline_search.rounds.r16.k1ref.workflow test \
      --generation "$K1_TEST_GEN" --previous "$K1_PREVIOUS"
  fi
  bash "$R16_K1REF_OUTPUT/run_queue.sh" "$K1_TEST_QUEUE" 64
done
```

For example, generation 1 uses `analyses/g00/analysis.json`, generation 2 uses
`g01`, and so on. Each sealed plan pins its previous analysis hash. Only raised
tasks lacking that exact A outcome are dispatched. Empty queues launch nothing;
they can reflect reuse or the final completion check. Every non-full-system task
must pass or sit at c10 before final analysis is allowed.

## 8. Final analysis

```bash
"${K1_CPU[@]}" -m exp.offline_search.rounds.r16.k1ref.workflow analyze \
  --output "$R16_K1REF_OUTPUT/results"
```

`results/RETEST.md` has the R16-style before/after cell table, the off-rule
decision table, all 160 final task levels/SRs/deficits, per-cell and size-pooled
paired comparisons and CIs. `retest.json` preserves every final Y/N/V/M/cost
array, provenance, selected path and rule. `summary.json` has exactly 16 rows:
`model, suite, size, sr, baseline_sr, diff_pp, ir, pure_ir, rule`, where rule is
`knob` or `full_system`. The baseline is always k1 pure.

Numerical estimators and D5 prices are the original R16 functions: SR means,
IR=sum(cost)/sum(N), fixed task mixture, 5,000 shared suite/task/init bootstrap
draws, seed 20261005, and the existing paired confidence family. The legacy
calculator's reference slots both receive the same k1 pure array internally;
duplicate slots are removed and labeled `k1_pure` in the output. Reused reference
episodes are weights, not extra independent trials. Post-test selection remains
exploratory; descriptive intervals do not adjust for selection. Policy RNG draw
order retains R16's concurrency dependence.

Resume a started queue by repeating its `run_queue.sh` command; accepted episodes
and attempt fences remain. The old receipt-authenticated lifecycle, global
orchestration/fleet locks and 90-second worker lease apply. Never re-emit a
started/frozen generation, overwrite an existing result, or use a legacy R16 loop
whose default driver is 23179. No monitors are launched by this handback.

## CPU evidence

- 79 focused tests pass (`cpu_snapshot_final.xml` also includes one legacy
  synthetic emitter check, for 80 passes).
- 350 CPU tests pass in the restored snapshot, with four path-dependent legacy
  metadata assertions deselected (`cpu_regression_final.xml`). Those four were
  also called directly with their canonical module paths and snapshot source
  reads; all pass (`canonical_audit_checks.json`). Thus 354 distinct checks pass.
- TCP driver/worker integration and census tests were excluded to honor the
  no-server/no-worker constraint. Socketpair serialization tests are CPU-only.
- The snapshot's original local SHA list verifies fully with the three R16
  restorations. Prepared smoke/off groups pass sealed-plan, sparse-identity,
  fit and fixed-port validation. The original A references are 91.8% / 98.4%
  for π0.5 and 88.2% / 95.0% for GR00T, each from 500 paired A episodes.

Focused CPU rerun after building the serving snapshot:

```bash
"${K1_CPU[@]}" -m pytest -q -p no:cacheprovider \
  exp/offline_search/rounds/r16/k1ref/test_workflow.py
```

All cohort measurements, actual smoke memory checks and final measured output
remain coordinator work. No SR or rule decision for the missing episodes is
fabricated in this handback.
