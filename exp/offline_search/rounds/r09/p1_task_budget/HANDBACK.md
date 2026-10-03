# S-P1 handback — per-task call budgets

The twelve standard-mode discovery arms are ready in
`/home/weiland/trace_runs/os_closed_loop/r09_p1`. Every arm selects the same
300 official pairs: tasks 0–9 × inits 0–29. No closed-loop experiment, remote
installation, sync, chain, or local GPU computation was performed.

## Implementation and frozen selection

`TaskBudgetController` dispatches at episode reset to an unchanged, verified
R8 fitted controller. Zero-budget tasks use `BlindAWM` directly. CU tasks use
the R8 `CallController` at calibrated `rho=.3`, including its stall tracker,
lottery, cooldown, ambiguous extra LOOK, and lifecycle-only policy tail. P10
tasks use R8 `PolicyEveryTen`: a policy call at every fresh ten-control anchor
and one five-control tail. No retrieval result, action, coin, or result extras
are rewritten. CU's nominal lottery parameter is retained from its fit;
`rho=.3` is not substituted as a Bernoulli probability. This is a thin fitted
controller switch, rather than setting C's `rho=0`, which would retain stall
calls/extra LOOKs and can fall below its calibrated feasible budget floor.

All three cells use **k=3 and `gap_all`**, following PROPOSALS P1 exact
confirmation step 2. `rules.json` contains a sweep over k=2/3/4/5 rather than
one uniquely mandated k per cell. The highlighted Spatial top-2 CU and pi05
L10 top-4 gripper P10 points are other operating points; no post-hoc k/signal
optimization was introduced. The emitted hard sets are:

| Cell | Descending frozen task ranking | Top-3 tasks |
|---|---|---|
| pi05_l10_50 | 0, 9, 7, 1, 3, 2, 4, 6, 5, 8 | 0, 9, 7 |
| groot_l10_50 | 7, 9, 0, 1, 3, 4, 6, 2, 8, 5 | 7, 9, 0 |
| pi05_spatial_50 | 6, 9, 0, 1, 5, 4, 8, 3, 7, 2 | 6, 9, 0 |

`task_tables.json` freezes full-precision scores, all ten rates for each
treatment, 30 discovery observations per task, deterministic task-ID tie
ordering, discovery init IDs, and SHA provenance for fable's proposal,
analysis, scoring code, rules, and shadow tables. The builder reads only
`task_id/init/gap_all` columns with `0 <= init < 30` parquet filters. It calls
fable's `task_scores` with a constant unused success sentinel because that
helper also computes `a_fail`; actual success labels never enter the ranking.
This confirmation freezes all 30 existing discovery inits per task. It does
not collect the separate ten-episode B-pool calibration suggested for future
deployment, and it does not access holdout outcomes for selection.

The A and CU rows preserve the original R8 method, kwargs, fit paths, plugin
flags, client overrides, and serving YAML bytes; arm/config names and manifest
change. Six A/CU fits are reused by SHA. Six task-switch fits embed clones of
the original A plus CU/P10 objects. Verification compares all twelve embedded
branch controller pickles byte for byte with their R8 originals, including
their fitted retrieval arrays, calibrated probability, stall model, and seeds.
The standard h100 asset planner successfully relocates metadata for all twelve
arms. Evidence commands and outputs are below.

## Files

All new code/artifacts are under this directory or the new run root.

- `methods.py`, `common.py`, `__init__.py`: serving switch, paths, bounded writes.
- `build.py`: freeze inputs, emit configs/rows, reuse fits, prefit wrappers, predict.
- `task_tables.json`, `source_lock.json`: frozen discovery allocation and R8 input SHAs.
- `arms_in.json`, `predictions.json`: emitter input and prior simulator predictions.
- `replay.py`: discovery-only post-projection logged-tape equivalence checks.
- `analyze.py`: collected standard-mode outcomes, paired contrasts, exact McNemar,
  Holm correction, and task-stratified paired bootstrap SR/pooled-IR intervals.
- `validate.py`, `audit.py`, `idempotence.py`, `test_task_budget.py`, `evidence/*`: verification.
- Run root: `arms.json`, twelve `config/r9p1_*.yaml`, twelve matrices,
  `manifests/discovery300.json` plus identical `discovery300.json`, six `fits/*.pkl`,
  `prefit_report.json`, and `local_asset_plan/*`. The local plan is a dry artifact;
  it is not `h100_sync/synced.json` and does not authorize a launch.

## Predicted SR / owner IR

These are **prior R8 dose-simulator estimates**, not R9 measurements. Frozen
columns simulate exactly the task table emitted here. CV columns are fable's
`gap_all`, top-3 CU/P10 rules in `rules.json`, which refit ranking on the other
two discovery folds. The frozen table uses the same discovery observations for
ranking and prediction, so its numbers are descriptive rather than independent
confirmation. A/CU use fable's discovery-only paired simulator directly.

Owner cost per decision is `.152*look + .848*call` for pi05 and
`.148*look + .852*call` for GR00T. IR pools cost and decision counts across
selected episodes; it does not average per-task or per-episode IR.

| Arm | Frozen SR | Frozen IR | Fable CV SR | Fable CV IR |
|---|---:|---:|---:|---:|
| r9p1_pi05_l10_50_A | .716667 | .076414 | — | — |
| r9p1_pi05_l10_50_CU | .880000 | .300696 | — | — |
| r9p1_pi05_l10_50_topk_cu | .833333 | .139233 | .833333 | .139233 |
| r9p1_pi05_l10_50_topk_p10 | .833333 | .194540 | .833333 | .194540 |
| r9p1_groot_l10_50_A | .636667 | .074303 | — | — |
| r9p1_groot_l10_50_CU | .796667 | .303407 | — | — |
| r9p1_groot_l10_50_topk_cu | .753333 | .146902 | .753333 | .146902 |
| r9p1_groot_l10_50_topk_p10 | .810000 | .200019 | .810000 | .200019 |
| r9p1_pi05_spatial_50_A | .800000 | .077452 | — | — |
| r9p1_pi05_spatial_50_CU | .926667 | .304420 | — | — |
| r9p1_pi05_spatial_50_topk_cu | .890000 | .143029 | .886667 | .144627 |
| r9p1_pi05_spatial_50_topk_p10 | .906667 | .193983 | .903333 | .200361 |

The exact proposed top-3 arms are predicted to improve over A and reduce cost
relative to CU, but **none of the six treatment predictions clears P1's stated
point-estimate bar**: Spatial SR ≥ CU at IR ≤ .6×CU, or L10 SR ≥ CU + .04 at IR
≤ .5×CU. GR00T top-3 P10 improves SR over CU by only .0133 at .6592×CU IR.
Other per-task predictions have lower SR than CU. Fable's positive reported
gains primarily compare against an **IR-matched uniform mixture**, a different
reference from the full CU arm. The implementation preserves the requested
confirmation design; it does not quietly substitute an easier acceptance bar.

## Coordinator commands — not executed here

Use the existing prepared timan107 h100 island. Do not refresh the entire
source snapshot or disturb existing chains. If the worker island is missing,
the RUNBOOK's scoped `control setup --worker-only` is a coordinator prerequisite;
its guard can refuse when that fleet is busy. Remote imports, checkpoints,
free disk, GPU/RAM capacity, and port availability were not checked here.

```bash
cd /home/weiland/projects/openpi
DIR=$PWD/exp/offline_search/rounds/r09/p1_task_budget
RUN=/home/weiland/trace_runs/os_closed_loop/r09_p1
T=exp.offline_search.rounds.r09.p1_task_budget
H=exp.offline_search.closed_loop.ops.h100.control
P=(taskset -c 22-29,66-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src "$PWD/.venv/bin/python")

# Local preparation is already complete. Idempotent verification/re-emission:
"${P[@]}" -m "$T.build"
mapfile -t ARMS < <("${P[@]}" -c 'import json,sys; print("\n".join(r["arm"] for r in json.load(open(sys.argv[1]))))' "$RUN/arms.json")
```

`sync` transfers fits and assets, not new serving Python modules. Before sync,
install only these two new module files in the isolated h100 tree. These are
new paths; push intentionally refuses overwrite. On a repeated preparation,
skip a push only after its existing remote SHA matches the listed local SHA;
if it differs, stop and resolve the code version. `mkdir` and SHA reads are
idempotent under the h100 duplicate-exec quirk. This does not use or modify
`/home/exouser/openpi`.

```bash
REMOTE=/data/oscl_h100/openpi/exp/offline_search/rounds/r09/p1_task_budget
tether exec h100 -- mkdir -p "$REMOTE"
tether push "$DIR/__init__.py" "h100:$REMOTE/__init__.py"
tether push "$DIR/methods.py" "h100:$REMOTE/methods.py"
tether exec h100 -- sha256sum "$REMOTE/__init__.py" "$REMOTE/methods.py"

# Plan is local. Sync and chain are coordinator actions.
WORKER_HOST=timan107 "${P[@]}" -m "$H" plan "$RUN" "${ARMS[@]}"
WORKER_HOST=timan107 SYNC_PORT=23195 \
  "${P[@]}" -m "$H" sync --concurrent "$RUN" "${ARMS[@]}"
WORKER_HOST=timan107 PORTS=23240,23241,23242,23243 WPS=12 SYNC_PORT=23195 \
  MAX_ATTEMPTS=3 POLL_SECONDS=60 OSCL_MANIFEST="$RUN/manifests/discovery300.json" \
  "${P[@]}" -m "$H" chain "$RUN" "${ARMS[@]}"
```

These use the same `control` implementation as `sync_assets.sh` and
`chain_h100.sh`, while keeping the local CPU affinity within S-P1's allocation.
Four ports × twelve workers = 48 workers on **timan107**, within that fleet's
64-worker guard. Do not apply these settings to timan108. The specified
manifest selects 300 pairs despite the driver's full 50-init pool attestation.
The existing h100 transfer guard must leave ≥30 GiB free and concurrent sync
must refuse conflicting existing assets. The local plan totals 20.581 GiB
before accounting for already installed assets; current remote capacity is
unverified. No port, process, listener, session, or running-chain cleanup is
part of these preparation commands.

Expected remote module SHAs (local verification only):

```text
__init__.py 28d89c26f26b0ec99325672499ee5462d999e5af38a3d0cd6887d1a90b5f228b
methods.py  1f83a19ad30149e233dd6bf347a36224695b6f8d7f75af56219b951b653537af
```

After the coordinator collects standard logs:

```bash
WORKER_HOST=timan107 "${P[@]}" -m "$H" collect "$RUN" "${ARMS[@]}"
"${P[@]}" -m "$T.analyze" --run-root "$RUN" --n-boot 2000
# Output: $RUN/paired_analysis.json, with per-task results and all 12 contrasts.
```

The analysis requires all 300 pairs for each arm, retains the final accepted
error-free done/failed journal row, matches the accepted server attempt,
deduplicates identical decision records, refuses conflicting/missing server
steps, and pairs by official `(task_id, init)`. It reports treatment-vs-CU and
treatment-vs-A SR changes, exact two-sided McNemar wins/losses and p-values,
Holm p-values over twelve contrasts, task-stratified paired bootstrap SR/IR
changes and IR ratios, and separate point-bar feasibility. Actual R9 analysis
is unverified because no R9 episodes were run. Its standard-log reader was
tested with synthetic accepted/retried journals and server logs; the same
statistical path was exercised on the prior R8 simulator.

## Verification commands and observed outputs

Use the `P`, `DIR`, `RUN`, and `T` definitions above. Complete captured outputs
and structured reports are in `evidence/`; summaries below are verbatim except
that the long per-arm/per-task lines are linked rather than repeated.

```bash
"${P[@]}" -m "$T.build" --rebuild-wrappers > "$DIR/evidence/build.txt" 2>&1
```

```text
P1_BUILD_OK arms=12 pairs=300 reused=6 wrappers=6 capture=off
TASKS pi05_l10_50 order=[0, 9, 7, 1, 3, 2, 4, 6, 5, 8] hard=[0, 9, 7]
TASKS groot_l10_50 order=[7, 9, 0, 1, 3, 4, 6, 2, 8, 5] hard=[7, 9, 0]
TASKS pi05_spatial_50 order=[6, 9, 0, 1, 5, 4, 8, 3, 7, 2] hard=[6, 9, 0]
```

[Full build output, twelve predictions and fit SHAs](evidence/build.txt).
[Frozen ranking/provenance](task_tables.json).
[Per-arm prefit hashes](/home/weiland/trace_runs/os_closed_loop/r09_p1/prefit_report.json).

```bash
"${P[@]}" -m pytest "$DIR/test_task_budget.py" -q -p no:cacheprovider \
  --basetemp="$RUN/pytest_tmp" > "$DIR/evidence/pytest.txt" 2>&1
```

```text
......................                                                   [100%]
22 passed in 3.44s
```

Tests cover invalid tables, raw-key A equivalence for both models, task changes
and implicit reset, P10 policy tails, connection state isolation, tampered fit
SHAs, exact frozen packaging/manifest, accepted retries and owner pricing,
paired simulator reproduction, completeness/duplicate refusal, and Holm math.

```bash
"${P[@]}" -m "$T.replay" > "$DIR/evidence/replay.txt" 2>&1
```

```text
P1_REPLAY_OK checks=9 inits=30 branch_exact=88508 report=/home/weiland/projects/openpi/exp/offline_search/rounds/r09/p1_task_budget/evidence/replay.json
```

[Nine replay check counts](evidence/replay.txt) and [structured scope/limits](evidence/replay.json).
The selected zero-rate tasks replay 210 A episodes per cell, through both task
switches; the selected high-rate tasks replay 90 CU and 90 P10 episodes per
cell. This totals 1,170 branch episodes and 57,170 logged decisions, with 88,508
switch/reference comparisons. All 47,435 non-call chunks checked were exact
over the full valid-action horizon (pi05 10, GR00T 16 controls × seven dims),
maximum absolute error 0. LOOK/call decisions and all returned branch fields,
full action-array bytes, and extras matched exactly.

Replay limitation: raw 32,768-D keys were captured only on a subsample. Full
logged replay starts from the exact saved post-projection PCA keys and
normalized state, replacing projection only in detached replay objects.
Raw-key synthetic-library unit tests separately exercise the original
projection path. Policy calls replay historical full chunks as history; no
fresh policy forward, observation extraction, wire transform, GPU backend, or
closed-loop outcome equivalence was measured. No original fit was mutated.

```bash
"${P[@]}" -m "$T.validate" > "$DIR/evidence/validate.txt" 2>&1
```

```text
P1_VALIDATE_OK arms=12 pairs=300 source_SHA=9 exact_wrapper_branches=12 relocated_metadata=12 plan_files=105 plan_GiB=20.581 remote_actions=0
```

[Local asset-plan output](evidence/validate.txt), [branch/SHA verification](evidence/validation.json),
and [relocated plan](/home/weiland/trace_runs/os_closed_loop/r09_p1/local_asset_plan/plan.json).

```bash
"${P[@]}" -m "$T.analyze" --simulate-r8 > "$DIR/evidence/paired_simulator.txt" 2>&1
```

```text
P1_PAIRED_OK kind=prior R8 fixed-table dose simulation (discovery reused for ranking) arms=12 pairs_per_arm=300 report=/home/weiland/projects/openpi/exp/offline_search/rounds/r09/p1_task_budget/evidence/paired_simulator.json
```

[Twelve simulated paired contrasts, CIs and p-values](evidence/paired_simulator.txt).
The simulator file is explicitly labelled prior R8 simulation and is not stored
as the measured R9 `paired_analysis.json`.

```bash
"${P[@]}" -m "$T.idempotence" > "$DIR/evidence/idempotence.txt" 2>&1
"${P[@]}" -m "$T.audit" > "$DIR/evidence/final_audit.txt" 2>&1
```

```text
P1_IDEMPOTENCE_OK unchanged_arms=1 unchanged_tables=1 unchanged_source_lock=1 unchanged_wrapper_fits=6
```

[Recorded before/after check](evidence/idempotence.txt).
The final audit prints inventory, replay totals, zero simulator bar passes,
serving-file SHAs, and the test result; [full output](evidence/final_audit.txt)
and [inventory hashes](evidence/final_audit.json) are retained.

## Open risks

Remote installation/imports, available ports/disk/GPU/RAM, actual fleet load,
cross-hardware numerics, policy RNG effects across arm names/processes,
throughput, and actual R9 SR/IR remain unverified. The new fits add two frozen
controllers per task switch; CPU memory under four server replicas has not
been measured remotely. The current predictions do not support declaring the
strict P1 success bar reached. Inits 30–49 remain a later coordinator holdout.
