Six frozen arms are ready for coordinator staging: two byte-identical round6 per-task controls and two task-id-free candidates in each of the two GR00T cells. The standard emitter produced the run root; the standard controller plan completed. Nothing has been synced, launched, killed, restarted or otherwise changed remotely.

Run root: `/home/weiland/trace_runs/os_closed_loop/r09_astra_r7/`.
Package: `exp/offline_search/rounds/r09/explore_astra/round7/`.
Manifest: `manifests/eval100.json`, exactly tasks 0–9 × inits 20–29, 100 pairs/arm, 600 episodes total. No pi0.5 arms.

## Frozen arm order and local checks

From the repository root:

```bash
cd /home/weiland/projects/openpi
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
R7=/home/weiland/trace_runs/os_closed_loop/r09_astra_r7
PKG=exp.offline_search.rounds.r09.explore_astra.round7.tools
CONTROL=exp.offline_search.closed_loop.ops.h100.control
ARMS=(
  r9a7_groot_l10_50_control
  r9a7_groot_l10_50_capacity
  r9a7_groot_l10_50_confidence
  r9a7_groot_spatial_50_control
  r9a7_groot_spatial_50_capacity
  r9a7_groot_spatial_50_confidence
)
taskset -c 10-21,54-65 .venv/bin/python -m "$PKG.verify"
taskset -c 10-21,54-65 .venv/bin/python -m "$PKG.test_tools"
taskset -c 10-21,54-65 .venv/bin/python -m "$PKG.selftest"
WORKER_HOST=timan108 taskset -c 10-21,54-65 .venv/bin/python -m "$CONTROL" plan "$R7" "${ARMS[@]}"
```

Already completed: 15 unit tests, 12 real CPU plugin tests, standard plan (44 files, 7,375,204,959 bytes). `selftest` reuses existing successful reports. Its initial execution ran only local CPU replay processes, with identity-filtered training observations and the real plugin/orchestrator; no server or evaluation worker was launched. Forced-call fixtures are separate test processes, absent from production artifacts. Two reset episodes and two connections per test, 48 decisions, policy-miss and tail paths exercised.

The CPU identity reader is test-local. It never replaces the standard serving store and is absent from the serving source closure. Production plugin args point to `/home/weiland/trace_runs/offline_search_store`. The standard plan includes its normal query episode identity maps as deployment dependencies; it does not decode outcome trajectories. No alternate controller or lock adapter is used.

`verify` checks local frozen files and source hashes without invoking the controller. It asserts this run root has no `state/`, `runs/`, or `serving_store/`. Run its prelaunch audit before the coordinator's sync/chain; that no-launch assertion is intentionally unsuitable after launching.

## Serving source list and SHA-256

Standard sync relocates fit artifacts and method dependency paths; it does not install new Python modules. Four new files must be staged in the h100 code tree before unpickling:

- `exp/offline_search/rounds/r09/explore_astra/round7/__init__.py`
- `exp/offline_search/rounds/r09/explore_astra/round7/tools/__init__.py`
- `exp/offline_search/rounds/r09/explore_astra/round7/tools/methods.py`
- `exp/offline_search/rounds/r09/explore_astra/round7/tools/numeric.py`

`H100_NEW_SOURCES.sha256` hashes those four. `H100_SOURCES.json` maps all 69 experiment source dependencies to local/remote paths; `H100_SOURCES.sha256` hashes their full static `exp.offline_search` import closure, including round6 NumPy serving math and inherited stack/retrieval code. The existing standard OpenPI environment is assumed. Analysis, training and test modules are not required by serving. Local hashes were checked; no h100 source or fleet state was inspected.

The following are **coordinator-only commands, not executed here**. They preserve the round6 source-staging discipline: refuse differing existing files, copy new paths, and verify the inherited closure. Reconcile a differing inherited dependency before any launch.

```bash
set -euo pipefail
R7SRC=exp/offline_search/rounds/r09/explore_astra/round7
taskset -c 10-21,54-65 sha256sum -c "$R7SRC/H100_SOURCES.sha256"
while read -r digest rel; do
  target="/data/oscl_h100/openpi/$rel"
  tether exec h100 -- bash -c '
    if [ -e "$1" ]; then
      actual=$(sha256sum "$1")
      test "${actual%% *}" = "$2"
    else
      mkdir -p "${1%/*}"
    fi
  ' _ "$target" "$digest"
  tether push --force "$rel" "h100:$target"
  tether exec h100 -- bash -c '
    actual=$(sha256sum "$1")
    test "${actual%% *}" = "$2"
  ' _ "$target" "$digest"
done < "$R7SRC/H100_NEW_SOURCES.sha256"
tether exec h100 -- mkdir -p /data/oscl_h100/runs/r09_astra_r7/source_check
tether push --force "$R7SRC/H100_SOURCES.sha256" h100:/data/oscl_h100/runs/r09_astra_r7/source_check/H100_SOURCES.sha256
tether exec h100 -- bash -c 'cd /data/oscl_h100/openpi && sha256sum -c /data/oscl_h100/runs/r09_astra_r7/source_check/H100_SOURCES.sha256'
```

## Standard sync and chain for the coordinator

Not executed. Use the coordinator's existing fleet discipline and assigned worker; timan108 is the explicit plan default, matching the preceding round's stated evaluation fleet. Use the same worker selection for sync and chain, and retain all six arms in one batch.

```bash
export WORKER_HOST=timan108
export OSCL_MANIFEST="$R7/manifests/eval100.json"
taskset -c 10-21,54-65 .venv/bin/python -m "$CONTROL" sync --concurrent "$R7" "${ARMS[@]}"
taskset -c 10-21,54-65 .venv/bin/python -m "$CONTROL" chain "$R7" "${ARMS[@]}"
```

These directly invoke `exp.offline_search.closed_loop.ops.h100.control`; no round7 wrapper owns the chain lock. All arms preserve `guard_only`, full-model availability, cost ledger, resize 256, five-control decisions with one policy-tail block/ten-control commitment, no forced grasp calls, no added escalation, no shadow-native/debug/oracle flags. Only the corrector differs. Both candidate/control comparisons use their new same-batch per-task control, never a historical control from another batch.

## Files to preserve and scoring

Preserve the whole `round7/` package and run root `arms.json`, `config/`, `fits/`, `manifests/eval100.json`, `h100_sync/plan.json` and local `selftest/` evidence. Four numerical NPZs are explicit residual-path dependencies discovered by the standard plan. `FROZEN.json` records exact r6 control source hashes, output hashes and prediction hash. `INTEGRITY.json` seals the local handoff. Do not regenerate `SELECTION.json`, predictions, fits or arms after launch.

Predictions were frozen at **2026-10-02 11:41:27 UTC**, before emission. See `PREDICTION.md` for four point forecasts, uncertainty, evaluation rules and the predeclared SR/IR screen. Score success, paired +/− counts, init-cluster intervals, owner IR, looks, calls and decision counts. Continue applying identity admission before deserializing outcome records. No init 30–49 or forbidden root is needed.

Reproduction sequence, already completed: `tools.analyze` → `tools.learn select` (0–19 only) → `tools.learn evaluate` (20–29) → timestamped prediction → `tools.build` (standard emitter) → CPU checks and standard plan → source inventory → integrity seal. `build` refuses an already emitted run. Reports and candidate diagnostics are generated by `tools.write_reports` and `tools.diagnostics`; neither refits models or changes predictions.
