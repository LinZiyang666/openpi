Eight frozen arms are ready for coordinator staging: four copied current-corrector controls and four task-id-free replacements. All use `/home/weiland/trace_runs/offline_search_store` and the standard emitter/control modules. No custom control adapter or filtered serving store exists. Nothing has been synced or launched.

## Local verification already completed

From the repository root:

```bash
cd /home/weiland/projects/openpi
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
R6=/home/weiland/trace_runs/os_closed_loop/r09_astra_r6
PKG=exp.offline_search.rounds.r09.explore_astra.round6.tools
CONTROL=exp.offline_search.closed_loop.ops.h100.control
ARMS=(
  r9a6_pi05_l10_50_control r9a6_pi05_l10_50_taskfree
  r9a6_pi05_spatial_50_control r9a6_pi05_spatial_50_taskfree
  r9a6_groot_l10_50_control r9a6_groot_l10_50_taskfree
  r9a6_groot_spatial_50_control r9a6_groot_spatial_50_taskfree
)
taskset -c 10-21,54-65 .venv/bin/python -m "$PKG.verify"
taskset -c 10-21,54-65 .venv/bin/python -m "$PKG.test_tools"
taskset -c 10-21,54-65 .venv/bin/python -m "$PKG.selftest"
WORKER_HOST=timan107 taskset -c 10-21,54-65 .venv/bin/python -m "$CONTROL" plan "$R6" "${ARMS[@]}"
```

`selftest` reuses completed successful evidence; it does not relaunch a server. Its child processes are CPU-only replay tests and inherit the affinity/thread settings. The real standard plugin/orchestrator is exercised, with local identity-first input readers. Those readers are not installed into serving. Unit tests: 10/10. Production/forced plugin tests: 16/16. Standard plan: exit 0, 102 files, 23,870,705,656 bytes. `verify` is a local integrity checker; it never calls the controller.

## Coordinator: stage the new serving sources

Standard asset sync relocates numerical fits and paths; it does **not** install new Python modules into the h100 code tree. Four new serving files must be present there before unpickling the new fits. `H100_NEW_SOURCES.sha256` lists exactly those four files; `H100_SOURCES.sha256` lists the 66-file experiment import closure, including inherited stack/retrieval dependencies. No local analysis/test module is required by serving.

The following are coordinator-only source-copy commands, **not executed for this package**. They copy only the four new round6 paths, refuse an existing differing file, and check the full source closure. They do not start any server, worker or controller, and take no run-root lock. Use the coordinator's existing fleet scheduling discipline before remote writes.

```bash
set -euo pipefail
R6SRC=exp/offline_search/rounds/r09/explore_astra/round6
sha256sum -c "$R6SRC/H100_SOURCES.sha256"
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
done < "$R6SRC/H100_NEW_SOURCES.sha256"
tether exec h100 -- mkdir -p /data/oscl_h100/runs/r09_astra_r6/source_check
tether push --force "$R6SRC/H100_SOURCES.sha256" h100:/data/oscl_h100/runs/r09_astra_r6/source_check/H100_SOURCES.sha256
tether exec h100 -- bash -c 'cd /data/oscl_h100/openpi && sha256sum -c /data/oscl_h100/runs/r09_astra_r6/source_check/H100_SOURCES.sha256'
```

An absent/different inherited dependency must be reconciled by the coordinator before launch; these commands deliberately do not overwrite inherited source files. Remote source hashes were not inspected during this work. No custom module should own a chain lock.

## Coordinator: exact standard sync and chain

Continue in the same shell with the variables above. The commands use timan107 explicitly; the coordinator may select its assigned idle fleet consistently for both sync and chain. All arms must retain the same fleet and manifest for the paired screen. The explicit manifest prevents a stale environment override from widening the init selection.

```bash
export WORKER_HOST=timan107
export OSCL_MANIFEST="$R6/manifests/eval100.json"
taskset -c 10-21,54-65 .venv/bin/python -m "$CONTROL" sync --concurrent "$R6" "${ARMS[@]}"
taskset -c 10-21,54-65 .venv/bin/python -m "$CONTROL" chain "$R6" "${ARMS[@]}"
```

These invoke **`exp.offline_search.closed_loop.ops.h100.control` itself** as the main module. No round6 process wraps `chain`, no alternate store argument is supplied, and no lock authentication workaround is used. Sync/chain commands above were not executed locally.

Each arm has the identical explicit manifest: task IDs 0–9 × init IDs 20–29 (100 pairs). Total: 800 episodes. All arms have full-model availability, `guard_only`, ten-control cache/policy commitment through five-control decisions and one policy-tail block, no shadow-native/debug/oracle flags, no added recovery trigger, and `cost_ledger=true`. GR00T retains `resize_size=256`.

| Cell | Copied control | Replacement | Escalation |
|---|---|---|---|
| π0.5 LIBERO-10-50 | `r9a6_pi05_l10_50_control` | `r9a6_pi05_l10_50_taskfree` | Existing lag 12 / deadline 80 in both |
| π0.5 Spatial-50 | `r9a6_pi05_spatial_50_control` | `r9a6_pi05_spatial_50_taskfree` | None in either |
| GR00T LIBERO-10-50 | `r9a6_groot_l10_50_control` | `r9a6_groot_l10_50_taskfree` | None in either |
| GR00T Spatial-50 | `r9a6_groot_spatial_50_control` | `r9a6_groot_spatial_50_taskfree` | None in either |

Prediction was frozen before emission. Score all four paired cells using `PREDICTION.md`; report success, discordant pairs, owner IR, episode count, decision count and paired init-cluster uncertainty. Do not substitute an earlier-batch control, drop tasks, or broaden the manifest. Any later outcome analysis must apply the same identity-first admission before decoding records; never open the prohibited roots or init 30–49 trajectories.

## Reproduction code and frozen outputs

The completed fitting sequence was `tools.learn select`, then `tools.learn evaluate`, then the timestamped prediction, then `tools.build` using the standard emitter. The recipe is fixed by `SELECTION.json`; exact control provenance and hashes are in `FROZEN.json`. `build` refuses a second emission into an existing run root. Existing artifacts should be verified, not silently regenerated after launch.

Files requiring handoff: this entire `round6/` package plus the run root's `arms.json`, `config/`, `fits/`, `manifests/eval100.json` and standard `h100_sync/plan.json`. Standard sync resolves all external frozen fit/library dependencies from `arms.json`. The plan includes the standard store's identity metadata and payload tables, not a run-local subset. Four new numerical artifacts are also listed as explicit method dependencies.

Do not run round5's control adapter or reuse its `serving_store`. The round6 local CPU reader exists only in `tools/selftest.py`, which is absent from the serving source closure.
