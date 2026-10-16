# Fresh ffk execution worker counts — 2026-10-16

`filter_failed.resume --run ... --worker-counts ...` now accepts a fresh,
unstaged `ffk1015_*` draft with `OFFLINE_READY.json` and the existing frozen-fit
row-filter method. It uses `worker_topology.redistribute`, exactly as a
continuation does. It refuses `PREPARED.json`, `LAUNCH.json`, or `RECEIPT.json`.
No sealed check was changed. No real campaign was modified, prepared, or launched
during these checks.

The native knob records `execution_worker_nodes`, `execution_workers`,
`cache_worker_orchestration`, and `WORKER_REDISTRIBUTION.json`. Canonical
`workers=120` and `worker_nodes` remain unchanged for the batch-treatment seal.
Native receipts expose the execution map; knob analysis now also includes the
native `lane_execution_segments` resource records in each result cell.

| Campaign | Server | R / max B | Proposed execution workers | Planned episodes | CPU startup identity | Equals source actual identity | Frozen planner |
|---|---|---|---|---:|---|---|---|
| ffk1015_dp_mt50_reduced_descent | H100 | 4 / 16 | timan107 × 40 + timan108 × 40 | 90,000 | PASS | yes | all 90,000 task records identical; capacity 80 |
| ffk1015_fastwam_mt50_reduced_descent | H100 | 2 / 16 | timan107 × 40 + timan108 × 40 | 90,000 | PASS | yes | all 90,000 task records identical; capacity 80 |

The CPU checks used private proposal copies in this directory, the frozen
runtime, and each job's sealed-version server interpreter. Both package-version
checks passed. Fits, selected rows, runtime, R/B, task UIDs, seeds, identities,
and task ordering are unchanged. See `PROOF.json` and each campaign's
`STARTUP.json`, `PLANNER.json`, and `ANALYSIS_RESOURCES.json`. The planner only
memoized immutable benchmark identity properties, as the existing CPU build
check does. Seven focused tests passed (`TESTS.log`).

MT50 does use worker GPU rendering: `exp/metaworld/env.py` supplies EGL-rendered
480 × 480 camera images to the policy; the frozen worker probe requires NVIDIA
EGL and records GL renderer/vendor/version. It does not certify pixel equality
across GPU models. The archived environment/renderer/probe files match the
checkout byte-for-byte (`RENDERER_SOURCE.json`). The proposed execution keeps
the existing source timan107/timan108 device settings (timan107 devices 0–7;
timan108 device 3) and drops timan1. Renderer settings and per-worker VRAM
admission are unchanged. Reduced concurrency can change model batch composition;
the native orchestration record explicitly reports this.

## Coordinator commands

Apply both overrides **before preparing**. These commands have not been run on
the real campaigns. They do not rebuild either campaign.

```bash
cd /home/weiland/projects/openpi
CPU=(env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python)
"${CPU[@]}" -m exp.offline_search.rounds.r17.filter_failed.resume --run /data/mw_finetune/eval/ffk1015_dp_mt50_reduced_descent --worker-counts timan107=40,timan108=40
"${CPU[@]}" -m exp.offline_search.rounds.r17.filter_failed.resume --run /data/mw_finetune/eval/ffk1015_fastwam_mt50_reduced_descent --worker-counts timan107=40,timan108=40
```

Use the existing knob prepare/launch scripts, which preserve the frozen ffk
bootstrap and current worker-command wrapper. Run the lanes serially because
they share the H100 and workers. The execution map makes preparation and launch
use only timan107/timan108; canonical timan1 metadata is retained for the seal.

```bash
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_dp_mt50_reduced_descent.sh
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_dp_mt50_reduced_descent.sh
```

After DP closes:

```bash
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_fastwam_mt50_reduced_descent.sh
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_fastwam_mt50_reduced_descent.sh
```

Focused tests:

```bash
"${CPU[@]}" -m pytest -q exp/offline_search/rounds/r17/filter_failed/test_resume.py
```
