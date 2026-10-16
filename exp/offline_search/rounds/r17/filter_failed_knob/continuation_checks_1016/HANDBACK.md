# DP MT50 reduced knob continuation — 2026-10-16

Built locally: `/data/mw_finetune/eval/ffk1015_dp_mt50_reduced_descentc1`.
It retains 1,059 accepted, error-free outcomes from
`ffk1015_dp_mt50_reduced_descent` and schedules the remaining **88,941** episodes.
Server: H100, R4 / max B16. Execution workers: **timan107 × 40 + timan108 × 40**.
The continuation has not been staged or launched.

The existing `filter_failed.resume` path now accepts ffk continuations and
`--verify-closed`. That option calls the existing native
`library_salvage.prove_closed` read-only ownership probe. It does not rewrite the
failed source receipt. Here it proved the local driver, 12 H100 process
identities, two timan107 identities and two timan108 identities dead, also
checking process groups and campaign worker argv. See `CLOSURE.json` and the
continuation's `SOURCE_CLOSURE.json`. The latter is embedded in the continuation
selection and binds the original campaign/receipt hashes. The source receipt
still records `FAIL`, `all_owned_closed=false`.

The continuation is registered in `filter_failed/CONTINUATIONS.json`. Knob
workflow and analysis now resolve registered continuation tips, then use the
existing native cache loader to read all ancestor segments. The original
`CAMPAIGNS.json` entry and ordinary descent/result paths remain valid. No source
campaign, study, fit, plan, or launched run was rebuilt, reset or re-staged.

CPU results:

| Check | Result |
|---|---|
| Actual retained terminal outcomes | 1,059 unique; zero errors |
| Missing planned episodes | 88,941 |
| Full unique size/level/task/init identities | 90,000 |
| Retained/missing overlap, by UID and by coordinate | 0 |
| Original full plan / missing task records | identical |
| Frozen driver planner | PASS; all 88,941 records identical; capacity 80 |
| Frozen CPU startup identity and sealed interpreter packages | PASS; equals source actual serving identity |
| Serving assets | all 96 files byte-identical |
| Runtime and worker archives; driver pin; R/B; selected rows; seeds and identities | unchanged |
| Native parser with private synthetic completion fixture | 90,000 unique outcomes, despite 1,059 copied ancestor rows; original execution origins preserved |
| Focused continuation and workflow tests | 11 passed |

The synthetic missing outcomes exist only in `synthetic_merge/`, a CPU test
fixture. Actual accepted outcomes remain 1,059 until the coordinator launches
the continuation. `COVERAGE.json`, `PLANNER.json`, `ANALYSIS_RESOURCES.json`,
`EXECUTION_SEGMENTS.json` and `TESTS.log` contain the checks. Both execution
segments appear in each of the six analysis cells with their actual 80-worker
topology.

`STARTUP.json` records the native startup identity probe on the continuation's
actual campaign and descriptors. The probe body was transported with xz
compression to fit the read-only exec payload limit with its 90,000-UID resume
selection; no probe condition or campaign field was changed.

## Coordinator commands

The build has already completed. Prepare and launch only the new continuation:

```bash
cd /home/weiland/projects/openpi
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_ffk1015_dp_mt50_reduced_descentc1.sh
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_ffk1015_dp_mt50_reduced_descentc1.sh
```

After it completes, the ordinary descent command automatically reads both
segments as one lane:

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.workflow descent --model dp --benchmark mt50 --mode reduced
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.analyze
```

Build provenance (already run; do not repeat with the same name):

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.resume --source /data/mw_finetune/eval/ffk1015_dp_mt50_reduced_descent --name ffk1015_dp_mt50_reduced_descentc1 --worker-counts timan107=40,timan108=40 --verify-closed
```
