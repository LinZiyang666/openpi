# R10 astra handback

Status: complete, frozen blind to all closed-loop results; local CPU preparation only.
Start with [REPORT.md](REPORT.md), then [PREDICTION.md](PREDICTION.md).

## Frozen deliverable

24 GC_dist arms: 4 cells × 50/100/200/300/400/500 episodes, split 12 per root:

- `/home/weiland/trace_runs/os_closed_loop/r10_corr3_pi05`
- `/home/weiland/trace_runs/os_closed_loop/r10_corr3_groot`

Arm names: `r10_<model>_<suite>_<size>_GC_dist`.
Method: `exp.offline_search.rounds.r10.astra.method:DistanceController`.
Exact existing G and Stage 2b LOEO heads; distance attenuation only, no augmentation.
Strength `.5*clip((2-r)/1.25,0,1)`; step 0 disabled; one scalar calibration per cell-size.
All serving parameters and prediction frozen at **2026-10-02T18:22:47.770823+00:00**.
`freeze.json` binds protocol, prediction, selection, comparison, 24 calibrations,
and the method/offline sources. Both `emission_receipt.json` timestamps are later.

Official A manifest is exactly 500 task/init pairs, byte-identical to corr2.
Existing R10 arms/fits/heads/source files were imported/read only, not edited.
No closed-loop client/journal/summary/server log or result was read, including R10.
No A/B-val recording was used. No sync, chain, server, worker, remote launch or git command ran.

## Evidence

- `PROTOCOL.md`: candidates and selection criterion stated before astra CV.
- `OFFLINE.md`, `offline_cv.csv`, `distance_bins.csv`, `offline_comparison.json`:
  full comparison, by-distance diagnostics and conditional uncertainty.
- `cv/*/folds.json`: all training/held-out episode identities and training-only scales.
- `cv/*/rows.npz`: row-level held-out sufficient statistics for every candidate.
- `selection.json`, `calibration/*.json`: single global selected gate, per-cell-size scalar.
- `fit_audit.json`: unchanged G/LOEO attributes and artifact SHA256 for all 24 arms.
- `selftests.json`: all 24 standard real-plugin CPU selftests pass, 1,152 decisions.
- `serving_gate_audit.json`, `relocation_audit.json`, `final_audit.json`: all pass.
- `input_hashes.json`: 134 protected inputs unchanged.
- `deployment.json`: extra store=0; new run/calibration=1,030,759,546 bytes;
  full dependency union=25,414,418,693 bytes. New serving sources=3,740 bytes.
- `H100_SOURCES.sha256`: the two new serving files; `ALL_SOURCES.sha256`: every helper.
- New-root `h100_sync/plan.json`: standard, complete bounded dependency plan.

Main offline result: GC_dist beats LOEO in 19/24 panels. Equal-panel MSE change vs
no correction is −4.096% vs LOEO −3.669%. All 12 GR00T panels improve; pi05 500
loses roughly 0.85%/1.01% relative to LOEO. This is tuning CV with a fixed PCA basis,
not an independent test and not a success-rate estimate. New-episode metric calibration
uses smaller donor folds without extrapolation; see the limitations in OFFLINE.md.

## Local checks only

All Python commands use `.venv/bin/python`, `PYTHONPATH=.:src`, CPUs `10-21,54-65`,
and OMP/OPENBLAS/MKL thread counts 1. Bytecode writes are disabled. The original
build was `astra.build build --workers 4`; new roots must not exist for that command.
Do not overwrite the frozen selection/calibrations/prediction in place.

Revalidate existing prepared artifacts, with no launch:

```sh
taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -m exp.offline_search.rounds.r10.astra.tests
taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -m exp.offline_search.rounds.r10.astra.build validate --workers 4
taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -m exp.offline_search.rounds.r10.astra.audit
```

`run_tool.py plan <root> <12 explicit arm names>` invokes the unmodified standard
`closed_loop.ops.h100.control plan` under the file-access boundary. Both were run.
`run_tool.py selftest` imports sol's unmodified test; its synthetic logs stay in astra.
No remaining fitting, emission or local validation work is pending. Closed-loop
evaluation and any transfer remain the owner's separate task.
