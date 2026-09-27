# Reproduce R4-A diagnostics

Run from `/home/weiland/projects/openpi`. The hot copy was absent during this analysis; scripts use the cold copy read-only. No fitted artifacts are changed. Python bytecode writes are disabled. All data outputs are individually below 50 MB; there are no newly written arrays above 50 MB.

Execute these commands sequentially (each pool has eight workers plus its parent):

```bash
taskset -c 0-11,44-55 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r04/ideation_A/measure_blind.py anchors --workers 8
taskset -c 0-11,44-55 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r04/ideation_A/measure_blind.py analyze --workers 8
taskset -c 0-11,44-55 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r04/ideation_A/measure_logs.py
taskset -c 0-11,44-55 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r04/ideation_A/audit_logs.py
taskset -c 0-11,44-55 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r04/ideation_A/schedules.py
taskset -c 0-11,44-55 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r04/ideation_A/build_tables.py
taskset -c 0-11,44-55 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r04/ideation_A/validate_outputs.py
taskset -c 0-11,44-55 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r04/ideation_A/assemble_report.py
```

`anchors` reuses existing outputs. Delete only named generated anchor files if deliberately regenerating their baseline; no deletion is needed for the other scripts. During the original run at most two pools ran simultaneously (18 processes), never exceeding the assigned 24 logical CPUs. The scripts are diagnostics only, not implementations of a deployable Method, server, plugin, or LIBERO worker.

Artifacts:

- `anchors_*.json`: scalar-API parity checks, query counts, source fit artifact and its byte size.
- `metrics_*.csv`: all cells, both scales, horizons 1–4, all serving alternatives, near/far transitions, thirds and their intersections; error mean/median/p90, gripper mismatch, AWM action drift, Δerror and tail incidence.
- `triggers_*.csv`, `residuals_*.csv`: independently firing triggers and residual quantiles.
- `windows_*.npz`: compact paired horizon errors and trigger masks, no raw images/keys.
- `log_summary_*.json`, `log_timing_*.csv`: accepted-attempt closed-loop anatomy and first-spell trigger alignment.
- `command_phase_*.csv`, `command_phase_summary.json`: gripper-command phase at the first failed-episode spell and effective episode count in the logged kernel.
- `offline_schedules_*.csv`: frozen-observation/frozen-anchor-history schedule replay, gates and budgets.
- `log_schedule_aggregate.csv`: mixed and pure-cache IR estimates keeping recorded MISS timing unchanged.
- `paired_error_ci.csv`: whole-episode bootstrap action-error comparisons.
- `bytes.json`: actual R2 artifact sizes and proposed compact payload accounting.
- `validation.json`: complete-cell/phase/horizon coverage, weight and finite-value checks, same-library fit provenance, accepted log counts, IR identities, and array file-size checks.
- `MEASUREMENTS.md`: generated measurement tables. `REPORT.md` contains the interpretation and implementation specification; `REPORT_body.md` is its authored body and `assemble_report.py` appends the self-contained numerical appendix.

On the inf recordings, the AWM anchor and the recorded-history vision comparator see the recorded previous policy chunk. `awm_allhit` removes that privileged fresh branch at future comparison points. A real hypothetical blind rollout would change both state and history; these scripts do not simulate that rollout. Closed-loop logs retain only top 10/16 members; offline replay retains all 16.
