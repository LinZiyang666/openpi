# Reproduce R5-A diagnostics

Run from `/home/weiland/projects/openpi`. These scripts read the offline store and existing closed-loop logs. They write only to this ideation directory. They do not fit policy models, load a GPU, launch LIBERO, use a remote host, or change serving code. No saved array exceeds 50 MB. The library fits here are small CPU ridge regressions; the existing AWM artifacts are inventoried by file size, not refitted.

```bash
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r05/ideation_A/inventory.py
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r05/ideation_A/analyze_logs.py
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r05/ideation_A/analyze_chunks.py
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r05/ideation_A/analyze_cadence.py
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r05/ideation_A/make_review_tables.py
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r05/ideation_A/make_arm_specs.py
taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r05/ideation_A/validate_outputs.py
```

Run sequentially. `analyze_logs.py` uses four worker processes; the remaining scripts are single-process with single-threaded BLAS. A later inventory may discover more completed R4 arms: preserve the delivered outputs when comparing snapshots. The report is tied to the delivered results, not to any future re-run.

Validation checks accepted-attempt counts, paired denominators and discordances, full-miss cost arithmetic, valid library edges, reconstruction of all 16-member anchor heads/tails, calibration dimensions, library separation, and review-recipe uniqueness. It does not test new controller code: those classes/flags are design specifications only. Server patches, method fitting, transport tests, and all closed-loop pilots belong to the coordinator.

`log_summary.json` includes some R3 historical arms whose HIT actions were not logged; their action statistics are deliberately incomplete and are not used for the report's pure-inference jitter comparisons. `projections` fields freeze subsequent logged decisions and can leave invalid inherited blind continuations; `cadence_audit.json` counts these orphan slots. These masks must not be read as closed-loop outcomes. The corrected accounting sensitivity is in `review_tables.json`.

Sources read beyond the data: R5 `FINDINGS.md`; R4 `SELECTION.md`, `CODING_BRIEF.md`, available k1…k7 hand-backs, k8/k9 reports, k10 implementation/preparation code, ideation A/B/C reports including C's second report; R3/R2 `ANALYSIS.md`; exploration log §§8–10 (R4); `IDEATION_BRIEF.md`; harness and closed-loop READMEs; AWM, AWM3, G3, MixedJudge, K1/K7/K10, control-step library, GPU-retrieval design, plugin/blind API, KPI/chain/remote driver; GR00T LIBERO serving/adapter settings; and step-diagnostic §6.15. No `tests/review_tests/` content was read.
