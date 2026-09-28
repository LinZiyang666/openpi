**Reproduction**

Run from `/home/weiland/projects/openpi`. All inputs are read-only; all outputs go beside these scripts. No GPU or simulator is used. Scripts create no single output array larger than 50 MB. Existing R4-A replay arrays and deployed fit pickles are read, not refitted or modified.

Every Python invocation must use this complete prefix:

```bash
taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r05/ideation_C/audit_logs.py
taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r05/ideation_C/library_study.py --workers 2
taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r05/ideation_C/memory_yield.py
taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r05/ideation_C/guard_memory.py
taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r05/ideation_C/capacity_growth.py
taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r05/ideation_C/report_tables.py
```

Execute sequentially for the simplest resource accounting. `audit_logs.py` uses four workers plus a parent; `library_study.py` uses two workers plus a parent. The original execution overlapped those two jobs for at most eight processes, all inheriting the permitted affinity; later jobs were run after the audit completed.

`audit_logs.py` snapshots file byte lengths and accepted journal entries. A future rerun deliberately makes a new snapshot and may observe additional coordinator runs. The original snapshot and results are recorded in the files delivered with this report. `usage/*.npz` columns in `info` and `miss_info` are task, init, decision step, success, hit, vision, top1, reason, library-name index, stuck count. `top` contains up to sixteen reported members, with -1 padding. Pool files contain usage masks, frequency and two-successor closure. Arrays do not contain raw keys or full MISS tails that were absent from JSON.

`library_study.py` runs eight own-fit cell/scale jobs, uses existing R4-A full16 replays for the pruning/scene diagnostics, and creates separate frozen-fit policy-trajectory growth measurements. It never fits a method on held-out query actions. All collection policy calls, failed source rows, successful rows, admissions and deduplication counts are recorded.

`memory_yield.py` and `guard_memory.py` count arrays in actual fit artifacts. Packed byte counts are projected storage layouts, not newly built compact fit artifacts or GPU measurements. `capacity_growth.py` is an arrival-count replay before visual deduplication, not a replay of changed control. `report_tables.py` produces the report's detailed tables and exact cost arithmetic.
