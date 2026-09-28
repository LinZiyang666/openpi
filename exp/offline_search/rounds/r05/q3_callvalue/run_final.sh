#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
Q3=exp/offline_search/rounds/r05/q3_callvalue
bash "$Q3/run_analysis.sh" --from-audited "$Q3/results" > "$Q3/final_analysis.log" 2>&1
bash "$Q3/run_analysis.sh" --from-audited "$Q3/results" --out "$Q3/rerun" > "$Q3/rerun.log" 2>&1
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HIGHS_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python "$Q3/verify.py" > "$Q3/verification.log" 2>&1
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HIGHS_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.closed_loop.ops.emit_arms --run-root /tmp/q3_empty_arms_check --spec "$Q3/arms_q3.json" > "$Q3/emitter.log" 2>&1
