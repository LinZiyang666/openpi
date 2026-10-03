#!/bin/bash
# run_audit.sh [arm ...] -- full value audit, single-threaded BLAS per worker process
R=/home/weiland/trace_runs/dual_20260923
cd /home/weiland/projects/openpi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python $R/ops/audit_values.py "$@" 2>&1 | grep --line-buffered -v -i 'warn\|pynvml' | tee $R/audit/run.log
echo "AUDIT_EXIT=${PIPESTATUS[0]}" | tee -a $R/audit/run.log
