#!/bin/bash
R=/home/weiland/trace_runs/dual_20260923
cd /home/weiland/projects/openpi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python $R/ops/cross_check.py 2>&1 | grep --line-buffered -v -i 'warn\|pynvml' | tee $R/audit/cross.log
echo "CROSS_EXIT=${PIPESTATUS[0]}" | tee -a $R/audit/cross.log
