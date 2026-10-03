#!/bin/bash
R=/home/weiland/trace_runs/dual_20260923
cd /home/weiland/projects/openpi
export OMP_NUM_THREADS=4 HOME=/home/weiland
.venv/bin/python $R/ops/replay_pi05.py 2>&1 | grep --line-buffered -v -i 'warn\|pynvml' | tee $R/audit/replay_pi05.log
echo "REPLAY_EXIT=${PIPESTATUS[0]}" | tee -a $R/audit/replay_pi05.log
