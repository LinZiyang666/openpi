#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
B=exp/offline_search/rounds/r05/q5_gpu
for scale in 50 500; do
  for mode in shadow serve; do
    taskset -c 34-37,78-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu Q5_GPU=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$B/boot:.:src .venv/bin/python "$B/stage1_test.py" --scale "$scale" --mode "$mode" > "$B/results/stage1_${scale}_${mode}.log" 2>&1
    echo "PASS real stage1 $scale $mode"
  done
done
