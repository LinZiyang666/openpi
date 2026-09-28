#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
B=exp/offline_search/rounds/r05/q5_gpu
for family in AWM MixedJudge; do
  for scale in 50 500; do
    for mode in shadow serve; do
      name=${scale}_${family}_${mode}
      taskset -c 34-37,78-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$B/boot:.:src .venv/bin/python "$B/stack_test.py" --config "pi05_l10_${scale}_${family}" --mode "$mode" --out "/tmp/q5_stack_${Q5_RUN:-dev64}_$name" > "$B/results/stack_$name.log" 2>&1
      echo "PASS $name"
    done
  done
done
