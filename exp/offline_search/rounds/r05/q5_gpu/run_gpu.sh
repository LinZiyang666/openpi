#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
B=exp/offline_search/rounds/r05/q5_gpu
P=(taskset -c 34-37,78-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src)
for family in AWM MixedJudge; do
  for scale in 50 500; do
    for run in 1 2; do
      name=pi05_l10_${scale}_${family}
      "${P[@]}" .venv/bin/python "$B/gpu_replay.py" --config "$name" --run "$run" > "$B/results/gpu_${name}_r${run}.log" 2>&1
      echo "PASS $name run $run"
    done
  done
done
