#!/bin/bash
set -euo pipefail
cd /home/weiland/projects/openpi
K9=exp/offline_search/rounds/r04/k9_gpu_retrieval
precision=${K9_PRECISION:-float32}
for run in "$@"; do
  for suite in spatial l10; do
    for scale in 50 500; do
      for method in AWM MixedJudge; do
        cell=pi05_${suite}_${scale}_${method}
        nvidia-smi > "$K9/admission_${cell}_${precision}_r${run}.txt"
        taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src TMPDIR=/tmp/k9_scratch JAX_PLATFORMS=cpu .venv/bin/python "$K9/benchmark.py" --config "$cell" --precision "$precision" --run "$run" > "$K9/bench_${cell}_${precision}_r${run}.log" 2>&1
      done
    done
  done
done
