#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
mkdir -p /tmp/r6_q2
for script in measure price_index library_proxy analyze extend verify write_report; do
  taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    HIGHS_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/r6_q2/matplotlib .venv/bin/python \
    "exp/offline_search/rounds/r06/ideation_Q2/${script}.py" \
    > "exp/offline_search/rounds/r06/ideation_Q2/${script}.log" 2>&1
done
