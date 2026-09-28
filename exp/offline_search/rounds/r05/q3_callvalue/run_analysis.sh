#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
exec taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HIGHS_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r05/q3_callvalue/analyze.py "$@"
