#!/usr/bin/env bash
set -euo pipefail
# No new fits: exact spec/kwargs/cell checked by make_arms.py and runtime.
taskset -c 34-37,78-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r05/q5_gpu/make_arms.py
