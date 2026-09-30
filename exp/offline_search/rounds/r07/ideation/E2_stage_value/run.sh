#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
taskset -c 24-25,68-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r07/ideation/E2_stage_value/analyze_stage.py "$@"
