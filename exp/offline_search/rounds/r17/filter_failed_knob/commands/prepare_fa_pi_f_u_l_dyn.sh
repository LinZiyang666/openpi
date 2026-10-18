#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m exp.offline_search.rounds.r17.autoscale_1017.formal_dynamic prepare --name fa_pi_f_u_l
