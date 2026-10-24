#!/usr/bin/env bash
# New subset seeds (seeds_1023): success-filtered library (subset seeds 20261003 and 20261004) . pi0.5 . LIBERO . reduced K1 . long segment on and off . descent round 4: ir25 for the tasks that passed every earlier level . episodes set when defined (prepare)
# H100 residents (/data/oscl_h100/openpi_r16_q3_server); 3 pi0.5 replicas; Q3 --os-judge stall_only. See seeds_1023/vla/build.py.
set -euo pipefail
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.seeds_1023.vla.build dispatch launch --name fa_x34_pi_r_k4
