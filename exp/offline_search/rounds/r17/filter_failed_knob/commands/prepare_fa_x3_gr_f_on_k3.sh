#!/usr/bin/env bash
# New subset seeds (seeds_1023): success-filtered library (subset seed 20261003) . GR00T . LIBERO . full step . long segment on . descent round 3: ir32 for the tasks that passed every earlier level . episodes set when defined (prepare)
# H100 servers (/data/oscl_h100/openpi); 5 GR00T replicas; Q3 --os-judge stall_only. See seeds_1023/vla/build.py.
set -euo pipefail
cd /home/weiland/projects/openpi
export WORKER_HOST=${WORKER_HOST:-a100} WORKER_HOSTS=${WORKER_HOSTS:-a100,timan107,timan1}
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.seeds_1023.vla.build dispatch prepare --name fa_x3_gr_f_on_k3
