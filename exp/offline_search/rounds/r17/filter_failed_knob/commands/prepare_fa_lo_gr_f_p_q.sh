#!/usr/bin/env bash
# Same-card reference for longseg_off_1022: no library (pure inference) . GR00T . LIBERO . full step . H100 pure-inference reference (formal-A pure A, C10: every decision a model call) . 1000 episodes
# H100 servers (/data/oscl_h100/openpi); 4 GR00T replicas; source arms ['r14dt_groot_l10_50_pure_c10', 'r14dt_groot_spatial_50_pure_c10']. See longseg_off_1022/HANDBACK.md.
set -euo pipefail
cd /home/weiland/projects/openpi
export WORKER_HOST=${WORKER_HOST:-a100} WORKER_HOSTS=${WORKER_HOSTS:-a100,timan107,timan1}
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.longseg_off_1022.build dispatch prepare --name fa_lo_gr_f_p_q
