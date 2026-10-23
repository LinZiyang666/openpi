#!/usr/bin/env bash
# Same-card reference for longseg_off_1022: no library (pure inference) . pi0.5 . LIBERO . reduced K1 . H100 pure-inference reference C5 (a model call every 5 controls, no long segment) . 1000 episodes
# H100 servers (/data/oscl_h100/openpi); 3 pi0.5 replicas; source arms ['r16p1_pi05_l10_k1_pure_c10', 'r16p1_pi05_spatial_k1_pure_c10']. See longseg_off_1022/HANDBACK.md.
set -euo pipefail
cd /home/weiland/projects/openpi
export WORKER_HOST=${WORKER_HOST:-a100} WORKER_HOSTS=${WORKER_HOSTS:-a100,timan107,timan1}
env -u CUDA_VISIBLE_DEVICES -u JAX_PLATFORMS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.longseg_off_1022.build dispatch launch --name fa_lo_pi_r_p5_q_s
