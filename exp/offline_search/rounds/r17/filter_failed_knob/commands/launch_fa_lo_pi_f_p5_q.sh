#!/usr/bin/env bash
# Same-card reference for longseg_off_1022: no library (pure inference) . pi0.5 . LIBERO . full step . H100 pure-inference reference C5 (a model call every 5 controls, no long segment) . 1000 episodes
# H100 servers (/data/oscl_h100/openpi); 3 pi0.5 replicas; source arms ['ffk1015_pi05_libero_full_formal_A_pure_l10_r1_c10', 'ffk1015_pi05_libero_full_formal_A_pure_spatial_c10']. See longseg_off_1022/HANDBACK.md.
set -euo pipefail
cd /home/weiland/projects/openpi
export WORKER_HOST=${WORKER_HOST:-a100} WORKER_HOSTS=${WORKER_HOSTS:-a100,timan107,timan1}
env -u CUDA_VISIBLE_DEVICES -u JAX_PLATFORMS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.longseg_off_1022.build dispatch launch --name fa_lo_pi_f_p5_q
