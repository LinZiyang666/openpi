#!/usr/bin/env bash
# Q3 stall gate (q3_port_1020): original library . pi0.5 . LIBERO . full step . levels ir45/40/32/25 . 16000 episodes
# H100 servers (/data/oscl_h100/openpi); 3 pi0.5 replicas. See q3_port_1020/HANDBACK.md.
set -euo pipefail
cd /home/weiland/projects/openpi
# R15 multi-node (d2320e5b, opt-in): driver + agent on timan107, agent-only on timan1 (timan107 accepts
# campus connections only); per-node counts from the central scheduler.
export WORKER_HOST=timan107 WORKER_HOSTS=timan107,timan1
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.q3_port_1020.build dispatch prepare --name fa_pi_f_u_l_q
