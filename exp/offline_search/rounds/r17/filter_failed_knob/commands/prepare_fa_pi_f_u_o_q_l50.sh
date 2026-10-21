#!/usr/bin/env bash
# Q3 stall gate (q3_port_1020): original library . pi0.5 . LIBERO . full step . knob off . 4000 episodes
# H100 servers (/data/oscl_h100/openpi); 3 pi0.5 replicas. See q3_port_1020/HANDBACK.md.
set -euo pipefail
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.q3_port_1020.build dispatch prepare --name fa_pi_f_u_o_q_l50
