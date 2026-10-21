#!/usr/bin/env bash
# Q3 stall gate (q3_port_1020): success-filtered library . pi0.5 . LIBERO . full step . levels ir45/40/32/25 . 16000 episodes
# H100 servers (/data/oscl_h100/openpi); 3 pi0.5 replicas. See q3_port_1020/HANDBACK.md.
set -euo pipefail
cd /home/weiland/projects/openpi
env -u CUDA_VISIBLE_DEVICES -u JAX_PLATFORMS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.q3_port_1020.build dispatch launch --name fa_pi_f_f_l_q_s100
