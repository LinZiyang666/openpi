#!/usr/bin/env bash
# Long segment OFF (longseg_off_1022): success-filtered library . pi0.5 . LIBERO . full step . knob off . 4000 episodes
# H100 servers (/data/oscl_h100/openpi); 3 pi0.5 replicas; Q3 --os-judge stall_only; no --os-blind/--os-policy-tail. See longseg_off_1022/HANDBACK.md.
set -euo pipefail
cd /home/weiland/projects/openpi
export WORKER_HOST=${WORKER_HOST:-a100} WORKER_HOSTS=${WORKER_HOSTS:-a100,timan107,timan1}
env -u CUDA_VISIBLE_DEVICES -u JAX_PLATFORMS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.longseg_off_1022.build dispatch launch --name fa_lo_pi_f_f_o_q_l100
