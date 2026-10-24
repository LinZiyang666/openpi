#!/usr/bin/env bash
# Q3 stall gate (q3_port_1020): success-filtered library . GR00T . LIBERO . full step . knob off . 4000 episodes
# 4090 servers from /home/weiland/projects/openpi_r16_q3_server; 5 GR00T replicas. See q3_port_1020/HANDBACK.md.
set -euo pipefail
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.q3_port_1020.build dispatch prepare --name fa_gr_f_f_o_q
