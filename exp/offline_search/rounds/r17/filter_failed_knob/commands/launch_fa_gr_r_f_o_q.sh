#!/usr/bin/env bash
# Q3 stall gate (q3_port_1020): success-filtered library . GR00T . LIBERO . reduced K1 . knob off . 4000 episodes
# 10-21: libero_spatial (220 steps) finishes first; 3 libero_10 + 1 spatial residents (conductor_runner --groot-suite-servers).
# 4090 servers from /home/weiland/projects/openpi_r16_q3_server; 4 GR00T replicas. See q3_port_1020/HANDBACK.md.
set -euo pipefail
cd /home/weiland/projects/openpi
env -u CUDA_VISIBLE_DEVICES -u JAX_PLATFORMS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.q3_port_1020.runner run --group /data/mw_finetune/eval/fa_gr_r_f_o_q/group.json --agent-island /scratch/zixuans8/openpi_r16_conductor_v1 --port 23159 --ports 23150,23151,23152,23153 --groot-suite-servers 3,1
