#!/usr/bin/env bash
# New subset seeds (seeds_1023): success-filtered library (subset seeds 20261003 and 20261004) . pi0.5 . LIBERO . reduced K1 . long segment on and off . descent round 1: ir45 for the tasks that passed every earlier level . 10000 episodes
# H100 residents (/data/oscl_h100/openpi_r16_q3_server); 3 pi0.5 replicas; Q3 --os-judge stall_only. See seeds_1023/vla/build.py.
set -euo pipefail
cd /home/weiland/projects/openpi
env -u CUDA_VISIBLE_DEVICES -u JAX_PLATFORMS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.longseg_off_1022.runner run --group /data/mw_finetune/eval/fa_x34_pi_r_k1/group.json --agent-island /scratch/zixuans8/openpi_r16_conductor_v1 --port 23157 --ports 23150,23151,23152 --no-local-workers
