#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
export VLA_WORKER_HOST=auto
cd /home/weiland/projects/openpi_r16_k1ref_server
env -u CUDA_VISIBLE_DEVICES -u JAX_PLATFORMS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -c 'import sys;import exp.offline_search.rounds.r17 as r;r.__path__.append('"'"'/home/weiland/projects/openpi/exp/offline_search/rounds/r17'"'"');from exp.offline_search.rounds.r17.filter_failed_knob.formal_a import run_r16;run_r16()' run --group /data/mw_finetune/eval/fa_pi_r_u_c/group.json --agent-island /scratch/zixuans8/openpi_r16_conductor_v1 --workers 64 --port 23159 --ports 23150,23151,23152
