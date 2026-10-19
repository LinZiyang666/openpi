#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
env -u CUDA_VISIBLE_DEVICES -u JAX_PLATFORMS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r16.conductor_runner run --group /data/mw_finetune/eval/fa_gr_r_u_l/group.json --agent-island /scratch/zixuans8/openpi_r16_conductor_v1 --port 23159 --ports 23150,23151,23152,23153
