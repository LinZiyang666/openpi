#!/usr/bin/env bash
# coordinator 10-18 18:4x: continuation of fa_co_r_f_l (13,729 kept, 16,271 remaining), stopped to switch to adaptive workers (owner order).
set -euo pipefail
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.stage --run /data/mw_finetune/eval/fa_co_r_f_lc2
