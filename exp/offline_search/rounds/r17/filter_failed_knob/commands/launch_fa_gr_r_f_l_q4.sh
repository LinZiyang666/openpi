#!/usr/bin/env bash
# 10-22 03:0x: fa_gr_r_f_l_q (success-filtered library . GR00T . LIBERO . reduced K1 . Q3 levels) resumed after libero_spatial
# finished (2,000/2,000) and its resident idled: four libero_10 residents (23150-23152 kept, 23154 new), 23153 retired.
# Resumes from the group's own journal. See launch_fa_gr_r_f_l_q.sh.
set -euo pipefail
cd /home/weiland/projects/openpi
env -u CUDA_VISIBLE_DEVICES -u JAX_PLATFORMS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.q3_port_1020.runner run --group /data/mw_finetune/eval/fa_gr_r_f_l_q/group.json --agent-island /scratch/zixuans8/openpi_r16_conductor_v1 --port 23159 --ports 23150,23151,23152,23154 --groot-suite-servers 4,0
