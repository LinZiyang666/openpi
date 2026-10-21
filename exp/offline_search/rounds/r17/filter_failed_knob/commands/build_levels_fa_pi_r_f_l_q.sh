#!/usr/bin/env bash
# Q3 port: define fa_pi_r_f_l_q (levels ir45/40/32/25 for the cells whose new knob-off cohort is below pure A).
# Run after fa_pi_r_f_o_q is complete; then prepare_/launch_fa_pi_r_f_l_q.sh. CPU only.
set -euo pipefail
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.q3_port_1020.build levels --model pi05 --mode reduced --arm filtered
