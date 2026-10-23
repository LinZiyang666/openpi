#!/usr/bin/env bash
# Long segment OFF: define fa_lo_gr_r_f_l_q (ir45/40/32/25 for the cells whose OFF cohort is below the same-step pure A).
# Run after fa_lo_gr_r_f_o_q is complete; then prepare_/launch_fa_lo_gr_r_f_l_q.sh. CPU only.
set -euo pipefail
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.longseg_off_1022.build levels --model groot --mode reduced
