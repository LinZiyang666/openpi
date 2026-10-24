#!/usr/bin/env bash
# New subset seeds: define the full-step levels ir45/40/32/25 of pi05 for the cells whose knob-off cohort
# is below the same-card same-step pure reference. Run after those knob-off units/group. CPU only.
set -euo pipefail
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.seeds_1023.vla.build levels --model pi05 --mode full
