#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
C_DIR=exp/offline_search/rounds/r06/ideation_Q1/method_c
for cell in pi05_l10_50 pi05_l10_500 pi05_spatial_50 pi05_spatial_500 groot_l10_50 groot_l10_500 groot_spatial_50 groot_spatial_500; do
  if [ -f "/tmp/q1_method_c_fits/prefits/r6c_cal_${cell}.pkl" ]; then continue; fi
  taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.ideation_Q1.method_c.prefit --spec "$C_DIR/recording_prefit_specs.json" --name "r6c_cal_${cell}" > "$C_DIR/prefit_recording_${cell}.log" 2>&1
done
