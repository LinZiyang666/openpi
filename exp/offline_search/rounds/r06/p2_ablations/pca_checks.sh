#!/usr/bin/env bash
# One 3-thread process; can overlap the four 3-thread fit workers + 1-thread coordinator (16 total).
set -euo pipefail
cd /home/weiland/projects/openpi
D=exp/offline_search/rounds/r06/p2_ablations
RUN=/home/weiland/trace_runs/os_closed_loop/r06_abl
P=(taskset -c 18-25,62-69 env OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
for arm in r6p2_direct_p_sp_50 r6p2_direct_p_l10_50 r6p2_direct_g_sp_50 r6p2_direct_g_l10_50 r6p2_direct_p_sp_500 r6p2_direct_p_l10_500 r6p2_direct_g_sp_500 r6p2_direct_g_l10_500; do
  while [ ! -f "$RUN/fits/$arm.pkl" ]; do sleep 10; done
  "${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.pca_check --arm "$arm" > "$D/results/pca/${arm}_check.log" 2>&1
  echo "$arm PASS"
done
"${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.pca_validate > "$D/results/pca/validation.log" 2>&1
for model in pi05 groot; do
  "${P[@]}" -m exp.offline_search.rounds.r06.p2_ablations.pca_smoke --model "$model" --out "$RUN/pca_replays/$model" > "$D/results/pca/smoke_${model}.log" 2>&1
done
