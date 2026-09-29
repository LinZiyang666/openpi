#!/usr/bin/env bash
# R6-C-v2 + SELECTION 7b replays (TAG=v2b). The replays_v2 outputs were made by
# the pre-7b code with TAG=v2 and case list without stall30.
set -euo pipefail
cd /home/weiland/projects/openpi
C_DIR=exp/offline_search/rounds/r06/ideation_Q1/method_c
TAG=${TAG:-v2b}
DRYRUN=${DRYRUN:-dryrun_v2b}
PY=(taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
for c in pi05_l10_50 groot_l10_50; do
  for variant in A floor uniform R stall R45 stall30; do
    dest=/tmp/q1_method_c_fits/replays_${TAG}/${c}_${variant}
    if [ -f "$dest/report.json" ]; then continue; fi
    "${PY[@]}" -m exp.offline_search.rounds.r06.ideation_Q1.method_c.replay_test --cell "$c" --case "$variant" --dryrun "$DRYRUN" --out "$dest" > "$C_DIR/replay_${TAG}_${c}_${variant}.log" 2>&1
  done
  dest=/tmp/q1_method_c_fits/replays_${TAG}/${c}_R_reverse
  if [ ! -f "$dest/report.json" ]; then
    "${PY[@]}" -m exp.offline_search.rounds.r06.ideation_Q1.method_c.replay_test --cell "$c" --case R --reverse --dryrun "$DRYRUN" --out "$dest" > "$C_DIR/replay_${TAG}_${c}_R_reverse.log" 2>&1
  fi
done
