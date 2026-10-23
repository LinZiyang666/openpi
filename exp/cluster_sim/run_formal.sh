#!/usr/bin/env bash
# Step 3: the formal replay, one fixed server configuration per side, only the robot count N varies.
# 500 episodes per N, seed 0, 40 warm-up episodes at 20 robots first (not analyzed). Results: runs/{pure,cache}/RESULTS.json
set -euo pipefail
cd /home/weiland/projects/openpi
OUT=/nvme/weiland/openpi/cluster_sim/runs
LOG=/nvme/weiland/openpi/cluster_sim/logs
LIB=/nvme/weiland/openpi/cluster_sim/library
ROBOTS=5,10,20,30,40,50,60,70,80,90,100
COMMON=(--graphs --compile all --compile-mode default --fuse23)
PURE=(--batch 1:16,2:16,3:16 --omp 4 "${COMMON[@]}")
CACHE=(--batch 1:32,2:16,3:16 --priority 1:-3,3:0 --pipe1 --omp "${CACHE_OMP:-4}" --dot --lean "${COMMON[@]}")
mkdir -p "$OUT" "$LOG"
# cache_seg: the complete system with long execution segments (owner 10-23): the robot sends every other infer
# request and executes 2 x 5 actions per reply (interval 2 x 250 ms + 2 ms); same server configuration as cache.
for run in ${MODES:-pure cache}; do
  case "$run" in
    pure) mode=pure; opts=("${PURE[@]}") ;;
    cache) mode=cache; opts=("${CACHE[@]}") ;;
    cache_seg) mode=cache; opts=("${CACHE[@]}" --stride 2) ;;
  esac
  PYTHONPATH=.:src .venv/bin/python -m exp.cluster_sim.sweep --mode "$mode" --run "$OUT/$run" --library "$LIB" \
    --robots "$ROBOTS" --seed 0 --warmup 40 "${opts[@]}" 2>&1 | tee -a "$LOG/formal_$run.log"
done
