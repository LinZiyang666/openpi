#!/bin/bash
set -euo pipefail
cd /home/weiland/projects/openpi
for suite in spatial l10; do
  for lib in current big; do
    taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      .venv/bin/python -m exp.offline_search.closed_loop.selftest \
      --cell pi05_${suite}_cache --root /dev/shm/offline_search_store \
      --yaml exp/offline_search/rounds/r04/k3_cost/dev/pi05_${suite}.yaml \
      --method exp.offline_search.rounds.r04.k3_cost.method:WristMixedJudge \
      --kwargs "{\"lib\":\"$lib\",\"guards\":true,\"events\":\"none\"}" \
      --fit-artifact /home/weiland/trace_runs/offline_search_store/derived/r04/k3_cost/fits/pi05_${suite}_${lib}.pkl \
      --judge periodic:2 --no-shadow --episodes 4 \
      --out exp/offline_search/rounds/r04/k3_cost/results/wrist_${suite}_${lib} \
      > exp/offline_search/rounds/r04/k3_cost/results/wrist_${suite}_${lib}.log 2>&1
  done
done
