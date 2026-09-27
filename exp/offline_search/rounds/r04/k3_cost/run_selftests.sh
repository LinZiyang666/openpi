#!/bin/bash
set -euo pipefail
cd /home/weiland/projects/openpi
phase=${1:?before or after}
for model in pi05 groot; do
  for mode in cache miss; do
    args=()
    [ "$mode" = miss ] && args=(--judge periodic:1 --replay-cell ${model}_spatial_inf)
    taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      .venv/bin/python -m exp.offline_search.closed_loop.selftest \
      --cell ${model}_spatial_cache --root /dev/shm/offline_search_store \
      --yaml exp/offline_search/rounds/r04/k3_cost/dev/${model}.yaml \
      --method exp.offline_search.closed_loop.probe:ProbeB0 --episodes 4 \
      "${args[@]}" --out exp/offline_search/rounds/r04/k3_cost/results/${phase}_${model}_${mode} \
      > exp/offline_search/rounds/r04/k3_cost/results/${phase}_${model}_${mode}.log 2>&1
  done
done
