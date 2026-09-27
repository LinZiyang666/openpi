#!/bin/bash
set -euo pipefail
OUT=exp/offline_search/rounds/r04/k4_eval/results
ARTIFACT_ROOT=$(mktemp -d /tmp/k4_plugin.XXXXXX)
echo "$ARTIFACT_ROOT" > "$OUT/plugin_artifact_root.txt"
for model in pi05 groot; do
  for mode in cache miss; do
    args=()
    method=exp.offline_search.closed_loop.probe:ProbeB0
    if [ "$mode" = miss ]; then
      args=(--judge periodic:1 --replay-cell ${model}_spatial_inf)
      method=exp.offline_search.rounds.r04.k4_eval.seeded_inference:SeededInference
    fi
    taskset -c 30-33,74-77 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src \
      .venv/bin/python -m exp.offline_search.closed_loop.selftest --cell ${model}_spatial_cache \
      --root /home/weiland/trace_runs/offline_search_store \
      --yaml exp/trace_dual/config/tr_${model}_sp_cache.yaml --method "$method" --episodes 2 \
      --out "$ARTIFACT_ROOT/${model}_${mode}" "${args[@]}" > "$OUT/plugin_${model}_${mode}.log" 2>&1
    cp "$ARTIFACT_ROOT/${model}_${mode}/selftest_report.json" "$OUT/plugin_${model}_${mode}.json"
  done
done
