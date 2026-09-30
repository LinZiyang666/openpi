#!/bin/bash
set -euo pipefail
cd /home/weiland/projects/openpi
phase=${1:-after}
case "$phase" in before|after) ;; *) exit 2 ;; esac
PY=(taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
ROOT=/home/weiland/trace_runs/offline_search_store
for model in pi05 groot; do
  for mode in cache miss; do
    args=()
    if [ "$mode" = miss ]; then args=(--judge periodic:1 --replay-cell "${model}_spatial_inf"); fi
    "${PY[@]}" -m exp.offline_search.closed_loop.selftest --cell "${model}_spatial_cache" --root "$ROOT" \
      --yaml "/home/weiland/trace_runs/os_closed_loop/r06_c_cal/config/r6c_cal_${model}_spatial_50.yaml" \
      --method exp.offline_search.closed_loop.probe:ProbeB0 --episodes 4 "${args[@]}" \
      --out "/tmp/r7_C2/tests/${phase}_${model}_${mode}" > "/tmp/r7_C2/tests/${phase}_${model}_${mode}.log" 2>&1
  done
done
if [ "$phase" = after ]; then
  for model in pi05 groot; do
    "${PY[@]}" -c 'import sys; from pathlib import Path; [p.unlink() for p in (Path("/tmp/r7_C2/tests")/("blind_"+sys.argv[1])).glob("decisions_*.jsonl")]' "$model"
    "${PY[@]}" -m exp.offline_search.closed_loop.selftest --cell "${model}_spatial_cache" --root "$ROOT" \
      --yaml "/home/weiland/trace_runs/os_closed_loop/r06_c_cal/config/r6c_cal_${model}_spatial_50.yaml" \
      --method exp.offline_search.closed_loop.probe:ProbeBlind --kwargs '{"budget":2}' --blind --judge guard_only \
      --out "/tmp/r7_C2/tests/blind_${model}" > "/tmp/r7_C2/tests/blind_${model}.log" 2>&1
  done
  "${PY[@]}" -m unittest exp.offline_search.rounds.r07.c2_wrist.test_dispatch exp.offline_search.rounds.r07.c2_wrist.test_method \
    > /tmp/r7_C2/tests/unit.log 2>&1
  "${PY[@]}" -m exp.offline_search.rounds.r07.c2_wrist.verify > /tmp/r7_C2/tests/verify.log 2>&1
fi
