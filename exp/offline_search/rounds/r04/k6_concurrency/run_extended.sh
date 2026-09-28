#!/bin/bash
set -euo pipefail
cd /home/weiland/projects/openpi
base=exp/offline_search/rounds/r04/k6_concurrency
phase=${1:-all}
py=(taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python)
if [[ $phase != dependent ]]; then
    "${py[@]}" "$base/prepare_extended.py"
    "${py[@]}" "$base/k5_prepare_arms.py" --run-root /tmp/k6_arms_check > "$base/results/installed/arms_check.log" 2>&1
    "${py[@]}" "$base/k5_validate_estimator.py" > "$base/results/installed/validate_estimator.log" 2>&1
    echo 'PASS K5 estimator validation'
fi
if [[ $phase == independent ]]; then exit 0; fi
for name in check_replay_estimator check_ledger final_audit summarize_checks; do
    "${py[@]}" "$base/k5_${name}.py" > "$base/results/installed/${name}.log" 2>&1
    echo "PASS K5 $name"
done
