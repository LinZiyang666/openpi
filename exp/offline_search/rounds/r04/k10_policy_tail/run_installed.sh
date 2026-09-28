#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
B=exp/offline_search/rounds/r04/k10_policy_tail
P=(taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
"${P[@]}" "$B/prepare_checks.py" installed
"${P[@]}" "$B/prepare_extra.py"
for group in k2 k1 k4; do
  bash "$B/dev/installed_${group}.sh" > "$B/results/installed/existing_${group}.log" 2>&1
  echo "PASS $group matrix"
done
"${P[@]}" "$B/check_parity.py" installed > "$B/results/installed/parity.log" 2>&1
echo 'PASS six byte-parity modes'
"${P[@]}" "$B/dev/installed_test_overlay.py" --source installed --out /tmp/k10_overlay_installed > "$B/results/installed/overlay.log" 2>&1
"${P[@]}" "$B/dev/installed_run_random_replays.py" installed > "$B/results/installed/random_replays.log" 2>&1
echo 'PASS K5 overlay and four replays'
"${P[@]}" "$B/k5_prepare_arms.py" --run-root /tmp/k10_arms_check > "$B/results/installed/k5_arms.log" 2>&1
for check in validate_estimator check_replay_estimator check_ledger final_audit summarize_checks; do
  "${P[@]}" "$B/k5_${check}.py" > "$B/results/installed/k5_${check}.log" 2>&1
  echo "PASS K5 $check"
done
"${P[@]}" "$B/concurrency_test.py" --source installed --config all --out /tmp/k10_installed_concurrency > "$B/results/installed/concurrency.log" 2>&1
echo 'PASS K6 12-config concurrency matrix'
"${P[@]}" "$B/edge_test.py" --source installed --out /tmp/k10_installed_k6_edges > "$B/results/installed/k6_edges.log" 2>&1
echo 'PASS K6 edges'
"${P[@]}" "$B/run_k7.py" > "$B/results/installed/k7.log" 2>&1
echo 'PASS K7 checks'
"${P[@]}" "$B/tail_concurrency_test.py" --source installed --config all --out /tmp/k10_installed_tail_concurrency > "$B/results/installed/tail_concurrency.log" 2>&1
echo 'PASS K10 9-config concurrency matrix'
"${P[@]}" "$B/policy_tail_test.py" --source installed --out /tmp/k10_installed_edges > "$B/results/installed/edges.log" 2>&1
echo 'PASS K10 edge/transform/L10/ledger tests'
"${P[@]}" "$B/run_tail_matrix.py" installed > "$B/results/installed/tail_matrix.log" 2>&1
echo 'PASS K10 nine plugin selftests'
