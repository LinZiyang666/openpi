#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
B=exp/offline_search/rounds/r05/q1_commit
P=(taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python)
"${P[@]}" "$B/test_methods.py" > /tmp/q1_methods_final.log 2>&1
"${P[@]}" "$B/validate_delivery.py" > /tmp/q1_delivery.log 2>&1
# Three waiting drivers + three single-process workers. Together with the
# regression driver/worker this never exceeds eight Python processes.
"${P[@]}" "$B/plugin_matrix.py" --out /tmp/q1_final_matrix > /tmp/q1_matrix_final.log 2>&1 & matrix_pid=$!
"${P[@]}" "$B/concurrency_test.py" --source installed --config all --out /tmp/q1_final_concurrency > /tmp/q1_concurrency.log 2>&1 & concurrency_pid=$!
"${P[@]}" "$B/monitor_fit_test.py" > /tmp/q1_monitor.log 2>&1 & monitor_pid=$!
wait "$matrix_pid"
wait "$concurrency_pid"
wait "$monitor_pid"
