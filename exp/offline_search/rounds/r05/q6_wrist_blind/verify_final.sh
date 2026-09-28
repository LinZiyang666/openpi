#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
Q=exp/offline_search/rounds/r05/q6_wrist_blind
M=exp.offline_search.rounds.r05.q6_wrist_blind
P=(taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src XDG_CACHE_HOME=/tmp/q6_cache .venv/bin/python)
"${P[@]}" -m "$M.tests" > "$Q/results/unit_final.log" 2>&1
echo 'PASS unit and parity'
"${P[@]}" -m "$M.run_checks" plugin --tag final > "$Q/results/plugin_final.log" 2>&1
echo 'PASS plugin matrix'
"${P[@]}" -m "$M.run_checks" existing --tag final > "$Q/results/existing_final.log" 2>&1
echo 'PASS existing selftests'
"${P[@]}" -m "$M.run_checks" smoke --tag final > "$Q/results/smoke_final.log" 2>&1
echo 'PASS harness smokes'
"${P[@]}" -m "$M.validate_arms" --tag final > "$Q/results/arms_final.log" 2>&1
echo 'PASS emitted arms'
"${P[@]}" -m "$M.concurrency_test" --config all --out /tmp/q6_concurrency_final > "$Q/results/concurrency_final.log" 2>&1
echo 'PASS eight-connection parity'
"${P[@]}" -m "$M.run_checks" evidence --tag final > "$Q/results/evidence_final.log" 2>&1
echo 'PASS final evidence'
