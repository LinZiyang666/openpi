#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
R9R2_PY=(taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
R9R2_MOD=exp.offline_search.rounds.r09.explore_astra.round2.tools
R9R2_DIR=exp/offline_search/rounds/r09/explore_astra/round2
for R9R2_TOOL in shared_head call_value physical_audit wrist_head prepare_confirmation; do
  "${R9R2_PY[@]}" -m "$R9R2_MOD.$R9R2_TOOL" > "$R9R2_DIR/$R9R2_TOOL.log" 2>&1
done
"${R9R2_PY[@]}" -m unittest "$R9R2_MOD.test_tools" -v > "$R9R2_DIR/tests.log" 2>&1
"${R9R2_PY[@]}" -m "$R9R2_MOD.audit" > "$R9R2_DIR/audit.log" 2>&1
# No network, GPU, policy server, or simulator is launched by this script.
