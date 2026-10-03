#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
R9B_PY=(taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
R9B_MOD=exp.offline_search.rounds.r09.explore_astra.round2b.tools
"${R9B_PY[@]}" -m "$R9B_MOD.labels"
"${R9B_PY[@]}" -m "$R9B_MOD.screen"
"${R9B_PY[@]}" -m "$R9B_MOD.causal"
"${R9B_PY[@]}" -m "$R9B_MOD.reanchor"
"${R9B_PY[@]}" -m "$R9B_MOD.ablation"
"${R9B_PY[@]}" -m "$R9B_MOD.responses"
"${R9B_PY[@]}" -m "$R9B_MOD.bounds"
"${R9B_PY[@]}" -m "$R9B_MOD.prepare_confirmation"
"${R9B_PY[@]}" -m unittest "$R9B_MOD.test_tools" -v
"${R9B_PY[@]}" -m "$R9B_MOD.audit"
# No sync, remote operation, GPU inference, or closed-loop launch.
# live_results is intentionally separate: it reads a changing coordinator run.
