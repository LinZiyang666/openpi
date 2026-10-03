#!/bin/bash
set -euo pipefail
cd /home/weiland/projects/openpi
R9PY=(taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python)
R9MOD=exp.offline_search.rounds.r09.explore_astra.tools
"${R9PY[@]}" -m "$R9MOD.extract" --workers 6
"${R9PY[@]}" -m "$R9MOD.routing"
"${R9PY[@]}" -m "$R9MOD.shadows"
"${R9PY[@]}" -m "$R9MOD.student"
"${R9PY[@]}" -m "$R9MOD.student_audit"
"${R9PY[@]}" -m "$R9MOD.shadow_memory"
"${R9PY[@]}" -m "$R9MOD.horizon_metric"
"${R9PY[@]}" -m "$R9MOD.call_value"
"${R9PY[@]}" -m "$R9MOD.routing_uncertainty"
"${R9PY[@]}" -m "$R9MOD.evidence"
"${R9PY[@]}" -m "$R9MOD.prepare_confirmation"
"${R9PY[@]}" -m unittest "$R9MOD.test_tools" "$R9MOD.test_deployment" -v
"${R9PY[@]}" -m "$R9MOD.audit_outputs"
