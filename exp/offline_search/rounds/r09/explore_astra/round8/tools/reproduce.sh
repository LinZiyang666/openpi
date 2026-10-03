#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OUT=exp/offline_search/rounds/r09/explore_astra/round8
PKG=exp.offline_search.rounds.r09.explore_astra.round8.tools
taskset -c 10-21,54-65 .venv/bin/python -m "$PKG.analyze" > "$OUT/results/analyze.log" 2>&1
taskset -c 10-21,54-65 .venv/bin/python -m "$PKG.corrector" > "$OUT/results/corrector.log" 2>&1
taskset -c 10-21,54-65 .venv/bin/python -m "$PKG.diagnostics" > "$OUT/results/diagnostics.log" 2>&1
taskset -c 10-21,54-65 .venv/bin/python -m "$PKG.tasks"
taskset -c 10-21,54-65 .venv/bin/python -m "$PKG.provenance"
taskset -c 10-21,54-65 .venv/bin/python -m "$PKG.figures"
taskset -c 10-21,54-65 .venv/bin/python -m "$PKG.test_tools" > "$OUT/results/tests.log" 2>&1
taskset -c 10-21,54-65 .venv/bin/python -m "$PKG.write_reports"
