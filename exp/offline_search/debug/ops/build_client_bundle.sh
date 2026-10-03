#!/usr/bin/env bash
# No remote work; output directory must be new.
set -euo pipefail
R=/home/weiland/projects/openpi
cd "$R"
exec taskset -c "${OPS_CPUS:-18-21,62-65}" env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src "$R/.venv/bin/python" -m exp.offline_search.debug.ops.build_client_bundle "$@"
