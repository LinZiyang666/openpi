#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home/weiland/projects/openpi/src:/home/weiland/projects/openpi/packages/openpi-client/src:/home/weiland/projects/openpi
exec /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.qualification.run_slot --slot exp/r17_integration/qualification/specs/slot_G2_cosmos_libero_A.json "$@"
