#!/bin/bash
# Coordinator commands only. This file was not executed during implementation.
set -euo pipefail
cd /home/weiland/projects/openpi
export SERVER_HOST=local WORKER_HOST=timan107 LOCAL_SERVER_COUNT=1 WPS=1

r16_control() {
  taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m exp.offline_search.closed_loop.ops.h100.control "$@"
}

# Each isolated root contains exactly one arm and two frozen source identities.
# The controller is CPU-only; its explicit chain action starts the model server.
while IFS= read -r r16_entry; do
  r16_payload=${r16_entry#@local }
  r16_run_root=${r16_payload%%|*}
  r16_arm=${r16_payload#*|}
  r16_control sync --concurrent "$r16_run_root" "$r16_arm"
  r16_control chain "$r16_run_root" "$r16_arm"
done < exp/offline_search/rounds/r16/queue_smoke.txt
