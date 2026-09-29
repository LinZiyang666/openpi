#!/usr/bin/env bash
set -euo pipefail
Q2_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$Q2_DIR/../../../../.." && pwd)
mkdir -p "$Q2_DIR/.tmp" "$Q2_DIR/.mplconfig"
cd -- "$REPO_DIR"
taskset -c 2-5,46-49 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  TMPDIR="$Q2_DIR/.tmp" MPLCONFIGDIR="$Q2_DIR/.mplconfig" \
  .venv/bin/python "$Q2_DIR/pilot_q2.py" "$@"
