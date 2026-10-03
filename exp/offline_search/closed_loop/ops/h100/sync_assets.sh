#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$HERE/../../../../.." && pwd)
cd "$R"
exec taskset -c 14-17,58-61 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "$R/.venv/bin/python" -m exp.offline_search.closed_loop.ops.h100.control sync "$@"
