#!/usr/bin/env bash
set -euo pipefail
ROOT=${MW_ROOT:-/data/mw_finetune}
VENV=${FASTWAM_VENV:-$ROOT/envs/fastwam}
CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
if test "${R17_GPU_LOCK_HELD:-}" != 1; then
    exec flock -w 900 /tmp/r17_gpu.lock env R17_GPU_LOCK_HELD=1 bash "$0" "$@"
fi
nvidia-smi --query-gpu=name,memory.free --format=csv
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if test "$FREE" -lt 65000; then
    echo "FastWAM requires an idle H100 (>=65000 MiB free); refusing to interfere with other jobs" >&2
    exit 1
fi
for arg in "$@"; do
    if test "$arg" = --probe; then
        exec timeout --signal=TERM 900 "$VENV/bin/python" -m exp.mw_finetune.fastwam.train "$@"
    fi
done
exec "$VENV/bin/python" -m exp.mw_finetune.fastwam.train "$@"
