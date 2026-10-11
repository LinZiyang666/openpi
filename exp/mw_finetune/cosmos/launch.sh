#!/usr/bin/env bash
# Reuse the proven Cosmos dependency stack; every GPU mode takes the R17 lock.
set -euo pipefail
MW_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
COSMOS_REPO=${COSMOS_REPO:-/home/weiland/projects/openpi_ext/lines/cosmos-policy}
COSMOS_PYTHON="$COSMOS_REPO/.venv/bin/python"
COSMOS_NV="$COSMOS_REPO/.venv/lib/python3.10/site-packages/nvidia"
export CUDA_HOME="$COSMOS_NV" CUDNN_HOME="$COSMOS_NV/cudnn"
export HF_HOME=${HF_HOME:-/data/cosmos/hf_home}
export PYTHONPATH="$MW_REPO:$MW_REPO/src:$MW_REPO/packages/openpi-client/src:$COSMOS_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
cd "$MW_REPO"
MODE=${1:-smoke}
if (( $# )); then shift; fi
case "$MODE" in
  smoke)
    exec flock /tmp/r17_gpu.lock timeout 890s "$COSMOS_PYTHON" -m exp.mw_finetune.cosmos.smoke "$@"
    ;;
  probe)
    exec flock /tmp/r17_gpu.lock timeout --signal=TERM --kill-after=15s 1180s "$COSMOS_PYTHON" -m exp.mw_finetune.cosmos.reconstruction_probe "$@"
    ;;
  serve)
    nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
    exec flock /tmp/r17_gpu.lock "$COSMOS_PYTHON" -m exp.mw_finetune.serve --model cosmos "$@"
    ;;
  download)
    exec "$COSMOS_PYTHON" -m exp.mw_finetune.cosmos.download "$@"
    ;;
  test)
    exec "$COSMOS_PYTHON" -m pytest -q exp/mw_finetune/cosmos/test_policy.py "$@"
    ;;
  *) echo "Usage: launch.sh smoke|probe|serve|download|test [arguments]" >&2; exit 2 ;;
esac
