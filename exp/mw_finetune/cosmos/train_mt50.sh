#!/usr/bin/env bash
# Use the existing Cosmos island on either host. Coordinator schedules the GPU.
set -euo pipefail
MW_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ -d /data/cosmos/cosmos-policy ]]; then
  COSMOS_REPO=${COSMOS_REPO:-/data/cosmos/cosmos-policy}
else
  COSMOS_REPO=${COSMOS_REPO:-/home/weiland/projects/openpi_ext/lines/cosmos-policy}
fi
COSMOS_NV="$COSMOS_REPO/.venv/lib/python3.10/site-packages/nvidia"
export COSMOS_REPO
export PYTHONPATH="$MW_REPO:$MW_REPO/src:$MW_REPO/packages/openpi-client/src:/data/cosmos/r17_runtime_deps310:$COSMOS_REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME="$COSMOS_NV" CUDNN_HOME="$COSMOS_NV/cudnn"
export LD_LIBRARY_PATH="$COSMOS_NV/cuda_runtime/lib:$COSMOS_NV/cuda_nvrtc/lib:$COSMOS_NV/cudnn/lib:$COSMOS_NV/cublas/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME=/data/cosmos/hf_home HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export DETERMINISTIC=False
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$MW_REPO"
exec "$COSMOS_REPO/.venv/bin/python" -m exp.mw_finetune.cosmos.train_mt50 "$@"
