#!/usr/bin/env bash
# Coordinator executes H100; root may execute approved A100 only.
set -euo pipefail
XGPU_WORK=/home/exouser/xgpu_1017
XGPU_TAG=${1:?provide h100 or a100}
case "$XGPU_TAG" in
  h100) XGPU_PYTHON=/home/exouser/openpi/.venv/bin/python ;;
  a100) XGPU_PYTHON="$XGPU_WORK/env/bin/python" ;;
  *) exit 2 ;;
esac
XGPU_PROFILE=${2:-native}
XGPU_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$XGPU_WORK"
timeout --signal=TERM --kill-after=10s 310s "$XGPU_PYTHON" "$XGPU_WORK/tools/replay.py" \
  --job "$XGPU_WORK/prepared/job.json" \
  --runtime "$XGPU_WORK/rootfs/data/mw_finetune/qualification/formal_runtime_5927a753312bf5981cba81a2e389edea914bde0580523b0dc3c8ea8d58f1fa1e" \
  --runtime-identity 5927a753312bf5981cba81a2e389edea914bde0580523b0dc3c8ea8d58f1fa1e \
  --inputs "$XGPU_WORK/inputs_dp3" --cache-yaml "$XGPU_WORK/prepared/cache.yaml" \
  --output "$XGPU_WORK/results/${XGPU_TAG}_${XGPU_PROFILE}_${XGPU_STAMP}" \
  --device cuda:0 --repeats 3 --memory-gib 4 --deadline-seconds 300 --profile "$XGPU_PROFILE"
