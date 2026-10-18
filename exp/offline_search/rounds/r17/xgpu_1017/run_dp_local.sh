#!/usr/bin/env bash
# Coordinator only: 4 GiB allocator budget + 512 MiB free headroom; <=300 s.
set -euo pipefail
XGPU_WORK=/home/weiland/projects/openpi/exp/offline_search/rounds/r17/xgpu_1017
XGPU_PROFILE=${1:-native}
XGPU_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /home/weiland/projects/openpi
timeout --signal=TERM --kill-after=10s 310s .venv/bin/python "$XGPU_WORK/replay.py" \
  --job /data/mw_finetune/eval/fa_dp_f_u_o/campaign.json \
  --runtime /data/mw_finetune/qualification/formal_runtime_5927a753312bf5981cba81a2e389edea914bde0580523b0dc3c8ea8d58f1fa1e \
  --runtime-identity 5927a753312bf5981cba81a2e389edea914bde0580523b0dc3c8ea8d58f1fa1e \
  --inputs "$XGPU_WORK/inputs_dp3" \
  --cache-yaml /data/mw_finetune/eval/fa_dp_f_u_o/assets/cache_yamls/n50_sp_chrg_koff.yaml \
  --output "$XGPU_WORK/results/4090_${XGPU_PROFILE}_${XGPU_STAMP}" \
  --device cuda:0 --repeats 3 --memory-gib 4 --deadline-seconds 300 --profile "$XGPU_PROFILE"
