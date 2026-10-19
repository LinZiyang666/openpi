#!/bin/bash
# A100 B1 DP replay (diagnostic only): same tool/inputs/recipe/fit as the 4090 B1
# native row (results/4090_native_20261018T045343Z); waits for the exclusive GPU lock.
set -euo pipefail
W=/home/exouser/xgpu_1017
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd $W
exec 9>>$W/cl18/exclusive_gpu.lock
flock 9
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
timeout --signal=TERM --kill-after=10s 310s $W/env/bin/python $W/tools/replay.py \
  --job $W/cl18/b1_dp_1018b/job.json \
  --runtime $W/rootfs/data/mw_finetune/qualification/formal_runtime_5927a753312bf5981cba81a2e389edea914bde0580523b0dc3c8ea8d58f1fa1e \
  --runtime-identity 5927a753312bf5981cba81a2e389edea914bde0580523b0dc3c8ea8d58f1fa1e \
  --inputs $W/inputs_dp3 --cache-yaml $W/cl18/b1_dp_1018b/cache.yaml \
  --output $W/cl18/b1_dp_1018b/results/a100_native_$STAMP \
  --device cuda:0 --repeats 3 --memory-gib 4 --deadline-seconds 300 --profile native
