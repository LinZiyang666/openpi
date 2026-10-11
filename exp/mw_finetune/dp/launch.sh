#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD:$PWD/src:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
DP_PYTHON=${DP_PYTHON:-/data/mw_finetune/dp/env/bin/python}
nvidia-smi --query-gpu=name,memory.free --format=csv
DP_PROBE_UPDATES=0
DP_PROBE_NEXT=0
for DP_ARGUMENT in "$@"; do
  if [[ "$DP_PROBE_NEXT" == 1 ]]; then
    DP_PROBE_UPDATES="$DP_ARGUMENT"; DP_PROBE_NEXT=0
  fi
  case "$DP_ARGUMENT" in
    --probe) DP_PROBE_NEXT=1 ;;
    --probe=*) DP_PROBE_UPDATES="${DP_ARGUMENT#--probe=}" ;;
  esac
done
if [[ "$DP_PROBE_UPDATES" =~ ^[0-9]+$ ]] && (( DP_PROBE_UPDATES > 0 )); then
  # Explicit owner concurrent-smoke exception: never take the long-job GPU lock.
  exec timeout 850s "$DP_PYTHON" -m exp.mw_finetune.dp.train "$@"
else
  exec flock /tmp/r17_gpu.lock "$DP_PYTHON" -m exp.mw_finetune.dp.train "$@"
fi
