#!/usr/bin/env bash
# Coordinator commands only. This implementation never executes this script.
set -euo pipefail
cd /home/weiland/projects/openpi
R16_DIR=/home/weiland/projects/openpi/exp/offline_search/rounds/r16
R16_ISLAND=${R16_CONDUCTOR_ISLAND:-/scratch/zixuans8/openpi_r16_conductor_v1}
R16_CPU=(taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python)
R16_MODE=${1:?smoke, dev, or test}
case "$R16_MODE" in
  smoke) R16_QUEUE="$R16_DIR/queue_conductor_smoke.txt"; R16_OPTIONS=(--workers 4 --smoke-episodes 2) ;;
  dev) R16_QUEUE="$R16_DIR/queue_conductor_dev.txt"; R16_OPTIONS=(--workers 64) ;;
  test) R16_QUEUE="$R16_DIR/queue_test.txt"; R16_OPTIONS=(--workers 64) ;;
  *) exit 2 ;;
esac
while IFS= read -r R16_LINE; do
  [[ "$R16_LINE" == '@local '* && "$R16_LINE" != *'|'* ]] || { echo 'Expected resident group queue' >&2; exit 2; }
  R16_GROUP=${R16_LINE#@local }
  "${R16_CPU[@]}" -m exp.offline_search.rounds.r16.conductor_runner run \
    --group "$R16_GROUP" --agent-island "$R16_ISLAND" --port 23179 "${R16_OPTIONS[@]}"
done < "$R16_QUEUE"
