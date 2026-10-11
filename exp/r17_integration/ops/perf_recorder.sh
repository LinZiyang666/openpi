#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT=/data/mw_finetune/perf
mkdir -p "$OUT"
if [[ -f "$OUT/recorder.pid" ]] && kill -0 "$(cat "$OUT/recorder.pid")" 2>/dev/null; then
    printf 'Recorder already running: PID %s\n' "$(cat "$OUT/recorder.pid")"
    exit 0
fi
nohup setsid env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 -u "$ROOT/exp/r17_integration/ops/perf_recorder.py" "$@" \
    >> "$OUT/recorder.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUT/recorder.pid"
disown "$PID"
sleep 1
kill -0 "$PID"
printf 'Recorder PID %s; stop: kill -TERM %s\n' "$PID" "$PID"
