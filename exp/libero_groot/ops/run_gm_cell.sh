#!/usr/bin/env bash
# One G-M latency cell for LIBERO x GR00T, with the CUDA-graph evidence.
#
# Three steps, because an external profiler only finalises its report after the
# measured process exits: measure (writes an uncertified record), export the
# trace, then certify (reads the trace back and fills the launch / capture
# counts the record cannot fill for itself).
#
# The measurement window is bounded by cudaProfilerStart/Stop inside the
# benchmark, and nsys is told to capture exactly that range -- so warmup, where
# capture is legal, is outside the counted region.
#
# usage: run_gm_cell.sh <ckpt> <k> <prompt-index> <proc-idx> [out-dir]
set -uo pipefail

CKPT=${1:?checkpoint}
K=${2:?k}
P=${3:-0}
R=${4:-0}
OUT=${5:-/data/openpi_lg/exp/libero_groot/data/latency}

REPO=/data/openpi_lg
G=/home/weiland/gr00t_n15
PY=/home/weiland/gr00t_n15_venv/.venv/bin/python
TR=/tmp/lg_traces
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH=$G:$G/examples/Libero:$REPO/src:$REPO
mkdir -p "$OUT" "$TR"
cd "$REPO" || exit 1

CELL="libero_cg_k${K}_p${P}_r${R}"
JSON="$OUT/$CELL.json"
if [ -f "$JSON" ] && grep -q '"certified": true' "$JSON"; then
  echo "[gm] $CELL already certified"; exit 0
fi

echo "[gm] === $CELL $(date -Is)"
nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=true --output "$TR/$CELL" \
  "$PY" exp/robocasa365/bench_groot_stages.py --mode measure --profile libero \
    --checkpoint "$CKPT" --k "$K" --prompt-index "$P" --proc-idx "$R" --seed "$R" \
    --out "$JSON"
rc=$?
[ $rc -ne 0 ] && { echo "[gm] MEASURE FAILED rc=$rc"; exit $rc; }

nsys stats --report cuda_api_sum,nvtx_pushpop_sum --format csv --force-export=true \
  "$TR/$CELL.nsys-rep" > "$TR/$CELL.trace.csv"
echo "[gm] trace rows: $(wc -l < "$TR/$CELL.trace.csv")"

"$PY" exp/robocasa365/bench_groot_stages.py --mode certify \
  --out "$JSON" --cuda-trace "$TR/$CELL.trace.csv"
echo "[gm] certify rc=$?  $CELL"
