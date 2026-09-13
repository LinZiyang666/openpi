#!/usr/bin/env bash
# The CP2 encoder unit cost E for one suite (plan §3.11), with CUDA-graph
# evidence -- the same three-step shape as run_gm_cell.sh: measure (writes an
# uncertified record), export the nsys trace, certify (reads the trace back and
# fills the launch / capture counts).
#
# usage: run_cp2_encoder_cost.sh <suite> <ckpt> [repo] [out-json]
set -uo pipefail

SUITE=${1:?suite}
CKPT=${2:?checkpoint}
REPO=${3:-/data/openpi_lg}
OUT=${4:-$REPO/exp/libero_groot/config/actioncache/cost_groot_cp2_encoded_$SUITE.json}

G=/home/weiland/gr00t_n15
PY=/home/weiland/gr00t_n15_venv/.venv/bin/python
TR=/tmp/lg_traces
COST=$REPO/exp/libero_groot/config/rit/cost_groot_libero_measured.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH=$G:$G/examples/Libero:$REPO/src:$REPO
mkdir -p "$(dirname "$OUT")" "$TR"
cd "$REPO" || exit 1
[ -f "$COST" ] || { echo "[E] missing teacher cost table $COST"; exit 1; }

CELL="cp2_encoder_$SUITE"
# Never reuse an existing record on the strength of its "certified" label: E is
# bound to the checkpoint / GPU / sampling by content, and the consumer
# (libs.groot_cost_record) re-checks that binding; a stale record is archived.
if [ -f "$OUT" ]; then
  ARCH="$OUT.superseded_$(date +%s)"
  echo "[E] archiving previous record -> $ARCH"
  mv "$OUT" "$ARCH"
fi

echo "[E] === $CELL $(date -Is)"
nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=true --output "$TR/$CELL" \
  "$PY" exp/libero_groot/bench_cp2_overhead_groot.py --mode encoder-cost \
    --suite "$SUITE" --checkpoint "$CKPT" --cost-record "$COST" --out "$OUT"
rc=$?
[ $rc -ne 0 ] && { echo "[E] MEASURE FAILED rc=$rc"; exit $rc; }

nsys stats --report cuda_api_sum,nvtx_pushpop_sum --format csv --force-export=true \
  "$TR/$CELL.nsys-rep" > "$TR/$CELL.trace.csv"
rc=$?
[ $rc -ne 0 ] && { echo "[E] TRACE EXPORT FAILED rc=$rc"; exit $rc; }
echo "[E] trace rows: $(wc -l < "$TR/$CELL.trace.csv")"

"$PY" exp/libero_groot/bench_cp2_overhead_groot.py --mode certify-encoder \
  --out "$OUT" --cuda-trace "$TR/$CELL.trace.csv"
rc=$?
echo "[E] certify rc=$rc  $CELL -> $OUT"
exit "$rc"
