#!/usr/bin/env bash
# W2: G-M measurement cells on weilandserver, strictly serial, resumable.
# Each cell = nsys profile (capture range = measurement window) -> raw JSON
#             -> nsys stats CSV -> --mode certify (fills launch/capture counts).
# Usage: run_w2.sh <prompt-indices e.g. "0 1 2 3 4"> <ks e.g. "1 2 3 4"> <procs e.g. "0 1 2">
set -u
ROOT=/tmp/openpi-stageA
PY=/home/weiland/gr00t_n15_venv/.venv/bin/python
CKPT=/home/weiland/ckpt_n15_robocasa_tp/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000
OUT=$ROOT/exp/robocasa365/data/latency
TR=/tmp/w2_traces
LOG=/tmp/stageA/w2.log
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH=/home/weiland/gr00t_n15:$ROOT/src:$ROOT
mkdir -p "$OUT" "$TR"
cd "$ROOT"
PROMPTS=${1:-"0"}; KS=${2:-"1 2 3 4"}; PROCS=${3:-"0 1 2"}
echo "[w2] start $(date -Is) prompts=[$PROMPTS] ks=[$KS] procs=[$PROCS]" | tee -a "$LOG"
for p in $PROMPTS; do for k in $KS; do for r in $PROCS; do
  cell="groot_cg_k${k}_p${p}_r${r}"
  json="$OUT/$cell.json"
  if [ -f "$json" ] && grep -q '"certified": true' "$json"; then
    echo "[w2] skip $cell (certified)" | tee -a "$LOG"; continue
  fi
  echo "[w2] === $cell $(date -Is)" | tee -a "$LOG"
  t0=$(date +%s)
  nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite=true --output "$TR/$cell" \
    "$PY" exp/robocasa365/bench_groot_stages.py --mode measure --checkpoint "$CKPT" \
      --k "$k" --prompt-index "$p" --proc-idx "$r" --seed "$r" --out "$json" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "[w2] MEASURE FAILED rc=$rc $cell" | tee -a "$LOG"; continue; fi
  nsys stats --report cuda_api_sum,nvtx_pushpop_sum --format csv --force-export=true \
    "$TR/$cell.nsys-rep" > "$TR/$cell.trace.csv" 2>> "$LOG"
  "$PY" exp/robocasa365/bench_groot_stages.py --mode certify --out "$json" \
    --cuda-trace "$TR/$cell.trace.csv" >> "$LOG" 2>&1
  echo "[w2] certify rc=$? $cell elapsed=$(( $(date +%s) - t0 ))s" | tee -a "$LOG"
  "$PY" - "$json" <<'PYEOF' 2>>"$LOG" | tee -a "$LOG"
import json, sys, statistics as st
d = json.load(open(sys.argv[1]))
med = lambda k: st.median(d[k]) if d.get(k) else float("nan")
print(f"[w2]   valid={d.get('valid')} launches={d.get('cudagraph_launch_count')}/{d.get('expected_cudagraph_launch_count')} "
      f"s1={med('stage1_ms'):.2f} s2={med('stage2_llm_ms'):.2f} s3={med('stage3_ms'):.2f} tot={med('total_ms'):.2f} ms  "
      f"void={d.get('void_reasons')}")
PYEOF
done; done; done
echo "[w2] done $(date -Is)" | tee -a "$LOG"
