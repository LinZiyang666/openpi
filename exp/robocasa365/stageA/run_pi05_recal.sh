#!/usr/bin/env bash
# pi0.5 same-card recalibration of the ledger's CUDA-Graph tier (3 repeats).
# Same recipe as data/pi05_compile_ro_3stage.json: 3-stage, reduce-overhead,
# mark_step, n=30. bench_teacher.py pins sys.path to /home/weiland/openpi,
# so the pi0.5 model code comes from the working repo (unchanged by 2e51b02).
set -u
PY=/home/weiland/openpi/.venv/bin/python
OUT=/tmp/openpi-stageA/exp/robocasa365/data/latency
LOG=/tmp/stageA/pi05_recal.log
mkdir -p "$OUT"; cd /home/weiland/openpi
echo "[pi05] start $(date -Is)" | tee -a "$LOG"
for r in 0 1 2; do
  echo "[pi05] === repeat $r $(date -Is)" | tee -a "$LOG"
  "$PY" exp/ablation_study/latency_bench/bench_teacher.py --mode reduce-overhead --mark-step \
    --warmup 10 --iters 30 --out "$OUT/pi05_compile_ro_3stage_recal_r$r.json" >> "$LOG" 2>&1
  echo "[pi05] rc=$?" | tee -a "$LOG"
  "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print('[pi05]  ', {k: round(v,2) if isinstance(v,float) else v for k,v in d.items() if k.endswith('_ms') or k in ('n','torch')})" "$OUT/pi05_compile_ro_3stage_recal_r$r.json" | tee -a "$LOG"
done
echo "[pi05] done $(date -Is)" | tee -a "$LOG"
