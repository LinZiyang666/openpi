#!/bin/bash
# run_group_ws.sh <suite> <layer ng|hg> [extra run_gtp args]
# Failover client fleet ON weilandserver (timan107 unreachable 2026-09-01 ~02:50):
# 48 workers -> local 4-replica server 127.0.0.1:23150. Same arms, same A-pool, same trials.
set -uo pipefail
cd /data/openpi_dispatch
export HOME=/home/weiland
export PYTHONPATH=/data/openpi_dispatch/src:/data/openpi_dispatch
export PATH=/home/weiland/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
PY=/home/weiland/openpi/.venv/bin/python
suite=$1 layer=$2; shift 2
case "$layer" in
  ng) GATE=always_search ;;
  hg) GATE=score_hysteresis ;;
  *) echo "layer must be ng|hg"; exit 2 ;;
esac
OUT=/tmp/dsp_shared/rit_pareto/runs/${suite}_${layer}
mkdir -p "$OUT"
$PY -m exp.gate_threshold_pareto.run_gtp \
  --arm-matrix /tmp/dsp_shared/rit_pareto/$suite/arms/arm_matrix_${layer}.yaml \
  --phase eval \
  --task-suite "$suite" \
  --servers 127.0.0.1:23150 \
  --workers 48 \
  --trials 50 \
  --gpus 1 \
  --conda-env /home/weiland/libero_sim \
  --judge-type dispatch_surface \
  --eval-gate $GATE \
  --journal "$OUT/journal.jsonl" \
  --per-step-out "$OUT/per_step.jsonl" \
  --apool-record exp/ablation_study/cache_size/config/apool_${suite}.yaml \
  --apool-dir /home/weiland/openpi/exp/common/data/db_init/libero/${suite}_apool \
  "$@"
echo "RIT_RUN_EXIT=$?"
