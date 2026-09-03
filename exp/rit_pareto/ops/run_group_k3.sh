#!/bin/bash
# run_group_k3.sh <suite> <rit|gst> [extra run_gtp args]
# RIT-Pareto K=3 (FULL / WARM@0.3 / WARM@0.5, no gate) on the timan108 fleet: 48 workers over 3 A5000 -> weilandserver :23150.
set -uo pipefail
cd /scratch/zixuans8/openpi_dispatch
export PYTHONPATH=/scratch/zixuans8/openpi_dispatch/packages/openpi-client/src:/scratch/zixuans8/openpi_dispatch/src:/scratch/zixuans8/openpi_dispatch
export PATH=/scratch/zixuans8/dsp_bin:/usr/local/bin:/usr/bin:/bin
export LIBERO_CONFIG_PATH=/home/zixuans8/.libero
PY=/scratch/zixuans8/openpi/.venv/bin/python
suite=$1 rule=$2; shift 2
case "$rule" in rit|gst) ;; *) echo "rule must be rit|gst"; exit 2 ;; esac
OUT=/tmp/dsp_precheck/rit_pareto/${suite}_k3_${rule}
/usr/bin/mkdir -p "$OUT"
$PY -m exp.gate_threshold_pareto.run_gtp \
  --arm-matrix /tmp/dsp_shared/rit_pareto/$suite/k3/arm_matrix_${rule}.yaml \
  --phase eval \
  --task-suite "$suite" \
  --servers ziyanglin.com:23150 \
  --workers 48 \
  --trials 50 \
  --gpus 3 \
  --conda-env /scratch/zixuans8/libero_sim \
  --judge-type threshold \
  --eval-gate always_search \
  --warm-tiers 0.3,0.5 \
  --journal "$OUT/journal.jsonl" \
  --per-step-out "$OUT/per_step.jsonl" \
  --apool-record exp/ablation_study/cache_size/config/apool_${suite}.yaml \
  --apool-dir /scratch/zixuans8/openpi_dispatch/exp/common/data/db_init/libero/${suite}_apool \
  "$@"
echo "RIT_RUN_EXIT=$?"
