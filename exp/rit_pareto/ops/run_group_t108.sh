#!/bin/bash
# run_group.sh <suite> <layer ng|hg> [extra run_gtp args]
# RIT-Pareto eval group on the timan107 fleet: 48 workers -> weilandserver :23150 (4 replicas).
set -uo pipefail
cd /scratch/zixuans8/openpi_dispatch
export PYTHONPATH=/scratch/zixuans8/openpi_dispatch/src:/scratch/zixuans8/openpi_dispatch
export PATH=/scratch/zixuans8/dsp_bin:/usr/local/bin:/usr/bin:/bin
export LIBERO_CONFIG_PATH=/home/zixuans8/.libero
PY=/scratch/zixuans8/openpi/.venv/bin/python
suite=$1 layer=$2; shift 2
case "$layer" in
  ng) GATE=always_search ;;
  hg) GATE=score_hysteresis ;;
  *) echo "layer must be ng|hg"; exit 2 ;;
esac
OUT=/tmp/dsp_precheck/rit_pareto/${suite}_${layer}
/usr/bin/mkdir -p "$OUT"
$PY -m exp.gate_threshold_pareto.run_gtp \
  --arm-matrix /tmp/dsp_shared/rit_pareto/$suite/arms/arm_matrix_${layer}.yaml \
  --phase eval \
  --task-suite "$suite" \
  --servers ziyanglin.com:23150 \
  --workers 48 \
  --trials 50 \
  --gpus 3 \
  --conda-env /scratch/zixuans8/libero_sim \
  --judge-type dispatch_surface \
  --eval-gate $GATE \
  --journal "$OUT/journal.jsonl" \
  --per-step-out "$OUT/per_step.jsonl" \
  --apool-record exp/ablation_study/cache_size/config/apool_${suite}.yaml \
  --apool-dir /scratch/zixuans8/openpi/exp/common/data/db_init/libero/${suite}_apool \
  "$@"
echo "RIT_RUN_EXIT=$?"
