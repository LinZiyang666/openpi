#!/bin/bash
# run_group_k3.sh <suite> <rit|gst|gsth> [extra run_gtp args]
# RIT-Pareto K=3 (FULL / WARM@0.3 / WARM@0.5) on a timan fleet: 48 workers -> weilandserver :23150 (4 replicas).
#   rit / gst  : no gate (always_search)          -- 2026-09-01/02 groups
#   gsth       : GST cuts behind the H gate (score_hysteresis, K=2 gate theta) -- 2026-09-03 groups
#   rith       : RIT cuts behind the same H gate                                -- 2026-09-05 groups
# Host layout: timan108 = /scratch/zixuans8/openpi_dispatch (3 A5000); timan107 = /scratch/zixuans8/openpi_dispatch_k3 (8 GTX1080).
set -uo pipefail
case "$(hostname -s)" in
  timan107) DISPATCH=/scratch/zixuans8/openpi_dispatch_k3; GPUS=8
            APOOL_ROOT=/scratch/zixuans8/openpi/exp/common/data/db_init/libero ;;
  *)        DISPATCH=/scratch/zixuans8/openpi_dispatch; GPUS=3
            APOOL_ROOT=/scratch/zixuans8/openpi_dispatch/exp/common/data/db_init/libero ;;
esac
cd "$DISPATCH"
export PYTHONPATH=$DISPATCH/packages/openpi-client/src:$DISPATCH/src:$DISPATCH
export PATH=/scratch/zixuans8/dsp_bin:/usr/local/bin:/usr/bin:/bin
export LIBERO_CONFIG_PATH=/home/zixuans8/.libero
# HOME must be the NFS home: libero resolves its asset cache under ~/.cache/libero/assets, and a tmux
# server created through the tether agent inherits HOME=/srv/local/<user>/tether-home, whose partial
# asset copy lacks the libero_10 scenes (2026-09-04 g7 false start). timan108's long-lived tmux server
# already carried the NFS HOME, which is why the earlier l10 groups never hit this.
export HOME=/home/zixuans8
PY=/scratch/zixuans8/openpi/.venv/bin/python
suite=$1 rule=$2; shift 2
case "$rule" in
  rit|gst) GATE=always_search ;;
  gsth|rith) GATE=score_hysteresis ;;
  *) echo "rule must be rit|gst|gsth|rith"; exit 2 ;;
esac
OUT=/tmp/dsp_precheck/rit_pareto/${suite}_k3_${rule}
/usr/bin/mkdir -p "$OUT"
$PY -m exp.gate_threshold_pareto.run_gtp \
  --arm-matrix /tmp/dsp_shared/rit_pareto/$suite/k3/arm_matrix_${rule}.yaml \
  --phase eval \
  --task-suite "$suite" \
  --servers ziyanglin.com:23150 \
  --workers 48 \
  --trials 50 \
  --gpus $GPUS \
  --conda-env /scratch/zixuans8/libero_sim \
  --judge-type threshold \
  --eval-gate $GATE \
  --warm-tiers 0.3,0.5 \
  --journal "$OUT/journal.jsonl" \
  --per-step-out "$OUT/per_step.jsonl" \
  --apool-record exp/ablation_study/cache_size/config/apool_${suite}.yaml \
  --apool-dir $APOOL_ROOT/${suite}_apool \
  "$@"
echo "RIT_RUN_EXIT=$?"
