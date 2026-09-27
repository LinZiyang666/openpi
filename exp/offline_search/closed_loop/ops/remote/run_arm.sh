#!/bin/bash
# run_arm.sh <suite> <arm> <servers> <server-workers> <out-dir> [extra run_gtp args]
# timan107 side (pushed to /scratch/zixuans8/openpi_trace/os_cl/): run_gtp (driver + WorkerAgent in one process)
# against weilandserver servers; same recipe as exp/trace_dual/ops/run_group.sh. OSCL_EPISODES=<ep_idx,...> restricts
# every task to those episode indices (smoke), via run_gtp_subset.py.
set -uo pipefail
SUITE=${1:?suite}; ARM=${2:?arm}; SERVERS=${3:?servers}; SW=${4:?server-workers}; OUT=${5:?out}
shift 5
R=/scratch/zixuans8/openpi_trace
export HOME=/home/zixuans8
export LIBERO_CONFIG_PATH=$HOME/.libero
export PYTHONPATH=$R/packages/openpi-client/src:$R/src:$R
export PATH=/scratch/zixuans8/dsp_bin:/usr/local/bin:/usr/bin:/bin
PY=/scratch/zixuans8/openpi/.venv/bin/python
MANIFEST=${OSCL_MANIFEST:-}
REPLAN=${OSCL_REPLAN_STEPS:-}
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --manifest) MANIFEST=${2:?manifest}; shift 2 ;;
    --manifest=*) MANIFEST=${1#*=}; shift ;;
    --replan-steps) REPLAN=${2:?replan-steps}; shift 2 ;;
    --replan-steps=*) REPLAN=${1#*=}; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done
# run_gtp reads only arm/yaml/suite from the matrix. Translate the per-arm client
# settings here so the matrix and arms.json remain the same source of truth.
CLIENT=$($PY - "$R/os_cl/cfg/matrix_$ARM.yaml" <<'PY'
import sys, yaml
r = yaml.safe_load(open(sys.argv[1]))["arms"][0]
c = r.get("client_overrides", {})
print(c.get("replan_steps", r.get("replan_steps", 5)), c.get("resize_size", "default"))
PY
) || exit 2
read -r MATRIX_REPLAN MATRIX_RESIZE <<< "$CLIENT"
REPLAN=${REPLAN:-$MATRIX_REPLAN}
[[ "$REPLAN" =~ ^[1-9][0-9]*$ ]] || { echo "invalid replan_steps: $REPLAN" >&2; exit 2; }
if [ -n "$MANIFEST" ]; then
  [[ "$MANIFEST" = /* ]] || MANIFEST="$PWD/$MANIFEST"
  $PY "$R/os_cl/count.py" --manifest-info "$MANIFEST" >/dev/null || exit 2
  export OSCL_MANIFEST="$MANIFEST"
fi
# GR00T arms need the raw 256x256 render (the transform chain crops to 224). Decide from the arm name OR the arm's
# yaml (GR00T yamls carry the cp1_groot_* key builder) so arm naming cannot silently break it.
if [[ "$ARM" == *groot* ]] || grep -qi groot "$R/os_cl/cfg/$ARM.yaml" 2>/dev/null; then RES=(--resize-size 256); else RES=(); fi
[ "$MATRIX_RESIZE" != default ] && RES=(--resize-size "$MATRIX_RESIZE")
echo "RUN_ARM_RESIZE arm=$ARM res=${RES[*]:-none}"
W=0; for n in $(echo "$SW" | tr ',' ' '); do W=$((W+n)); done
if [ -n "${OSCL_EPISODES:-}$MANIFEST" ]; then ENTRY=("$R/os_cl/run_gtp_subset.py"); else ENTRY=(-m exp.gate_threshold_pareto.run_gtp); fi
mkdir -p "$OUT"
cd "$R"
echo "RUN_ARM_START $(date -Is) arm=$ARM suite=$SUITE servers=$SERVERS sw=$SW episodes=${OSCL_EPISODES:-all}"
$PY "${ENTRY[@]}" \
  --arm-matrix os_cl/cfg/matrix_${ARM}.yaml \
  --phase eval \
  --task-suite "$SUITE" \
  --servers "$SERVERS" \
  --server-workers "$SW" \
  --workers "$W" \
  --trials 50 \
  --gpus 8 \
  --conda-env /scratch/zixuans8/libero_sim \
  --judge-type threshold \
  --eval-gate always_search \
  "${RES[@]}" \
  --replan-steps "$REPLAN" \
  --journal "$OUT/journal.jsonl" \
  --per-step-out "$OUT/per_step.jsonl" \
  --apool-record exp/ablation_study/cache_size/config/apool_${SUITE}.yaml \
  --apool-dir $R/exp/common/data/db_init/libero/${SUITE}_apool \
  "${EXTRA_ARGS[@]}"
rc=$?
echo "RUN_ARM_EXIT=$rc $(date -Is)"
exit $rc
