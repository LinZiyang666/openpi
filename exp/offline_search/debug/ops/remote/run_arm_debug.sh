#!/bin/bash
# run_arm_debug.sh (coordinator only; derived from stock run_arm.sh) <suite> <arm> <servers> <server-workers> <out-dir> [extra run_gtp args]
# timan107 side (pushed to /scratch/zixuans8/openpi_trace/os_cl/): run_gtp (driver + WorkerAgent in one process)
# against weilandserver servers; same recipe as exp/trace_dual/ops/run_group.sh. OSCL_EPISODES=<ep_idx,...> restricts
# every task to those episode indices (smoke), via run_gtp_subset.py.
set -uo pipefail
SUITE=${1:?suite}; ARM=${2:?arm}; SERVERS=${3:?servers}; SW=${4:?server-workers}; OUT=${5:?out}
shift 5
R=/scratch/zixuans8/openpi_trace
export LIBERO_CONFIG_PATH=/home/zixuans8/.libero
export PYTHONPATH=$R/packages/openpi-client/src:$R/src:$R
export PATH=/scratch/zixuans8/dsp_bin:/usr/local/bin:/usr/bin:/bin
PY=/scratch/zixuans8/openpi/.venv/bin/python
if [ -n "${OSDEBUG_ENV_FILE:-}" ]; then
  # Parse before exporting: no eval or shell interpolation of tokens/config.
  ENV_ROWS=$($PY - "$OSDEBUG_ENV_FILE" <<'PY'
import json, re, sys
env = json.load(open(sys.argv[1]))
for k, v in env.items():
    if not re.fullmatch(r'OSDEBUG_[A-Z0-9_]+', k) or not isinstance(v, str) or '\0' in v:
        raise ValueError('invalid debug client environment')
print(json.dumps(env))
PY
  ) || exit 2
  while IFS= read -r -d '' entry; do export "$entry"; done < <($PY -c 'import json,sys; sys.stdout.write("".join(k+"="+v+"\0" for k,v in json.loads(sys.argv[1]).items()))' "$ENV_ROWS")
fi
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
ENTRY=(-m exp.offline_search.debug.client.driver)
export OSDEBUG_CLIENT_DIR=${OSDEBUG_CLIENT_DIR:?client capture directory}
export OSDEBUG_CAMPAIGN=${OSDEBUG_CAMPAIGN:?campaign}
export OSDEBUG_ARM=$ARM
if [ "${OSDEBUG_MODE:-stream}" = stream ]; then
  export OSDEBUG_STREAM=${OSDEBUG_STREAM:?receiver address}
  export OSDEBUG_STREAM_TOKEN=${OSDEBUG_STREAM_TOKEN:?receiver token}
fi
POOL_ARGS=()
if [ -n "${OSDEBUG_APOOL_RECORD:-}" ]; then
  # Non-test smoke: the hash-bound B-val pool replaces the official test pool (argparse: last value wins).
  test -f "$OSDEBUG_APOOL_RECORD" && test -d "${OSDEBUG_APOOL_DIR:?pool dir}" || { echo "bad B-val pool" >&2; exit 2; }
  for f in "$OSDEBUG_APOOL_DIR"/*.pruned_init; do if [ -e "$f" ]; then echo "B-val pool shadowed by $f" >&2; exit 2; fi; done
  [ -n "${OSCL_MANIFEST:-}" ] || { echo "exact B-val manifest required" >&2; exit 2; }
  POOL_ARGS=(--apool-record "$OSDEBUG_APOOL_RECORD" --apool-dir "$OSDEBUG_APOOL_DIR")
  echo "OSDEBUG_BVAL record=$OSDEBUG_APOOL_RECORD pool=$OSDEBUG_APOOL_DIR manifest=$OSCL_MANIFEST"
fi
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
  "${POOL_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
rc=$?
echo "RUN_ARM_EXIT=$rc $(date -Is)"
exit $rc
