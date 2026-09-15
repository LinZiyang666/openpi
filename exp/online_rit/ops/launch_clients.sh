#!/usr/bin/env bash
# run_gtp driver for ONE arm matrix from a sim box. The matrix file names its
# cohort (pool / trials / judge_type / single_process); this launcher binds it
# explicitly and refuses a mismatch instead of guessing.
#
# usage: launch_clients.sh <suite> <arm-matrix.yaml> <servers> <workers> <gpus> <run-dir> \
#            <apool-record> <apool-dir> [init_pools_manifest.json] [extra run_gtp args]
#   A500 matrices:   apool = the official A pool record / dir, no manifest.
#   adapt matrix:    apool = data/<suite>/pools/apool_adapt.yaml + adapt_pool dir, manifest given.
# Online matrices always run with ONE server endpoint, --eval-concurrency 1, retries 0.
set -uo pipefail
SUITE=${1:?suite}; MATRIX=${2:?matrix}; SERVERS=${3:?servers}; WORKERS=${4:?workers}; GPUS=${5:?gpus}
RUN=${6:?run dir}; APOOL_REC=${7:?apool record}; APOOL_DIR=${8:?apool dir}; MANIFEST=${9:-}
shift 9 2>/dev/null || shift $#
R=${OPENPI_ROOT:-/scratch/zixuans8/openpi_lg}
export LIBERO_CONFIG_PATH=${SIM_HOME:-/home/zixuans8}/.libero
export PYTHONPATH=$R/packages/openpi-client/src:$R/src:$R
PY=${SIM_PY:-/scratch/zixuans8/openpi/.venv/bin/python}
mkdir -p "$RUN"
cd "$R"
read -r POOL TRIALS JUDGE SINGLE < <($PY - "$MATRIX" <<'PYEOF'
import sys, yaml
doc = yaml.safe_load(open(sys.argv[1]))
c = doc.get("cohort") or {}
if not c:
    sys.exit("matrix carries no cohort block; emit it with emit_online_arms")
print(c["pool"], c["trials"], c["judge_type"], "1" if c["single_process"] else "0")
PYEOF
) || exit 2
if [ "$POOL" != "a500" ] && [ -z "$MANIFEST" ]; then echo "subset cohort needs the init pool manifest"; exit 2; fi
if [ "$POOL" = "a500" ] && [ -n "$MANIFEST" ]; then echo "a500 cohort must not pass a subset manifest"; exit 2; fi
for arg in "$@"; do
  case "$arg" in
    --trials|--trials=*|--apool-*|--init-map*|--servers*|--task-suite*|--judge-type*|--arm-matrix*|--eval-concurrency*|--max-episode-retries*)
      echo "cohort argument cannot be overridden: $arg"; exit 2;;
  esac
done
EXTRA=()
if [ "$SINGLE" = "1" ]; then
  case "$SERVERS" in *,*) echo "online matrix must bind ONE server endpoint, got $SERVERS"; exit 2;; esac
  EXTRA+=(--eval-concurrency 1 --max-episode-retries 0)
fi
if [ -n "$MANIFEST" ]; then EXTRA+=(--init-map "$MANIFEST" --init-map-key "$POOL"); fi
if [ "$JUDGE" = "threshold" ]; then EXTRA+=(--warm-tiers 0.875,0.75,0.5); fi
$PY -m exp.gate_threshold_pareto.run_gtp \
  --arm-matrix "$MATRIX" --phase eval --task-suite "$SUITE" --servers "$SERVERS" \
  --workers "$WORKERS" --gpus "$GPUS" --trials "$TRIALS" --judge-type "$JUDGE" \
  --eval-gate score_hysteresis --resize-size 256 --replan-steps 5 \
  --conda-env ${SIM_CONDA:-/scratch/zixuans8/libero_sim} \
  --journal "$RUN/journal.jsonl" --per-step-out "$RUN/per_step.jsonl" \
  --apool-record "$APOOL_REC" --apool-dir "$APOOL_DIR" "$@" "${EXTRA[@]}"
status=$?
echo "ONLINE_RIT_EVAL_EXIT=$status"
exit $status
