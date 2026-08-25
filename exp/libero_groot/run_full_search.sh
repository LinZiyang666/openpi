#!/usr/bin/env bash
# Drive the remaining search unattended: finish spatial's fine round, then run
# libero_10 coarse -> derive its fine region from its own marginals -> fine.
#
# Runs on weilandserver in tmux, so it outlives the operator's session and the
# WSL box. Every stage is resumable: the scheduler skips cells whose complete
# results already exist, so a stage that is re-entered costs only the cells in
# flight when it died.
#
# The L1 watchdog is stopped across a stage transition and re-armed after the
# new scheduler is up. Otherwise both would race to launch: the watchdog reads
# the same state file and would start a second scheduler on the same six ports
# in the seconds between the file being updated and this script launching.
set -u
export HOME=/home/weiland
REPO=/home/weiland/openpi
PY=$REPO/.venv/bin/python
export PYTHONPATH=$REPO:$REPO/src
SEARCH=/data/libero_cache/search
STATE=$SEARCH/current_search.env
PORTS=23160,23161,23162,23163,23164,23165
# libero_10 workers hold ~3.1 GB each (longer horizon, larger scenes) vs
# spatial's 2.6 GB: 12 per slot took timan107 to 4 GB free, one balloon away
# from an OOM cascade that silently drops whole shards. 8 per slot leaves
# ~66 GB of headroom on a machine shared with dozens of users.
WORKERS=8
LOG=/tmp/libsearch.log

say() { echo "[$(date '+%m-%d %H:%M:%S')] CHAIN $*"; }

stop_watchdog() { tmux kill-session -t libwatch 2>/dev/null; sleep 2; }
start_watchdog() { tmux new -s libwatch -d "bash $REPO/exp/libero_groot/search_watchdog.sh 2>&1 | tee -a /tmp/libwatch.log"; }

complete_cells() { ls "$1"/*.json 2>/dev/null | grep -vc partial; }

wait_for_stage() {  # $1=results dir  $2=total
  while true; do
    sleep 120
    tmux has-session -t libsearch 2>/dev/null || break
  done
  local n; n=$(complete_cells "$1")
  say "stage scheduler exited at $n/$2"
  # One resume attempt: a stage can end with cells missing if a collection
  # failed. Re-entering costs only what is missing.
  if [ "$n" -lt "$2" ]; then
    say "resuming to fill $(( $2 - n )) missing cell(s)"
    tmux new -s libsearch -d "cd $REPO && PYTHONPATH=$PYTHONPATH $PY exp/libero_groot/orchestrate_search.py --yaml-dir $3 --results-dir $1 --suite $4 --checkpoint $5 --ports $PORTS --workers $WORKERS 2>&1 | tee -a $LOG"
    while true; do sleep 120; tmux has-session -t libsearch 2>/dev/null || break; done
    n=$(complete_cells "$1")
    say "after resume: $n/$2"
  fi
}

run_stage() {  # $1=yaml dir $2=results dir $3=total $4=suite $5=ckpt $6=label
  say "=== $6: $3 cells ==="
  stop_watchdog
  cat > "$STATE" <<EOF
SUITE=$4
CKPT=$5
YAML_DIR=$1
RESULTS=$2
PORTS=$PORTS
WORKERS=$WORKERS
TOTAL=$3
LOGFILE=$LOG
EOF
  : > "$LOG"
  tmux new -s libsearch -d "cd $REPO && PYTHONPATH=$PYTHONPATH $PY exp/libero_groot/orchestrate_search.py --yaml-dir $1 --results-dir $2 --suite $4 --checkpoint $5 --ports $PORTS --workers $WORKERS 2>&1 | tee -a $LOG"
  sleep 60
  start_watchdog
  wait_for_stage "$2" "$3" "$1" "$4" "$5"
  stop_watchdog
}

analyze() {  # $1=results $2=steps $3=summary out $4=label
  say "--- analysis: $4 ---"
  cd "$REPO" && $PY exp/libero_groot/analyze_search.py "$1" --steps "$2" --top 12 --json-out "$3" 2>&1 \
    | grep -vE "pynvml|FutureWarning|^  import"
}

# ---------------------------------------------------------------- spatial fine
SP_CKPT=/home/weiland/ckpt_n15_libero_spatial
SP_R2=$SEARCH/libero_spatial/r2
SP_R2_RES=$SEARCH/libero_spatial/r2_results
if tmux has-session -t libsearch 2>/dev/null; then
  # Adopt a stage that is already in flight rather than restarting it.
  say "spatial fine already running; adopting"
  wait_for_stage "$SP_R2_RES" 26 "$SP_R2" libero_spatial "$SP_CKPT"
  stop_watchdog
else
  # Not running is NOT the same as finished: on a chain restart the stage has
  # to be *started*, and the earlier version silently fell through to the next
  # suite with the round only partly done.
  done_n=$(complete_cells "$SP_R2_RES")
  if [ "${done_n:-0}" -lt 26 ]; then
    say "spatial fine at ${done_n:-0}/26 and not running -- starting it"
    run_stage "$SP_R2" "$SP_R2_RES" 26 libero_spatial "$SP_CKPT" "spatial fine"
  else
    say "spatial fine already complete ($done_n/26)"
  fi
fi
analyze "$SP_R2_RES" 12 "$SEARCH/libero_spatial/r2_summary.json" "spatial fine"

# ---------------------------------------------------------------- l10 coarse
L10_CKPT=/home/weiland/ckpt_n15_libero_10
L10_LIB=/data/libero_cache/libraries/libero_10/libero_10_sp16_S3.pkl
L10_CALIB=/data/libero_cache/calib_input_S3/libero_10_calibration.json
for _ in $(seq 1 60); do [ -s "$L10_CALIB" ] && break; say "waiting for l10 calibration"; sleep 60; done
[ -s "$L10_CALIB" ] || { say "ABORT: l10 calibration missing"; exit 1; }

rm -rf "$SEARCH/libero_10/r1"
cd "$REPO" && $PY exp/libero_groot/emit_search_yamls.py --calibration "$L10_CALIB" \
  --stem libero_10_sp16_S3 --preload "$L10_LIB" \
  --out-dir "$SEARCH/libero_10/r1" --steps 6 2>&1 | grep -vE "pynvml|FutureWarning|^  import"
run_stage "$SEARCH/libero_10/r1" "$SEARCH/libero_10/r1_results" 28 libero_10 "$L10_CKPT" "l10 coarse"
analyze "$SEARCH/libero_10/r1_results" 6 "$SEARCH/libero_10/r1_summary.json" "l10 coarse"

# ---------------------------------------------------------------- l10 fine
REGION=$(cd "$REPO" && $PY exp/libero_groot/derive_fine_region.py "$SEARCH/libero_10/r1_summary.json" 2>&1 | grep -E "^(peaks|FIELD_)")
say "l10 fine region: $(echo "$REGION" | tr '\n' ' ')"
FMIN=$(echo "$REGION" | grep FIELD_MIN | cut -d= -f2)
FMAX=$(echo "$REGION" | grep FIELD_MAX | cut -d= -f2)
rm -rf "$SEARCH/libero_10/r2"
cd "$REPO" && $PY exp/libero_groot/emit_search_yamls.py --calibration "$L10_CALIB" \
  --stem libero_10_sp16_S3 --preload "$L10_LIB" \
  --out-dir "$SEARCH/libero_10/r2" --steps 12 --field-min "$FMIN" --field-max "$FMAX" 2>&1 \
  | grep -vE "pynvml|FutureWarning|^  import"
N2=$(ls "$SEARCH/libero_10/r2"/*.yaml | wc -l)
run_stage "$SEARCH/libero_10/r2" "$SEARCH/libero_10/r2_results" "$N2" libero_10 "$L10_CKPT" "l10 fine"
analyze "$SEARCH/libero_10/r2_results" 12 "$SEARCH/libero_10/r2_summary.json" "l10 fine"

say "FULL-SEARCH-DONE"
