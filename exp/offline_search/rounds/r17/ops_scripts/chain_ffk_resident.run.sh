#!/bin/bash
# chain_ffk_resident: chain_ffk_stop2.run.sh for R17 filter_failed_knob campaigns that reuse one resident
# (seeds_1023/resident_attach.py). Item = J, J+ or WAIT:<regex>.
#   J   launch with --attach-resident: serve from the lane's kept resident when model, K, runtime, ports, replicas,
#       batching, MPS and bootstrap match (otherwise the launcher stops it and J starts its own); J closes it at the end.
#   J+  the same, plus --keep-resident: after a verified complete driver the resident stays up for the next item.
# Mark J+ when the next item serves the same model and K on the same lane; the last item of such a run has no +.
# Prepare: prepare_<J>.sh if present, else filter_failed.stage --run. launch_<J>.sh, if present, must be the plain
# filter_failed_knob.launch (R15/R16 units are not resident units). Lane env (R17_H100_NODE, R17_H100_HOST,
# R17_DRIVER_PORT[, R17_PUBLIC_PORT]) goes on this process, as for stop2.
# Any exit (queue done, hold file, failure, TERM/INT/HUP) stops a resident still kept for a next item
# (resident_attach stop --run), so no GPU resident outlives the chain. A failed item stops the chain as in stop2.
# The 4090 gate is skipped while the previous item keeps a resident for this one.
TAG=$1; WAIT=$2; shift 2
M=/home/weiland/projects/openpi_ext/scratch/r17_ops/mw; D=/home/weiland/projects/openpi_ext/scratch/r17_ops/ff
REPO=/home/weiland/projects/openpi; PY=$REPO/.venv/bin/python; E=/data/mw_finetune/eval
C=$REPO/exp/offline_search/rounds/r17/filter_failed_knob/commands
KEPT=
log(){ echo "$(date +%H:%M:%S) [ffk-$TAG] $*" >> $M/chain.log; }
field(){ python3 -c "import json;r=json.load(open('$E/$1/RECEIPT.json'));print($2)" 2>/dev/null; }
release(){ [ -n "$1" ] || return 0
  (cd $REPO && $PY -m exp.offline_search.rounds.r17.seeds_1023.resident_attach stop --run $E/$1) >> $D/$1.resident_stop.log 2>&1
  log "resident served to $1: stop rc=$? ($(tail -c 300 $D/$1.resident_stop.log | tr '\n' ' '))"; }
trap 'release $KEPT' EXIT
trap 'exit 143' TERM INT HUP
stop(){ log "STOPPED after failure: $1 rc=$2"; touch $M/hold_ff_$TAG; release $1; exit 1; }
until grep -qE "$WAIT" $M/chain.log; do sleep 30; done
log "queue start: $*${R17_H100_NODE:+ (h100 alias $R17_H100_NODE $R17_H100_HOST, driver port $R17_DRIVER_PORT${R17_PUBLIC_PORT:+, public port $R17_PUBLIC_PORT})}"
for I in "$@"; do
  [ -f $M/hold_ff_$TAG ] && { log "hold file present, stop before $I"; break; }
  case $I in WAIT:*) log "waiting for ${I#WAIT:}"; until grep -qE "${I#WAIT:}" $M/chain.log; do sleep 30; done; continue;; esac
  J=${I%+}; FLAGS=--attach-resident; [ "$J" = "$I" ] || FLAGS="$FLAGS --keep-resident"
  [ ! -f $C/launch_$J.sh ] || grep -q 'filter_failed_knob.launch --run' $C/launch_$J.sh || { log "$J is not an R17 filter_failed_knob unit"; stop $J 2; }
  case $TAG in 4090*) [ -n "$KEPT" ] || until [ "$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')" -lt 2000 ]; do sleep 20; done;; esac
  log "prepare $J"
  if [ -f $C/prepare_$J.sh ]; then bash $C/prepare_$J.sh; else (cd $REPO && $PY -m exp.offline_search.rounds.r17.filter_failed.stage --run $E/$J); fi > $D/$J.prep.log 2>&1; rc=$?
  [ $rc -eq 0 ] || { log "$J prepare FAILED rc=$rc"; stop $J $rc; }
  grep -qx "SKIP_OFF_SUFFICIENT $J" $D/$J.prep.log && { log "$J skipped: OFF sufficient, no levels job"; continue; }
  log "start $J $FLAGS"
  (cd $REPO && $PY -m exp.offline_search.rounds.r17.filter_failed_knob.launch --run $E/$J $FLAGS) > $D/$J.run.log 2>&1; rc=$?
  R=$(field $J "r.get('status')"); w=0
  while [ $rc -eq 0 ] && [ "$R" = PENDING_EVIDENCE ] && [ $w -lt 8000 ]; do sleep 20; w=$((w+20)); R=$(field $J "r.get('status')"); done
  ROLE=$(field $J "(r.get('resident') or {}).get('role')"); NEXT=
  [ "$(field $J "bool((r.get('resident') or {}).get('kept'))")" = True ] && NEXT=$J
  log "$J ended rc=$rc accepted=$(grep -c '"accepted": true' $E/$J/driver/journal.jsonl 2>/dev/null || echo -) receipt=$R resident=$ROLE${NEXT:+ kept}"
  [ $rc -eq 0 ] && [ "$R" = PASS ] || stop $J $rc
  KEPT=$NEXT
  [ -n "$KEPT" ] || sleep 60
done
log "queue done"
