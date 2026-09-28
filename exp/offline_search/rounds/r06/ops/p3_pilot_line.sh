#!/bin/bash
# P3 v2 pilot lane: p3_pilot_line.sh <label> <port> <server-cpus>
# Claims arms from $RUN/pilot_queue.txt in order (atomic mkdir claim), runs each through a frozen copy of chain_p3.sh
# (file mode, one arm per invocation), then verifies the collect (local sha == remote sha) and deletes that arm's
# remote telemetry directory and /tmp tar on timan107 so /scratch never fills. GPU-aware wait before every arm.
set -u
LABEL=${1:?label}; PORT=${2:?port}; SCPUS=${3:?server cpus}
R=/home/weiland/projects/openpi
RUN=/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot
TAG=$(basename "$RUN")
CHAIN=${CHAIN:-$RUN/ops/chain_p3.frozen.sh}
Q=$RUN/pilot_queue.txt
LOG=$RUN/lane_$LABEL.log
WPS=${WPS:-20}
STREAM_ENV=${STREAM_ENV:-}
say() { echo "$(date +%m-%d_%H:%M:%S) [$LABEL] $*" | tee -a "$LOG"; }
rx() { timeout 300 tether exec timan107 -- bash -c "$1" 2>/dev/null; }

need_for() { case "$1" in *_pi05_*) echo 10500;; *) echo 9500;; esac; }

source /home/weiland/.claude/jobs/a607dd74/tmp/gpu_gate.sh
wait_gpu() {  # $1 need MiB; budget gate (tmp/gpu_budget_mb), sustained 3 checks x 20 s
  local need=$1 ok=0
  while true; do
    [ -e "$RUN/STOP_$LABEL" ] && { say "STOP file"; exit 0; }
    if admit "$need" 0; then ok=$((ok+1)); else ok=0; fi
    [ $ok -ge 3 ] && return 0
    sleep 20
  done
}

orphan_cleanup() {  # kill our own leftover server on PORT (only pilot arms' os-tag)
  local pid
  pid=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1)
  if [ -n "$pid" ] && ps -o args= -p "$pid" | grep -q "os-tag r6p3v2_[a-z0-9_]*_$PORT"; then
    say "orphan server pid=$pid on $PORT -> kill"; kill "$pid"; sleep 5
    tmux kill-session -t "oscl$PORT" 2>/dev/null
  fi
}

remote_cleanup() {  # $1 arm; only after local == remote sha
  local arm=$1 j=$RUN/runs/$1/client_telemetry_collect.json ok
  ok=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(1 if d.get('files',0)>0 and d.get('local_sha256') and d['local_sha256']==d.get('remote_sha256') else 0)" "$j" 2>/dev/null)
  if [ "$ok" != 1 ]; then say "CLEANUP_SKIPPED arm=$arm (collect record missing or sha mismatch)"; return 1; fi
  local rdir=/scratch/zixuans8/openpi_trace/os_cl/p3_client/$TAG/$arm tarp=/tmp/p3_telemetry_${TAG}_$arm.tar
  if rx "set -e; if [ -d '$rdir' ]; then rm -r '$rdir'; fi; if [ -e '$tarp' ]; then rm '$tarp'; fi; echo CLEAN_OK" | grep -q CLEAN_OK; then
    say "REMOTE_CLEANED arm=$arm"; touch "$RUN/state/$arm.CLEANED"
  else
    say "REMOTE_CLEANUP_FAILED arm=$arm"
  fi
}

say "LANE_START port=$PORT cpus=$SCPUS wps=$WPS chain=$CHAIN stream='${STREAM_ENV}'"
cd "$R" || exit 1
while true; do
  [ -e "$RUN/STOP_$LABEL" ] && { say "STOP file"; break; }
  arm=""
  while read -r a; do
    [ -z "$a" ] && continue
    if mkdir "$RUN/state/claim_$a" 2>/dev/null; then arm=$a; break; fi
  done < "$Q"
  [ -z "$arm" ] && { say "QUEUE_EMPTY"; break; }
  echo "$LABEL $(date -Is)" > "$RUN/state/claim_$arm/by"
  tries=0
  while true; do
    tries=$((tries+1))
    if [ -f "$RUN/state/$arm.DONE" ] || ls "$RUN/state/$arm".manifest_*.DONE >/dev/null 2>&1; then break; fi
    wait_gpu "$(need_for "$arm")"
    orphan_cleanup
    rm -f "$RUN/state/$arm.ERROR"
    say "ARM_RUN arm=$arm try=$tries"
    env $STREAM_ENV PORTS=$PORT WPS=$WPS SERVER_CPUS=$SCPUS STAGE1_ONLY=0 P3_PHASE=pilot \
      bash "$CHAIN" "$RUN" "$arm" >> "$RUN/chain_console_$LABEL.log" 2>&1
    rc=$?
    if ls "$RUN/state/$arm".manifest_*.DONE >/dev/null 2>&1 || [ -f "$RUN/state/$arm.DONE" ]; then break; fi
    say "ARM_STOPPED arm=$arm rc=$rc try=$tries last=$(grep -E '^EV' "$RUN/runs/chain.log" | grep "arm=$arm" | tail -1 | cut -c1-160)"
    orphan_cleanup
    if [ $tries -ge 12 ]; then say "ARM_GAVE_UP arm=$arm"; touch "$RUN/state/$arm.GAVEUP"; break; fi
    sleep 90
  done
  if ! [ -e "$RUN/state/$arm.GAVEUP" ]; then
    say "ARM_OK arm=$arm"
    [ -n "$STREAM_ENV" ] || remote_cleanup "$arm"
  fi
done
say "LANE_END"
