#!/bin/bash
# Claim-based frontier lane: frontier_lane.sh <label> <ports> <server-cpus> <need-MiB>
# Claims arms from r06_frontier/frontier_queue.txt (atomic mkdir state/fclaim_<arm>), one stock-chain run per arm,
# budget-gated; done = <arm>.DONE or <arm>.manifest_*.DONE (checked separately).
set -u
LABEL=${1:?}; PORTS=${2:?}; SCPUS=${3:?}; NEED=${4:?}
source /home/weiland/.claude/jobs/a607dd74/tmp/gpu_gate.sh
R=/home/weiland/trace_runs/os_closed_loop/r06_frontier; Q=$R/frontier_queue.txt; LOG=$R/lane_$LABEL.log
say() { echo "$(date +%m-%d_%H:%M:%S) [$LABEL] $*" | tee -a "$LOG"; }
is_done() { [ -e "$R/state/$1.DONE" ] || compgen -G "$R/state/$1.manifest_*.DONE" >/dev/null; }
cd /home/weiland/projects/openpi; unset OSCL_MANIFEST OSCL_EPISODES OSCL_TASKS
say "LANE_START ports=$PORTS cpus=$SCPUS need=$NEED"
while true; do
  arm=""
  while read -r a; do [ -z "$a" ] && continue; is_done "$a" && continue; mkdir "$R/state/fclaim_$a" 2>/dev/null && { arm=$a; break; }; done < "$Q"
  [ -z "$arm" ] && { say "QUEUE_EMPTY"; break; }
  for try in $(seq 1 12); do
    is_done "$arm" && break
    ok=0; while [ $ok -lt 3 ]; do if admit "$NEED" "$(pilot_reserve_mb)"; then ok=$((ok+1)); else ok=0; fi; sleep 30; done
    [ -e "$R/state/$arm.ERROR" ] && rm "$R/state/$arm.ERROR"
    for p in ${PORTS//,/ }; do pid=$(ss -ltnp 2>/dev/null | grep ":$p " | grep -oP 'pid=\K[0-9]+' | head -1); [ -n "$pid" ] && ps -o args= -p $pid | grep -q "os-tag r6q2_[a-z0-9_]*_$p" && kill $pid && tmux kill-session -t oscl$p 2>/dev/null; done
    say "ARM_RUN $arm try=$try"
    PORTS=$PORTS WPS=24 SERVER_CPUS=$SCPUS taskset -c 34-37,78-81 bash exp/offline_search/closed_loop/ops/chain.sh $R $arm >> $R/chain_console_$LABEL.log 2>&1
    is_done "$arm" && { say "ARM_OK $arm"; break; }
    say "ARM_STOPPED $arm try=$try"; sleep 90
  done
done
say "LANE_END"
