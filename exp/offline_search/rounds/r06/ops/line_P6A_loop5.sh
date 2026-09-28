#!/bin/bash
# Line P6A (full model, 2 servers) with budget gate (tmp/gpu_gate.sh): ours+19000+pilot_reserve <= budget.
R=/home/weiland/trace_runs/os_closed_loop/r06_paper
source /home/weiland/.claude/jobs/a607dd74/tmp/gpu_gate.sh
for attempt in $(seq 1 50); do
  ok=0
  while true; do
    if admit 19000 "$(pilot_reserve_mb)"; then ok=$((ok+1)); else ok=0; fi
    [ $ok -ge 3 ] && break
    sleep 60
  done
  echo "ATTEMPT $attempt GPU_OK $(date -Is) $GATE_MSG"
  for RR in $R /home/weiland/trace_runs/os_closed_loop/r06_abl; do for f in $RR/state/*.ERROR; do [ -e "$f" ] && rm "$f"; done; done
  for p in 23150 23151; do pid=$(ss -ltnp 2>/dev/null | grep ":$p " | grep -oP 'pid=\K[0-9]+' | head -1); [ -n "$pid" ] && ps -o args= -p $pid | grep -q 'os-tag r[0-9a-z_]*_2315[01]' && kill $pid; done
  bash /home/weiland/.claude/jobs/a607dd74/tmp/line_P6A_v2.sh
  if ! ls $R/state/*.ERROR /home/weiland/trace_runs/os_closed_loop/r06_abl/state/*.ERROR >/dev/null 2>&1 && [ -e /home/weiland/trace_runs/os_closed_loop/r06_abl/state/r6p2_no_progress_g_sp_50.DONE ]; then echo "LOOP_DONE $(date -Is)"; break; fi
  echo "CHAIN_STOPPED_RETRY $(date -Is)"; sleep 120
done
