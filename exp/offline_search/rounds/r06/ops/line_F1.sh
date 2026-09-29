#!/bin/bash
# Frontier completion, placement-independent arms (both l10-50 cells, B + extra dose .25/.5/.75), single port 23162, budget-gated retry.
source /home/weiland/.claude/jobs/a607dd74/tmp/gpu_gate.sh
R=/home/weiland/trace_runs/os_closed_loop/r06_frontier
ARMS="r6q2_pi05_l10_50_B_dose0p25 r6q2_groot_l10_50_B_dose0p25 r6q2_pi05_l10_50_B_dose0p5 r6q2_groot_l10_50_B_dose0p5 r6q2_pi05_l10_50_B_dose0p75 r6q2_groot_l10_50_B_dose0p75"
cd /home/weiland/projects/openpi
unset OSCL_MANIFEST OSCL_EPISODES OSCL_TASKS
for attempt in $(seq 1 30); do
  ok=0; while [ $ok -lt 3 ]; do if admit 10500 "$(pilot_reserve_mb)"; then ok=$((ok+1)); else ok=0; fi; sleep 60; done
  echo "F1_ATTEMPT $attempt $(date -Is) $GATE_MSG"
  for a in $ARMS; do [ -e $R/state/$a.ERROR ] && rm $R/state/$a.ERROR; done
  pid=$(ss -ltnp 2>/dev/null | grep ":23162 " | grep -oP 'pid=\K[0-9]+' | head -1); [ -n "$pid" ] && ps -o args= -p $pid | grep -q 'os-tag r[0-9a-z_]*_23162' && kill $pid && tmux kill-session -t oscl23162 2>/dev/null
  PORTS=23162 WPS=24 SERVER_CPUS=26-29,70-73 taskset -c 34-37,78-81 bash exp/offline_search/closed_loop/ops/chain.sh $R $ARMS 2>&1 | tee -a $R/chain_console_F1.log
  left=0; for a in $ARMS; do ls $R/state/$a.DONE $R/state/$a.manifest_*.DONE >/dev/null 2>&1 || left=$((left+1)); done
  [ $left -eq 0 ] && { echo "F1_DONE $(date -Is)"; break; }
  echo "F1_RETRY left=$left $(date -Is)"; sleep 120
done
