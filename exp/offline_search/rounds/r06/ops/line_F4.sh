#!/bin/bash
# Frontier completion, GR00T LIBERO-10-500 arms (pilot cell complete), two servers on 23150/23151, budget-gated retry.
source /home/weiland/.claude/jobs/a607dd74/tmp/gpu_gate.sh
R=/home/weiland/trace_runs/os_closed_loop/r06_frontier
ARMS="r6q2_groot_l10_500_A_dose0p125 r6q2_groot_l10_500_A_dose0p25 r6q2_groot_l10_500_risk_rho0p13 r6q2_groot_l10_500_risk_rho0p18 r6q2_groot_l10_500_CycleTail_k8"
cd /home/weiland/projects/openpi
unset OSCL_MANIFEST OSCL_EPISODES OSCL_TASKS
for attempt in $(seq 1 30); do
  ok=0; while [ $ok -lt 3 ]; do if admit 19000 "$(pilot_reserve_mb)"; then ok=$((ok+1)); else ok=0; fi; sleep 60; done
  echo "F4_ATTEMPT $attempt $(date -Is) $GATE_MSG"
  for a in $ARMS; do [ -e $R/state/$a.ERROR ] && rm $R/state/$a.ERROR; done
  for p in 23150 23151; do pid=$(ss -ltnp 2>/dev/null | grep ":$p " | grep -oP 'pid=\K[0-9]+' | head -1); [ -n "$pid" ] && ps -o args= -p $pid | grep -q 'os-tag r[0-9a-z_]*_2315[01]' && kill $pid && tmux kill-session -t oscl$p 2>/dev/null; done
  PORTS=23150,23151 WPS=24 SERVER_CPUS=0-17,44-61 taskset -c 34-37,78-81 bash exp/offline_search/closed_loop/ops/chain.sh $R $ARMS 2>&1 | tee -a $R/chain_console_F4.log
  left=0; for a in $ARMS; do ls $R/state/$a.DONE $R/state/$a.manifest_*.DONE >/dev/null 2>&1 || left=$((left+1)); done
  [ $left -eq 0 ] && { echo "F4_DONE $(date -Is)"; break; }
  echo "F4_RETRY left=$left $(date -Is)"; sleep 120
done
