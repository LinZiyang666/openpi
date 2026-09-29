#!/bin/bash
# Frontier completion, spatial-500 arms (A15 duplicate and clamped rho .065 skipped), single port 23162, budget-gated retry.
source /home/weiland/.claude/jobs/a607dd74/tmp/gpu_gate.sh
R=/home/weiland/trace_runs/os_closed_loop/r06_frontier
ARMS="r6q2_pi05_spatial_500_A_dose0p03125 r6q2_groot_spatial_500_risk_rho0p09 r6q2_pi05_spatial_500_A_dose0p0625 r6q2_groot_spatial_500_A15_confirmation r6q2_pi05_spatial_500_risk_rho0p085 r6q2_groot_spatial_500_phase_confirmation r6q2_pi05_spatial_500_risk_rho0p105 r6q2_groot_spatial_500_CycleTail15_k8"
cd /home/weiland/projects/openpi
unset OSCL_MANIFEST OSCL_EPISODES OSCL_TASKS
for attempt in $(seq 1 30); do
  ok=0; while [ $ok -lt 3 ]; do if admit 10500 "$(pilot_reserve_mb)"; then ok=$((ok+1)); else ok=0; fi; sleep 60; done
  echo "F5_ATTEMPT $attempt $(date -Is) $GATE_MSG"
  for a in $ARMS; do [ -e $R/state/$a.ERROR ] && rm $R/state/$a.ERROR; done
  pid=$(ss -ltnp 2>/dev/null | grep ":23162 " | grep -oP 'pid=\K[0-9]+' | head -1); [ -n "$pid" ] && ps -o args= -p $pid | grep -q 'os-tag r[0-9a-z_]*_23162' && kill $pid && tmux kill-session -t oscl23162 2>/dev/null
  PORTS=23162 WPS=24 SERVER_CPUS=26-29,70-73 taskset -c 34-37,78-81 bash exp/offline_search/closed_loop/ops/chain.sh $R $ARMS 2>&1 | tee -a $R/chain_console_F5.log
  left=0; for a in $ARMS; do ls $R/state/$a.DONE $R/state/$a.manifest_*.DONE >/dev/null 2>&1 || left=$((left+1)); done
  [ $left -eq 0 ] && { echo "F5_DONE $(date -Is)"; break; }
  echo "F5_RETRY left=$left $(date -Is)"; sleep 120
done
