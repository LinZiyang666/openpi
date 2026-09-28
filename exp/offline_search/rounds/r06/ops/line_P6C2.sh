#!/bin/bash
# R6 paper pure-cache arms (stage-1-only servers, ~2.4 GB each), ports 23162/23163, GPU-aware retry.
# Gate (tmp/gpu_gate.sh): ours+6000+pilot_reserve <= budget and free >= 6000+pilot_reserve, 3 checks x 60 s.
cd /home/weiland/projects/openpi
source /home/weiland/.claude/jobs/a607dd74/tmp/gpu_gate.sh
R=/home/weiland/trace_runs/os_closed_loop/r06_paper
ARMS="r4b3_p_sp_500_tail1uc_rep3 r5x_g_l10_50_tail1u_rep2 r5x_g_l10_500_tail1u_rep2 r5x_g_sp_50_tail1u_rep2 r5x_g_sp_500_tail1u_rep2 r5x_g_l10_50_tail1u_rep3 r5x_g_l10_500_tail1u_rep3 r5x_g_sp_50_tail1u_rep3 r5x_g_sp_500_tail1u_rep3"
unset OSCL_MANIFEST OSCL_EPISODES OSCL_TASKS
for attempt in $(seq 1 60); do
  ok=0
  while true; do
    if admit 6000 "$(pilot_reserve_mb)"; then ok=$((ok+1)); else ok=0; fi
    [ $ok -ge 3 ] && break
    sleep 60
  done
  echo "P6C2_ATTEMPT $attempt $(date -Is) $GATE_MSG"
  for a in $ARMS; do [ -e $R/state/$a.ERROR ] && rm $R/state/$a.ERROR; done
  for p in 23162 23163; do pid=$(ss -ltnp 2>/dev/null | grep ":$p " | grep -oP 'pid=\K[0-9]+' | head -1); [ -n "$pid" ] && ps -o args= -p $pid | grep -q 'os-tag r[0-9a-z_]*_2316[23]' && kill $pid && tmux kill-session -t oscl$p 2>/dev/null; done
  PORTS=23162,23163 WPS=24 SERVER_CPUS=26-29,70-73 taskset -c 34-37,78-81 bash exp/offline_search/closed_loop/ops/chain.sh $R $ARMS 2>&1 | tee -a $R/chain_console_C2.log
  left=0; for a in $ARMS; do [ -e $R/state/$a.DONE ] || left=$((left+1)); done
  [ $left -eq 0 ] && { echo "P6C2_DONE $(date -Is)"; break; }
  echo "P6C2_RETRY left=$left $(date -Is)"; sleep 120
done
