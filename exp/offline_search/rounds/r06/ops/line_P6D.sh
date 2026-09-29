#!/bin/bash
# Direct-token PCA ablation (8 pure-cache arms, r06_abl), ports 23162/23163 after the PCA smoke passes; budget-gated retry.
source /home/weiland/.claude/jobs/a607dd74/tmp/gpu_gate.sh
R=/home/weiland/trace_runs/os_closed_loop/r06_abl
SM=/home/weiland/trace_runs/os_closed_loop/r06_abl_pca_smoke
ARMS="r6p2_direct_p_l10_50 r6p2_direct_g_l10_50 r6p2_direct_p_sp_50 r6p2_direct_g_sp_50 r6p2_direct_p_l10_500 r6p2_direct_g_l10_500 r6p2_direct_p_sp_500 r6p2_direct_g_sp_500"
while tmux has-session -t pca_smoke 2>/dev/null; do sleep 30; done
for a in r6p2_direct_p_sp_50 r6p2_direct_g_sp_50; do ls $SM/state/$a*.DONE >/dev/null 2>&1 || { echo "P6D_ABORT smoke arm $a not DONE"; exit 1; }; done
echo "P6D_SMOKE_OK $(date -Is)"
cd /home/weiland/projects/openpi
unset OSCL_MANIFEST OSCL_EPISODES OSCL_TASKS
for attempt in $(seq 1 30); do
  ok=0; while [ $ok -lt 3 ]; do if admit 6000 "$(pilot_reserve_mb)"; then ok=$((ok+1)); else ok=0; fi; sleep 60; done
  echo "P6D_ATTEMPT $attempt $(date -Is) $GATE_MSG"
  for a in $ARMS; do [ -e $R/state/$a.ERROR ] && rm $R/state/$a.ERROR; done
  for p in 23162 23163; do pid=$(ss -ltnp 2>/dev/null | grep ":$p " | grep -oP 'pid=\K[0-9]+' | head -1); [ -n "$pid" ] && ps -o args= -p $pid | grep -q 'os-tag r[0-9a-z_]*_2316[23]' && kill $pid && tmux kill-session -t oscl$p 2>/dev/null; done
  PORTS=23162,23163 WPS=24 SERVER_CPUS=26-29,70-73 taskset -c 34-37,78-81 bash exp/offline_search/closed_loop/ops/chain.sh $R $ARMS 2>&1 | tee -a $R/chain_console_D.log
  left=0; for a in $ARMS; do [ -e $R/state/$a.DONE ] || left=$((left+1)); done
  [ $left -eq 0 ] && { echo "P6D_DONE $(date -Is)"; break; }
  echo "P6D_RETRY left=$left $(date -Is)"; sleep 120
done
