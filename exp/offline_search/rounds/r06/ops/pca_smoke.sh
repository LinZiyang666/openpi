#!/bin/bash
# Direct-token PCA closed-loop smoke: 2 arms (pi05 / GR00T spatial 50) x task 0 x inits 0-3, port 23166, budget-gated.
source /home/weiland/.claude/jobs/a607dd74/tmp/gpu_gate.sh
S=/home/weiland/trace_runs/os_closed_loop/r06_abl_pca_smoke
ok=0; while [ $ok -lt 3 ]; do if admit 3500 "$(pilot_reserve_mb)"; then ok=$((ok+1)); else ok=0; fi; sleep 20; done
echo "PCA_SMOKE_ADMIT $GATE_MSG $(date -Is)"
cd /home/weiland/projects/openpi
unset OSCL_MANIFEST; export OSCL_EPISODES=0,1,2,3 OSCL_TASKS=0 PORTS=23166 WPS=4 SERVER_CPUS=18-21,62-65
taskset -c 34-37,78-81 bash exp/offline_search/closed_loop/ops/chain.sh $S r6p2_direct_p_sp_50 r6p2_direct_g_sp_50 2>&1 | tee -a $S/chain_console.log
echo "PCA_SMOKE_END $(date -Is)"
