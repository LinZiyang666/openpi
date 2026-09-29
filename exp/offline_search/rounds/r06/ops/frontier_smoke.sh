#!/bin/bash
# Frontier-completion adapter smoke: pi05 B+dose .5 and GR00T risk lottery rho .35, tasks 0-1 x inits 0-1, after line_P6D ends.
source /home/weiland/.claude/jobs/a607dd74/tmp/gpu_gate.sh
S=/home/weiland/trace_runs/os_closed_loop/r06_frontier_smoke
while tmux has-session -t line_P6D 2>/dev/null; do sleep 30; done
ok=0; while [ $ok -lt 3 ]; do if admit 10500 "$(pilot_reserve_mb)"; then ok=$((ok+1)); else ok=0; fi; sleep 20; done
echo "FRONTIER_SMOKE_ADMIT $GATE_MSG $(date -Is)"
cd /home/weiland/projects/openpi
unset OSCL_EPISODES OSCL_TASKS; export OSCL_MANIFEST=$S/manifests/smoke4.json PORTS=23162 WPS=4 SERVER_CPUS=26-29,70-73
taskset -c 34-37,78-81 bash exp/offline_search/closed_loop/ops/chain.sh $S r6q2_pi05_l10_50_B_dose0p5 r6q2_groot_l10_50_risk_rho0p35 2>&1 | tee -a $S/chain_console.log
echo "FRONTIER_SMOKE_END $(date -Is)"
