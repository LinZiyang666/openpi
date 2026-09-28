#!/bin/bash
# Ports 23150,23151, after relay_R5A4 ends: R5 add-on, two replicates of K7 anchor_tail l10-500 (R4 analysis §11 item 1).
cd /home/weiland/projects/openpi
R=/home/weiland/trace_runs/os_closed_loop
while tmux has-session -t relay_R5A4 2>/dev/null; do sleep 60; done
echo "RELAY_R5A5_START $(date -Is)"
unset OSCL_MANIFEST OSCL_EPISODES OSCL_TASKS
export PORTS=23150,23151 WPS=24 SERVER_CPUS=0-17,44-61
CH="taskset -c 34-37,78-81 bash exp/offline_search/closed_loop/ops/chain.sh"
$CH $R/r05_x r5x_p_l10_500_k7tail_repa r5x_p_l10_500_k7tail_repb 2>&1 | tee -a $R/r05_x/chain_console.log
echo "RELAY_R5A5_ALL_DONE $(date -Is)"
