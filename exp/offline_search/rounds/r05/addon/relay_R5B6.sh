#!/bin/bash
# Ports 23160,23161, after relay_R5B5 ends: R5 add-on, GR00T one-block (10-step) pure-cache anchor_tail (R4 analysis §11 item 6).
cd /home/weiland/projects/openpi
R=/home/weiland/trace_runs/os_closed_loop
while tmux has-session -t relay_R5B5 2>/dev/null; do sleep 60; done
echo "RELAY_R5B6_START $(date -Is)"
unset OSCL_MANIFEST OSCL_EPISODES OSCL_TASKS
export PORTS=23160,23161 WPS=24 SERVER_CPUS=18-25,62-69
CH="taskset -c 30-33,74-77 bash exp/offline_search/closed_loop/ops/chain.sh"
$CH $R/r05_x r5x_g_l10_500_tail1u r5x_g_l10_50_tail1u r5x_g_sp_500_tail1u r5x_g_sp_50_tail1u 2>&1 | tee -a $R/r05_x/chain_console.log
echo "RELAY_R5B6_ALL_DONE $(date -Is)"
