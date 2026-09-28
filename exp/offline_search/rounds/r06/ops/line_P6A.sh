#!/bin/bash
# R6 paper arms, ports 23150/23151: GR00T B 500 x2, pi0.5 B reps, GR00T A reps.
cd /home/weiland/projects/openpi
R=/home/weiland/trace_runs/os_closed_loop/r06_paper
unset OSCL_MANIFEST OSCL_EPISODES OSCL_TASKS
export PORTS=23150,23151 WPS=24 SERVER_CPUS=0-17,44-61
echo "LINE_P6A_START $(date -Is)"
taskset -c 34-37,78-81 bash exp/offline_search/closed_loop/ops/chain.sh $R \
  r6p1_c10_g_l10_500 r6p1_c10_g_sp_500 \
  r5q1_c10_p_l10_50_rep2 r5q1_c10_p_l10_500_rep2 r5q1_c10_p_sp_50_rep2 r5q1_c10_p_sp_500_rep2 \
  r5x_g_l10_50_tail1u_rep2 r5x_g_l10_500_tail1u_rep2 r5x_g_sp_50_tail1u_rep2 r5x_g_sp_500_tail1u_rep2 \
  r5q1_c10_p_l10_50_rep3 r5q1_c10_p_l10_500_rep3 r5q1_c10_p_sp_50_rep3 r5q1_c10_p_sp_500_rep3 \
  r5x_g_l10_50_tail1u_rep3 r5x_g_l10_500_tail1u_rep3 r5x_g_sp_50_tail1u_rep3 r5x_g_sp_500_tail1u_rep3 \
  2>&1 | tee -a $R/chain_console_A.log
echo "LINE_P6A_DONE $(date -Is)"
