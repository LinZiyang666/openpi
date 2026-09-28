#!/bin/bash
# R6 paper arms, ports 23160/23161: GR00T B 50 x2, pi0.5 A reps, GR00T B reps.
cd /home/weiland/projects/openpi
R=/home/weiland/trace_runs/os_closed_loop/r06_paper
unset OSCL_MANIFEST OSCL_EPISODES OSCL_TASKS
export PORTS=23160,23161 WPS=24 SERVER_CPUS=18-25,62-69
echo "LINE_P6B_START $(date -Is)"
taskset -c 30-33,74-77 bash exp/offline_search/closed_loop/ops/chain.sh $R \
  r6p1_c10_g_l10_50 r6p1_c10_g_sp_50 \
  r5t_p_l10_50_tail1uc_rep2 r5t_p_l10_500_tail1uc_rep2 r5t_p_sp_50_tail1uc_rep2 r4b3_p_sp_500_tail1uc_rep2 \
  r6p1_c10_g_l10_50_rep2 r6p1_c10_g_l10_500_rep2 r6p1_c10_g_sp_50_rep2 r6p1_c10_g_sp_500_rep2 \
  r5t_p_l10_50_tail1uc_rep3 r5t_p_l10_500_tail1uc_rep3 r5t_p_sp_50_tail1uc_rep3 r4b3_p_sp_500_tail1uc_rep3 \
  r6p1_c10_g_l10_50_rep3 r6p1_c10_g_l10_500_rep3 r6p1_c10_g_sp_50_rep3 r6p1_c10_g_sp_500_rep3 \
  2>&1 | tee -a $R/chain_console_B.log
echo "LINE_P6B_DONE $(date -Is)"
