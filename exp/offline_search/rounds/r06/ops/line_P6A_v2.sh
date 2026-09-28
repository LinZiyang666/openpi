#!/bin/bash
# R6 full-model arms, ports 23150/23151: pi0.5 B reps (6) + GR00T B rep3 (4) in r06_paper, then the 16 trigger
# leave-one-out arms in r06_abl. GR00T A reps moved to line_P6C2 (pure cache).
cd /home/weiland/projects/openpi
unset OSCL_MANIFEST OSCL_EPISODES OSCL_TASKS
export PORTS=23150,23151 WPS=24 SERVER_CPUS=0-17,44-61
echo "LINE_P6A_V2_START $(date -Is)"
taskset -c 34-37,78-81 bash exp/offline_search/closed_loop/ops/chain.sh /home/weiland/trace_runs/os_closed_loop/r06_paper \
  r5q1_c10_p_sp_50_rep2 r5q1_c10_p_sp_500_rep2 \
  r5q1_c10_p_l10_50_rep3 r5q1_c10_p_l10_500_rep3 r5q1_c10_p_sp_50_rep3 r5q1_c10_p_sp_500_rep3 \
  r6p1_c10_g_l10_50_rep3 r6p1_c10_g_l10_500_rep3 r6p1_c10_g_sp_50_rep3 r6p1_c10_g_sp_500_rep3 \
  2>&1 | tee -a /home/weiland/trace_runs/os_closed_loop/r06_paper/chain_console_A.log || exit 1
ls /home/weiland/trace_runs/os_closed_loop/r06_paper/state/*.ERROR >/dev/null 2>&1 && exit 1
taskset -c 34-37,78-81 bash exp/offline_search/closed_loop/ops/chain.sh /home/weiland/trace_runs/os_closed_loop/r06_abl \
  r6p2_stuck_p_l10_50 r6p2_terminal_p_l10_50 r6p2_overtime_p_l10_50 r6p2_no_progress_p_l10_50 \
  r6p2_stuck_g_l10_50 r6p2_terminal_g_l10_50 r6p2_overtime_g_l10_50 r6p2_no_progress_g_l10_50 \
  r6p2_stuck_p_sp_50 r6p2_terminal_p_sp_50 r6p2_overtime_p_sp_50 r6p2_no_progress_p_sp_50 \
  r6p2_stuck_g_sp_50 r6p2_terminal_g_sp_50 r6p2_overtime_g_sp_50 r6p2_no_progress_g_sp_50 \
  2>&1 | tee -a /home/weiland/trace_runs/os_closed_loop/r06_abl/chain_console_A.log
echo "LINE_P6A_V2_DONE $(date -Is)"
