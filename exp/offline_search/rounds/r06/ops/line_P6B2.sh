#!/bin/bash
# After line P6B: the 16 trigger leave-one-out arms (B variants, full model), ports 23160/23161.
cd /home/weiland/projects/openpi
R=/home/weiland/trace_runs/os_closed_loop/r06_abl
while tmux has-session -t line_P6B 2>/dev/null; do sleep 60; done
unset OSCL_MANIFEST OSCL_EPISODES OSCL_TASKS
export PORTS=23160,23161 WPS=24 SERVER_CPUS=18-25,62-69
echo "LINE_P6B2_START $(date -Is)"
taskset -c 30-33,74-77 bash exp/offline_search/closed_loop/ops/chain.sh $R \
  r6p2_stuck_p_l10_50 r6p2_terminal_p_l10_50 r6p2_overtime_p_l10_50 r6p2_no_progress_p_l10_50 \
  r6p2_stuck_g_l10_50 r6p2_terminal_g_l10_50 r6p2_overtime_g_l10_50 r6p2_no_progress_g_l10_50 \
  r6p2_stuck_p_sp_50 r6p2_terminal_p_sp_50 r6p2_overtime_p_sp_50 r6p2_no_progress_p_sp_50 \
  r6p2_stuck_g_sp_50 r6p2_terminal_g_sp_50 r6p2_overtime_g_sp_50 r6p2_no_progress_g_sp_50 \
  2>&1 | tee -a $R/chain_console_B2.log
echo "LINE_P6B2_DONE $(date -Is)"
