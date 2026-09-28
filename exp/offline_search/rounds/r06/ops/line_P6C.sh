#!/bin/bash
# R6 metric ablation (pure cache, stage-1-only servers ~2.4 GB each), ports 23162/23163, while the GPU is shared with another project.
cd /home/weiland/projects/openpi
R=/home/weiland/trace_runs/os_closed_loop/r06_abl
unset OSCL_MANIFEST OSCL_EPISODES OSCL_TASKS
export PORTS=23162,23163 WPS=24 SERVER_CPUS=26-29,70-73
echo "LINE_P6C_START $(date -Is)"
taskset -c 26-29,70-73 bash exp/offline_search/closed_loop/ops/chain.sh $R \
  r6p2_identity_p_l10_50 r6p2_identity_p_l10_500 r6p2_identity_p_sp_50 r6p2_identity_p_sp_500 \
  r6p2_identity_g_l10_50 r6p2_identity_g_l10_500 r6p2_identity_g_sp_50 r6p2_identity_g_sp_500 \
  2>&1 | tee -a $R/chain_console_C.log
echo "LINE_P6C_DONE $(date -Is)"
