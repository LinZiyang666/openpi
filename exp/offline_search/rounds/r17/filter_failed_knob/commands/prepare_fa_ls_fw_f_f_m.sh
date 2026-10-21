#!/usr/bin/env bash
set -euo pipefail
# Claimed 10-21 by jupyter-ziyang10 as fa_ls_fw_f_f_m_j (Fast-WAM without T5 on the MIG slice, with its pod pure
# reference fa_ls_fw_f_p_m_j). While the claim file exists the H100 chain skips this batch.
CLAIM=/home/weiland/projects/openpi_ext/scratch/r17_ops/mw/claim_pod1_fa_ls_fw_f_f_m
if [ -f "$CLAIM" ]; then echo "claimed by jupyter-ziyang10 ($CLAIM)"; exit 3; fi
exec bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/longseg_1017/commands/prepare_fa_ls_fw_f_f_m.sh
