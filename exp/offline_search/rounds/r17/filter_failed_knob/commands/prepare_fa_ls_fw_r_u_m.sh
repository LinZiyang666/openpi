#!/usr/bin/env bash
set -euo pipefail
# Claimed 10-21 by jupyter-ziyang10-02 as fa_ls_fw_r_u_m_j (Fast-WAM without T5 on the MIG slice, with its pod pure
# reference fa_ls_fw_r_p_m_j). While the claim file exists the H100 chain skips this batch.
CLAIM=/home/weiland/projects/openpi_ext/scratch/r17_ops/mw/claim_pod2_fa_ls_fw_r_u_m
if [ -f "$CLAIM" ]; then echo "claimed by jupyter-ziyang10-02 ($CLAIM)"; exit 3; fi
exec bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/longseg_1017/commands/prepare_fa_ls_fw_r_u_m.sh
