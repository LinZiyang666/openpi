#!/usr/bin/env bash
set -euo pipefail
# Claimed 10-21 by jupyter-ziyang10-02 as fa_ls_v08_1_j (pod same-card references fa_ls_fw_f_b_l_j, fa_ls_fw_f_p_l_j first).
# While the claim file exists the H100 chain skips this batch.
CLAIM=/home/weiland/projects/openpi_ext/scratch/r17_ops/mw/claim_pod2_fa_ls_v08_1
if [ -f "$CLAIM" ]; then echo "claimed by jupyter-ziyang10-02 ($CLAIM)"; exit 3; fi
exec bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/longseg_1017/commands/prepare_fa_ls_v08_1.sh
