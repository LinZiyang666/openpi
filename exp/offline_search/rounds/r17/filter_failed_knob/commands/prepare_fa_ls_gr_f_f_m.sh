#!/usr/bin/env bash
set -euo pipefail
# jupyter-ziyang10 serves a pod copy of this not-yet-started H100 batch (fa_ls_gr_f_f_m_j, longseg_1017.pod_copy).
# While the claim file exists the H100 chain skips this batch; remove it (and restore .before_pod1_claim) to hand it back.
CLAIM=/home/weiland/projects/openpi_ext/scratch/r17_ops/mw/claim_pod1_fa_ls_gr_f_f_m
if [ -f "$CLAIM" ]; then echo "claimed by jupyter-ziyang10 as fa_ls_gr_f_f_m_j ($CLAIM)"; exit 3; fi
exec bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/longseg_1017/commands/prepare_fa_ls_gr_f_f_m.sh
