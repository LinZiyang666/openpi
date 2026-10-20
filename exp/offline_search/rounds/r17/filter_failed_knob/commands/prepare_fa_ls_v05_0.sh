#!/usr/bin/env bash
set -euo pipefail
# h100-temp takes one not-yet-started batch at a time from the H100 queue tail (owner 10-20). While the claim
# file exists the H100 chain (no alias) skips this batch; the h100-temp chain (R17_H100_NODE=h100-temp) runs it.
CLAIM=/home/weiland/projects/openpi_ext/scratch/r17_ops/mw/claim_h100temp_fa_ls_v05_0
if [ "${R17_H100_NODE:-}" != h100-temp ] && [ -f "$CLAIM" ]; then echo "claimed by h100-temp ($CLAIM)"; exit 3; fi
exec bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/longseg_1017/commands/prepare_fa_ls_v05_0.sh
