#!/usr/bin/env bash
# Owner runs at a free H100 boundary. This file has NOT been executed.
set -euo pipefail
cd /home/weiland/projects/openpi
test "$(cat exp/offline_search/rounds/r17/autoscale_1017/ENABLED)" = off
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m exp.offline_search.rounds.r17.autoscale_1017.reset_zero \
  --run /data/mw_finetune/eval/fa_ls_fw_r_p_m \
  --backup /data/mw_finetune/eval/fa_ls_fw_r_p_m.before_retry_1205 \
  --remote-backup /data/mw_finetune/eval/fa_ls_fw_r_p_m.before_retry_1205 \
  --remote-stage-backup /tmp/r17_formal/fa_ls_fw_r_p_m.before_retry_1205 \
  --reprepare
.venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.launch \
  --run /data/mw_finetune/eval/fa_ls_fw_r_p_m
