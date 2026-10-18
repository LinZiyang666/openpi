#!/usr/bin/env bash
# coordinator 10-18 12:2x: reset fa_ls_fw_r_p_m (12:05 attempt closed FAIL/all_owned_closed/0 accepted) incl. H100 /tmp staging backup, re-prepare natively
set -euo pipefail
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m exp.offline_search.rounds.r17.autoscale_1017.reset_zero --run /data/mw_finetune/eval/fa_ls_fw_r_p_m --backup /data/mw_finetune/eval/fa_ls_fw_r_p_m.before_retry_1205 --remote-backup /data/mw_finetune/eval/fa_ls_fw_r_p_m.before_retry_1205 --remote-stage-backup /tmp/r17_formal/fa_ls_fw_r_p_m.before_retry_1205 --reprepare
