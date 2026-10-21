#!/usr/bin/env bash
set -euo pipefail
# H100 continuation of fa_ls_fw_r_f_mc1 (h100-temp's owner stopped our servers 10-20 22:41; routing restored to the H100).
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
unset R17_H100_NODE R17_H100_HOST R17_DRIVER_PORT
exec .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.launch --run /data/mw_finetune/eval/fa_ls_fw_r_f_mc2
