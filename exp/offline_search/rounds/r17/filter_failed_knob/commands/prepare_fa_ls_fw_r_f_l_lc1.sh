#!/usr/bin/env bash
set -euo pipefail
# Continuation with the shared-EGL worker code (10-19).
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
 .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.stage --run /data/mw_finetune/eval/fa_ls_fw_r_f_l_lc1
