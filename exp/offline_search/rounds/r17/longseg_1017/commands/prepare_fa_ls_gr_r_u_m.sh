#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
/home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.stage --run /data/mw_finetune/eval/fa_ls_gr_r_u_m --reprepare
