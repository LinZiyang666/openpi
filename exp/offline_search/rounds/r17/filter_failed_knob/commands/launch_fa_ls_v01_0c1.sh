#!/usr/bin/env bash
set -euo pipefail
# Continuation of fa_ls_v01_0 (stopped 01:41 by a stall-artifact ctime change, 12,500 accepted).
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
exec .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.launch --run /data/mw_finetune/eval/fa_ls_v01_0c1
