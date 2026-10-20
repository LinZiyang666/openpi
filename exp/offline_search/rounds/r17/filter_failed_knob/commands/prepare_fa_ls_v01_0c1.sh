#!/usr/bin/env bash
set -euo pipefail
# Continuation of fa_ls_v01_0 (stopped 01:41 by a stall-artifact ctime change, 12,500 accepted).
# replica_startup_width 1 (10-20 02:0x): 4 replicas writing servers/manifest.json at once raced in the frozen runtime.
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
.venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.stage --run /data/mw_finetune/eval/fa_ls_v01_0c1 --reprepare
