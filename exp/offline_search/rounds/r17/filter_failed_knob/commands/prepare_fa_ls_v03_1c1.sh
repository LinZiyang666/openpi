#!/usr/bin/env bash
set -euo pipefail
# H100 continuation of fa_ls_v03_1 (stopped 10-21 04:17 at 30,536/95,594): servers at OMP=2 instead of 4 (campaign server_threads).
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
unset R17_H100_NODE R17_H100_HOST R17_DRIVER_PORT
exec .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.stage --run /data/mw_finetune/eval/fa_ls_v03_1c1
