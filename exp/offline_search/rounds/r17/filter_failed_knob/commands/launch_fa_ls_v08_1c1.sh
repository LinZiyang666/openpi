#!/usr/bin/env bash
set -euo pipefail
# H100 rerun of fa_ls_v08_1 (40,000; 0 accepted): its 10-1x frozen runtime raised "resident batch implementation
# differs from sealed treatment" on the text-table server (10-22 10:43, FAIL, all_owned_closed). formal_batching.py
# now joins the text-table overlay (runtime_fix). resume.py --remaining; text table, 44 GiB, server_threads 4.
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
unset R17_H100_NODE R17_H100_HOST R17_DRIVER_PORT
exec .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.launch --run /data/mw_finetune/eval/fa_ls_v08_1c1
