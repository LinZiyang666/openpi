#!/usr/bin/env bash
# 10-22 19:2x: fa_ls_v07_0 (the audit's only real gap, 32 episodes) failed at launch with 0 episodes: its campaign
# predates the 'suites' field the text-table server reads (fixed: absent suites = all tasks). Reset and re-prepare.
set -euo pipefail
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.vfix retry --run /data/mw_finetune/eval/fa_ls_v07_0 --backup /data/mw_finetune/eval/fa_ls_v07_0.before_retry_1022a --remote-backup /data/mw_finetune/eval/fa_ls_v07_0.before_retry_1022a --remote-stage-backup /tmp/r17_formal/fa_ls_v07_0.before_retry_1022a
