#!/usr/bin/env bash
set -euo pipefail
# continuation of fa_sd_fw_f_l_k3 (sean lane 1; replicas OOMed 18:06 beside lane-2 DP R4): resume --remaining --verify-closed,
# 1,936 episodes, R3 / 57 GiB (same 19 GiB per replica). Lane env comes from the chain.
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
exec .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.stage --run /data/mw_finetune/eval/fa_sd_fw_f_l_k3c1
