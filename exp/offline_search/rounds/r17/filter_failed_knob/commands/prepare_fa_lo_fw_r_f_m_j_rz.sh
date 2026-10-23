#!/usr/bin/env bash
# 10-22 21:0x: fa_lo_fw_r_f_m_j failed at startup with 0 episodes (pod reverse tunnels lost in the 17:4x pod restart; rebuilt). Reset and re-prepare.
set -euo pipefail
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
export R17_H100_NODE=jupyter-ziyang10-02 R17_H100_HOST=149.165.153.233 R17_DRIVER_PORT=23192
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.vfix retry --run /data/mw_finetune/eval/fa_lo_fw_r_f_m_j --backup /data/mw_finetune/eval/fa_lo_fw_r_f_m_j.before_retry_1022a --remote-backup /data/mw_finetune/eval/fa_lo_fw_r_f_m_j.before_retry_1022a --remote-stage-backup /tmp/r17_formal/fa_lo_fw_r_f_m_j.before_retry_1022a
