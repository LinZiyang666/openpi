#!/usr/bin/env bash
set -euo pipefail
# Claimed 10-22 04:0x by jupyter-ziyang10-02 as fa_ls_fw_r_f_m_j (90,000 fresh on the pod; same-card reference
# fa_ls_fw_r_p_m_j PASS). While the claim file exists the H100 chain skips this batch.
CLAIM=/home/weiland/projects/openpi_ext/scratch/r17_ops/mw/claim_pod2_fa_ls_fw_r_f_m
if [ -f "$CLAIM" ]; then echo "claimed by jupyter-ziyang10-02 ($CLAIM)"; exit 3; fi
# H100 continuation of fa_ls_fw_r_f_mc1 (h100-temp's owner stopped our servers 10-20 22:41; routing restored to the H100).
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
unset R17_H100_NODE R17_H100_HOST R17_DRIVER_PORT
exec .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.stage --run /data/mw_finetune/eval/fa_ls_fw_r_f_mc2
