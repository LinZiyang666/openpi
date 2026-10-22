#!/usr/bin/env bash
set -euo pipefail
# H100 rerun of fa_ls_v08_1 (40,000; 0 accepted in v08_1 and v08_1c1) as originally sealed, without the Fast-WAM text
# table: the 10-1x frozen runtime and cache yamls reject the overlaid code (batch implementation identity, then
# adapter_binding checkpoint_identity, which digests the fastwam code). resume.py --remaining from v08_1c1.
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
unset R17_H100_NODE R17_H100_HOST R17_DRIVER_PORT
exec .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.launch --run /data/mw_finetune/eval/fa_ls_v08_1c2
