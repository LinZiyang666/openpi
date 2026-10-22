#!/usr/bin/env bash
set -euo pipefail
# H100 rerun of fa_ls_v08_1 (40,000; 0 accepted in v08_1 and v08_1c1) with the Fast-WAM text table; the old runtime's identity checks
# (removed by the owner 10-10) are overlaid away (runtime_fix.RELAXED_FILES + provider.py with the table).
# resume.py --remaining from v08_1c4 (c4 stopped at cache_recipe's teacher receipt; owner-removed checks are now stripped from old releases, runtime_fix.strip_defenses).
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
unset R17_H100_NODE R17_H100_HOST R17_DRIVER_PORT
exec .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.launch --run /data/mw_finetune/eval/fa_ls_v08_1c5
