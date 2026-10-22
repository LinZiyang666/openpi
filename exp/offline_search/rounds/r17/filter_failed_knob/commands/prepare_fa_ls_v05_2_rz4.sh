#!/usr/bin/env bash
# vfix 10-18: fa_ls_v05_2 (14:32 launch) is PARTIAL, all_owned_closed=true, stopped by SIGTERM. Its 4 accepted
# rows are all FatalEpisodeError ('LIBERO inference failed': the server's knob runtime raised AttributeError
# serve_loeo_episodes on the pre-LOEO 211cb3c9 LayoutRecipe). No clean episode exists, so retry from zero under
# the same name: longseg_1017.vfix retry archives the whole run (incl. the error rows) and every node's owned
# trees to .before_retry_1022d siblings, copies the definition, restores the source runtime archive, re-pins
# the bootstrap with the fallback, then runs the native formal_trial.prepare --reprepare. Restartable.
set -euo pipefail
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.vfix retry --run /data/mw_finetune/eval/fa_ls_v05_2 --backup /data/mw_finetune/eval/fa_ls_v05_2.before_retry_1022d --remote-backup /data/mw_finetune/eval/fa_ls_v05_2.before_retry_1022d --remote-stage-backup /tmp/r17_formal/fa_ls_v05_2.before_retry_1022d
