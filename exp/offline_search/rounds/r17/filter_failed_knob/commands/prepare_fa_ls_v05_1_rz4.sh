#!/usr/bin/env bash
# coordinator 10-18 14:3x: fa_ls_v05_1 failed at launch (0 accepted) because worker_resilience.install_worker
# required worker_memory.reservation_mib, absent from its 10-10 frozen worker payload. Fixed; reset and re-prepare.
# vfix 10-18: autoscale_1017.reset_zero refuses this run (no driver journal although replicas/workers were
# spawned and closed), and its copy keeps the 14:06 fixed runtime.tar without LONGSEG_RUNTIME_FIX.json, which
# staging rejects. longseg_1017.vfix retry keeps the same backups and reset_zero semantics, restores the
# source runtime archive, re-pins the bootstrap (serve_loeo_episodes fallback), then runs the native
# formal_trial.prepare --reprepare. Restartable after a failure past the reset.
set -euo pipefail
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.vfix retry --run /data/mw_finetune/eval/fa_ls_v05_1 --backup /data/mw_finetune/eval/fa_ls_v05_1.before_retry_1022d --remote-backup /data/mw_finetune/eval/fa_ls_v05_1.before_retry_1022d --remote-stage-backup /tmp/r17_formal/fa_ls_v05_1.before_retry_1022d
