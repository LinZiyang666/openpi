#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
exec .venv/bin/python -u -m exp.r17_integration.ops.cache_lane \
  --lane h100 --directory /data/mw_finetune/eval/csznh \
  --after-lane /data/mw_finetune/eval/csl10h --wait-for-pass --run
