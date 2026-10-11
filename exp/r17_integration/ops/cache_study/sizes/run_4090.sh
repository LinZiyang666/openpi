#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
exec .venv/bin/python -u -m exp.r17_integration.ops.cache_lane \
  --lane 4090 --directory /data/mw_finetune/eval/cszmdp \
  --after-lane /data/mw_finetune/eval/csl10dp --wait-for-pass --run
