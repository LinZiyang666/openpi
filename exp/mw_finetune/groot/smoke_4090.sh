#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/launch_4090.sh" --smoke --batch-size 2 --workers 4 \
  --dataset /data/mw_finetune/datasets/metaworld_mt50_groot_smoke \
  --output /data/mw_finetune/groot/4090_standard_smoke "$@"
