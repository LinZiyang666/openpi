#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/../launch.sh" --domain libero \
  --checkpoint /data/mw_finetune/dp/checkpoints/libero \
  --dataset /data/mw_finetune/dp/libero_96 --init pretrained "$@"
