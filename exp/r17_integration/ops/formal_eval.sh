#!/usr/bin/env bash
set -euo pipefail
# --mps is opt-in: private campaign pipes/logs, replica-only attachment, leased cleanup.
# Native batching is explicit: --max-batch 8 --batch-wait-ms 5.
# Batched Fast-WAM study arms also require --paired-control <closed control run>.
cd /home/weiland/projects/openpi
exec /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.formal_eval "$@"
