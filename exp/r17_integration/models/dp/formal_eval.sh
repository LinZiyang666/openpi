#!/bin/bash
# Owned local batching extension of the unchanged formal_eval launcher.
set -euo pipefail
cd /home/weiland/projects/openpi
exec /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.models.dp.formal_launch "$@"
