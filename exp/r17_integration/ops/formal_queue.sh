#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
exec /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.formal_queue "$@"
