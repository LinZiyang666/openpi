#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/vla.sh run --group /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/vla/runs/off/ffk1015_pi05_libero_reduced_off/group.json --workers 64 --port 23159
