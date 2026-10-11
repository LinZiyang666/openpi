#!/usr/bin/env bash
set -euo pipefail
hardware=$1
shift
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ $hardware == h100 ]]; then
  gr00t_root=${GR00T_ROOT:-/home/exouser/gr00t_n15}
  gr00t_python=${GR00T_PYTHON:-/home/exouser/gr00t_n15_venv/.venv/bin/python}
else
  gr00t_root=${GR00T_ROOT:-/home/weiland/projects/openpi_ext/third_party/gr00t_n15}
  gr00t_python=${GR00T_PYTHON:-/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python}
fi
export PYTHONPATH="$gr00t_root:$repo:$repo/src:${PYTHONPATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export R17_GPU_LOCK_HELD=1
mkdir -p /data/mw_finetune/groot/logs
log=/data/mw_finetune/groot/logs/${hardware}_$(date -u +%Y%m%dT%H%M%S)_$$.log
cd "$repo"
# Only probes/smokes have an enforced 15-minute wall-clock bound. Long training
# launch commands are handed to the coordinator and never run by this agent.
bounded=false
for arg in "$@"; do
  [[ $arg == --probe || $arg == --smoke ]] && bounded=true
done
run=("$gr00t_python" -u -m exp.mw_finetune.groot.train --hardware "$hardware" "$@")
if $bounded; then run=(timeout --signal=TERM --kill-after=10s 880s "${run[@]}"); fi
flock /tmp/r17_gpu.lock bash -c 'nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv; exec "$@"' r17 "${run[@]}" 2>&1 | tee "$log"
