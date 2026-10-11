#!/usr/bin/env bash
# A's harness supplies the websocket server; B supplies load_policy.
set -euo pipefail
hardware=${GR00T_HARDWARE:-4090}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ $hardware == h100 ]]; then
  gr00t_root=${GR00T_ROOT:-/home/exouser/gr00t_n15}
  gr00t_python=${GR00T_PYTHON:-/home/exouser/gr00t_n15_venv/.venv/bin/python}
else
  gr00t_root=${GR00T_ROOT:-/home/weiland/projects/openpi_ext/third_party/gr00t_n15}
  gr00t_python=${GR00T_PYTHON:-/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python}
fi
export PYTHONPATH="$gr00t_root:$repo:$repo/src:$repo/packages/openpi-client/src:${PYTHONPATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
cd "$repo"
flock /tmp/r17_gpu.lock bash -c 'nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv; exec "$@"' r17 \
  "$gr00t_python" -u -m exp.mw_finetune.serve --model groot "$@"
