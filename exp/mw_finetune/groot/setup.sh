#!/usr/bin/env bash
# Run on the host that will train. Existing islands are reused without upgrades.
set -euo pipefail
hardware=${1:-4090}
if [[ $hardware == h100 ]]; then
  gr00t_root=${GR00T_ROOT:-/home/exouser/gr00t_n15}
  gr00t_python=${GR00T_PYTHON:-/home/exouser/gr00t_n15_venv/.venv/bin/python}
else
  gr00t_root=${GR00T_ROOT:-/home/weiland/projects/openpi_ext/third_party/gr00t_n15}
  gr00t_python=${GR00T_PYTHON:-/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python}
fi
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="$gr00t_root:$repo:${PYTHONPATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1
[[ $(git -C "$gr00t_root" rev-parse HEAD) == 4af2b622892f7dcb5aae5a3fb70bcb02dc217b96 ]] || {
  echo 'Expected the pinned NVIDIA n1.5-release checkout (see RECIPE.md)' >&2; exit 1;
}
uv_bin=$(command -v uv || printf '%s/.local/bin/uv' "$HOME")
"$uv_bin" pip install --python "$gr00t_python" tyro==0.9.17 tensorboard==2.19.0 'pytest>=8,<9'
"$gr00t_python" - <<'PY'
import importlib.metadata as m
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
import torch
assert torch.__version__.split('+')[0] == '2.5.1'
assert m.version('transformers') == '4.51.3'
assert m.version('flash-attn') == '2.7.4.post1'
root = Path('/data/mw_finetune'); root.mkdir(parents=True, exist_ok=True)
assert shutil.disk_usage(root).free > 36_000_000_000
snapshot_download('nvidia/GR00T-N1.5-3B', revision='869830fc749c35f34771aa5209f923ac57e4564e',
                  local_dir=root / 'models/groot_base', max_workers=4)
print({k: m.version(k) for k in ['torch', 'transformers', 'flash-attn', 'numpy', 'accelerate', 'decord', 'av', 'tyro', 'tensorboard']})
PY
