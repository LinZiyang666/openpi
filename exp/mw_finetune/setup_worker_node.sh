#!/usr/bin/env bash
# Run on a timan worker after sync_workers.py; touches only the new R17 tree.
set -euo pipefail
ROOT=${1:-/scratch/zixuans8/mw_r17}
case "$ROOT" in */mw_r17) ;; *) echo 'Expected a new .../mw_r17 directory' >&2; exit 2;; esac
mkdir -p "$ROOT/envs" "$ROOT/cache" "$ROOT/python" "$ROOT/reports" "$ROOT/runs"
export UV_CACHE_DIR="$ROOT/cache/uv" UV_PYTHON_INSTALL_DIR="$ROOT/python" UV_LINK_MODE=copy
UV=${R17_UV:-/shared/nas/data/m1/zixuans8/miniconda3/bin/uv}
"$UV" python install --no-bin 3.11.15
if [[ ! -x "$ROOT/envs/sim/bin/python" ]]; then
  "$UV" venv --python 3.11.15 "$ROOT/envs/sim"
fi
"$UV" pip install --python "$ROOT/envs/sim/bin/python" \
  'metaworld==3.0.0' 'gymnasium==1.3.0' 'mujoco==3.14.0' \
  'numpy==1.26.4' 'msgpack==1.2.2' 'websockets==17.1' \
  'PyOpenGL==3.1.10' 'glfw==2.10.2' 'pillow==12.3.0' \
  'imageio==2.37.4' 'typing_extensions==4.16.0' 'packaging==26.3' 'zipp==4.1.0'
"$UV" pip freeze --python "$ROOT/envs/sim/bin/python" > "$ROOT/reports/sim_freeze.txt"
"$ROOT/envs/sim/bin/python" -c 'import importlib.metadata as m; print({p:m.version(p) for p in ("metaworld","gymnasium","mujoco","numpy","msgpack","websockets")})'
du -sh "$ROOT"
