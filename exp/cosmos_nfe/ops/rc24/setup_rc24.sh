#!/bin/bash
# Cosmos Policy x RoboCasa (2024, 24 tasks) environment, following ROBOCASA.md verbatim (no Docker):
#   uv sync --extra cu128 --group robocasa  ->  git clone moojink/robocasa-cosmos-policy + uv pip install -e
#   -> [clients only] download_kitchen_assets.py + setup_macros.py  ->  snapshot_download of the RoboCasa checkpoint.
# usage: setup_rc24.sh <repo dir> <fork dir> <HF_HOME> <UV_CACHE_DIR> <server|client> [UV_PYTHON_INSTALL_DIR]
# Idempotent: every step checks its own marker. `uv sync` is exact, so the editable fork is (re)installed after it.
set -u -o pipefail
REPO=${1:?repo}; FORK=${2:?fork dir}; export HF_HOME=${3:?HF_HOME}; export UV_CACHE_DIR=${4:?uv cache}; ROLE=${5:?server|client}
[ -n "${6:-}" ] && export UV_PYTHON_INSTALL_DIR=$6
export PATH=$HOME/.local/bin:${UV_HOME:-/home/zixuans8}/.local/bin:/usr/local/bin:/usr/bin:/bin
command -v uv >/dev/null || { echo "SETUP_FAIL uv not on PATH"; exit 1; }
mkdir -p "$HF_HOME" /tmp/cosmos
cd "$REPO" || exit 1
echo "== $(date +%H:%M:%S) uv sync robocasa group"
uv sync --extra cu128 --group robocasa --python 3.10 2>&1 | tail -4 || { echo SETUP_FAIL sync; exit 1; }
if [ ! -d "$FORK/.git" ]; then
  echo "== $(date +%H:%M:%S) clone fork"
  git clone --depth 1 https://github.com/moojink/robocasa-cosmos-policy.git "$FORK" 2>&1 | tail -2 || { echo SETUP_FAIL clone; exit 1; }
fi
echo "== $(date +%H:%M:%S) install fork (editable)"
uv pip install -e "$FORK" 2>&1 | tail -3 || { echo SETUP_FAIL pip; exit 1; }
# the fork pins numba==0.56.4 (numpy<1.24) but the cosmos venv is numpy 2.2 -> robosuite.utils.numba cannot import; take a numpy-2 numba
uv pip install "numba>=0.61,<0.62" 2>&1 | tail -2 || { echo SETUP_FAIL numba; exit 1; }
# remote-serving deps of exp/cosmos_nfe (not in the upstream lock; uv sync removes them)
uv pip install msgpack msgpack-numpy websockets 2>&1 | tail -2 || { echo SETUP_FAIL msgpack; exit 1; }
.venv/bin/python -c "import robosuite, robocasa, numpy; from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS as S; print('robocasa import ok; tasks', len(S), 'numpy', numpy.__version__)" || { echo SETUP_FAIL import; exit 1; }
if [ "$ROLE" = "client" ]; then
  A=$FORK/robocasa/models/assets
  [ -d "$A/textures" ] && [ -d "$A/fixtures" ] && [ -d "$A/objects/objaverse" ] && [ -d "$A/generative_textures" ] && touch "$FORK/.assets_done"
  if [ ! -f "$FORK/.assets_done" ]; then
    echo "== $(date +%H:%M:%S) kitchen assets (~8 GB: textures, fixtures, objaverse, generative_textures)"
    # printf (not yes) feeds the y/n prompts: yes would die of SIGPIPE and trip pipefail
    printf 'y\ny\ny\ny\ny\ny\ny\n' | .venv/bin/python "$FORK/robocasa/scripts/download_kitchen_assets.py" 2>&1 | tail -5
    [ -d "$A/textures" ] && [ -d "$A/fixtures" ] && [ -d "$A/objects/objaverse" ] && [ -d "$A/generative_textures" ] && touch "$FORK/.assets_done" || { echo SETUP_FAIL assets; exit 1; }
  fi
  if [ ! -f "$FORK/robocasa/macros_private.py" ]; then
    echo "== $(date +%H:%M:%S) macros"
    .venv/bin/python "$FORK/robocasa/scripts/setup_macros.py" 2>&1 | tail -2
  fi
fi
echo "== $(date +%H:%M:%S) checkpoint"
.venv/bin/python - <<PY || { echo SETUP_FAIL ckpt; exit 1; }
import os
from huggingface_hub import snapshot_download
# clients only need the dataset stats + T5 embeddings (the 3.9 GB .pt lives on the servers)
allow = None if "$ROLE" == "server" else ["*.json", "*.pkl", "*.md", ".gitattributes"]
p = snapshot_download("nvidia/Cosmos-Policy-RoboCasa-Predict2-2B", cache_dir=os.environ["HF_HOME"], allow_patterns=allow)
print("ckpt at", p, sorted(os.listdir(p)))
PY
echo "SETUP_RC24_DONE $(date +%H:%M:%S)"
