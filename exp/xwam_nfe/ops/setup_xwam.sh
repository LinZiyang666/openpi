#!/bin/bash
# X-WAM (sharinka0715/X-WAM, Apache-2.0) environment, following its README (torch 2.8 cu129 + requirements.txt + flash-attn 2.8.3
# prebuilt wheel); servers also download the Wan2.2-TI2V-5B base and the X-WAM checkpoints, clients also install the
# robocasa/robosuite submodules (the RoboCasa-2024 benchmark) + kitchen assets.
# usage: setup_xwam.sh <code dir (SSD)> <data dir (HDD/scratch)> <server|client> <uv home with .local/bin uv> [UV_CACHE_DIR]
# Storage discipline: code + venv under <code dir>, every model/dataset/cache under <data dir>. Idempotent.
set -u -o pipefail
CODE=${1:?code dir}; DATA=${2:?data dir}; ROLE=${3:?server|client}; UHOME=${4:?uv home}; export UV_CACHE_DIR=${5:-$UHOME/.cache/uv}
export HF_HOME=$DATA/hf_home; export PATH=$UHOME/.local/bin:/usr/local/bin:/usr/bin:/bin
command -v uv >/dev/null || { echo "SETUP_FAIL uv not on PATH"; exit 1; }
mkdir -p "$DATA" "$HF_HOME" "$(dirname "$CODE")"
FA=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
if [ ! -d "$CODE/.git" ]; then
  echo "== $(date +%H:%M:%S) clone"
  git clone https://github.com/sharinka0715/X-WAM.git "$CODE" 2>&1 | tail -1 || { echo SETUP_FAIL clone; exit 1; }
fi
cd "$CODE"
if [ "$ROLE" = "client" ] && [ ! -f third_party/robocasa/setup.py ]; then
  echo "== $(date +%H:%M:%S) submodules robocasa + robosuite"
  git submodule update --init --depth 1 third_party/robocasa third_party/robosuite 2>&1 | tail -2 || { echo SETUP_FAIL submodules; exit 1; }
fi
if [ ! -x .venv/bin/python ]; then
  echo "== $(date +%H:%M:%S) venv"
  uv venv --python 3.10 .venv 2>&1 | tail -1 || { echo SETUP_FAIL venv; exit 1; }
fi
if [ "$ROLE" = "server" ]; then
  echo "== $(date +%H:%M:%S) torch 2.8 cu129 + requirements + flash-attn"
  uv pip install --python .venv/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129 2>&1 | tail -1 || { echo SETUP_FAIL torch; exit 1; }
  uv pip install --python .venv/bin/python -r requirements.txt 2>&1 | tail -1 || { echo SETUP_FAIL requirements; exit 1; }
  uv pip install --python .venv/bin/python "$FA" 2>&1 | tail -1 || { echo SETUP_FAIL flash-attn; exit 1; }
  uv pip install --python .venv/bin/python pyzmq huggingface_hub 2>&1 | tail -1
  .venv/bin/python -c "import torch, flash_attn, diffusers, transformers; print('server import ok torch', torch.__version__, 'fa', flash_attn.__version__, 'cuda', torch.cuda.is_available())" || { echo SETUP_FAIL import; exit 1; }
  echo "== $(date +%H:%M:%S) checkpoints -> $DATA"
  .venv/bin/python - <<PY || { echo SETUP_FAIL ckpt; exit 1; }
import os
from huggingface_hub import snapshot_download
p = snapshot_download("sharinka0715/X-WAM-checkpoints", local_dir=os.path.join("$DATA", "checkpoints"))
print("xwam ckpts at", p, sorted(os.listdir(p)))
q = snapshot_download("Wan-AI/Wan2.2-TI2V-5B", local_dir=os.path.join("$DATA", "wan22_5b"))
print("wan2.2 at", q, sorted(os.listdir(q))[:12])
PY
else
  echo "== $(date +%H:%M:%S) client deps (sim only: robosuite + robocasa submodules, zmq, numpy 1.23.5)"
  uv pip install --python .venv/bin/python "numpy==1.23.5" pyzmq tyro imageio scipy tqdm h5py opencv-python 2>&1 | tail -1
  uv pip install --python .venv/bin/python -e third_party/robosuite 2>&1 | tail -1 || { echo SETUP_FAIL robosuite; exit 1; }
  uv pip install --python .venv/bin/python -e third_party/robocasa 2>&1 | tail -1 || { echo SETUP_FAIL robocasa; exit 1; }
  uv pip install --python .venv/bin/python "numpy==1.23.5" 2>&1 | tail -1
  .venv/bin/python -c "import robosuite, robocasa, zmq, numpy; print('client import ok numpy', numpy.__version__, 'robosuite', robosuite.__version__)" || { echo SETUP_FAIL import; exit 1; }
  A=third_party/robocasa/robocasa/models/assets
  if [ ! -d "$A/objects/objaverse" ]; then
    echo "== $(date +%H:%M:%S) kitchen assets"
    printf 'y\ny\ny\ny\ny\ny\ny\n' | .venv/bin/python third_party/robocasa/robocasa/scripts/download_kitchen_assets.py 2>&1 | tail -3
    [ -d "$A/objects/objaverse" ] || { echo SETUP_FAIL assets; exit 1; }
  fi
  [ -f third_party/robocasa/robocasa/macros_private.py ] || .venv/bin/python third_party/robocasa/robocasa/scripts/setup_macros.py 2>&1 | tail -1
fi
echo "SETUP_XWAM_DONE $ROLE $(date +%H:%M:%S)"
