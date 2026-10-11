#!/usr/bin/env bash
set -euo pipefail
ROOT=${MW_ROOT:-/data/mw_finetune}
VENV=$ROOT/envs/fastwam
UV=$(command -v uv || true)
if test -z "$UV"; then UV=/home/exouser/.local/bin/uv; fi
export UV_CACHE_DIR=$ROOT/fastwam/uv_cache HF_HOME=$ROOT/hf
mkdir -p "$ROOT/fastwam" "$ROOT/vendor" "$ROOT/models"
python3 - "$ROOT" <<'PY'
import shutil,sys
from pathlib import Path
p=Path(sys.argv[1]); free=shutil.disk_usage(p).free/1e9
if free < 65: raise SystemExit(f'Need 35GB setup headroom + 30GB floor; {free:.1f}GB free')
PY
if ! test -d "$ROOT/vendor/lerobot/.git"; then git clone https://github.com/huggingface/lerobot "$ROOT/vendor/lerobot"; git -C "$ROOT/vendor/lerobot" checkout b9cb121cb7d4e3e26ec5c906d08dda68d151ba52; fi
if ! test -d "$ROOT/vendor/FastWAM/.git"; then git clone https://github.com/yuantianyuan01/FastWAM "$ROOT/vendor/FastWAM"; git -C "$ROOT/vendor/FastWAM" checkout 7faa71108368fbb3b6885649f112af607427a2d4; fi
if ! test -x "$VENV/bin/python"; then "$UV" venv --python 3.12 "$VENV"; fi
"$UV" pip install --python "$VENV/bin/python" torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install --python "$VENV/bin/python" -e "$ROOT/vendor/lerobot[fastwam,training]" 'torch==2.7.1' 'torchvision==0.22.1' 'torchcodec==0.3.0' 'transformers==5.5.4' 'diffusers==0.39.0' modelscope omegaconf einops sentencepiece ftfy pyarrow pillow pytest websockets msgpack numpy
"$UV" pip install --python "$VENV/bin/python" --no-deps -e "$ROOT/vendor/FastWAM"
"$UV" pip freeze --python "$VENV/bin/python" > "$ROOT/fastwam/environment.txt"
"$VENV/bin/python" - "$ROOT" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download('lerobot/fastwam_libero_uncond_2cam224',revision='53983e1249b4eb4d89ab42b40b8678a882d331ad',local_dir=sys.argv[1]+'/models/fastwam_libero',allow_patterns=['*.json','model.safetensors','README.md'])
PY
