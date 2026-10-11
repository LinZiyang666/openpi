#!/usr/bin/env bash
set -euo pipefail
ROOT=${MW_ROOT:-/data/mw_finetune}
PYTHON=${FASTWAM_PYTHON:-$ROOT/envs/fastwam/bin/python}
export HF_HOME=$ROOT/hf
# Run on weilandserver. Reuse A's pinned download; never download raw data again on H100.
"$PYTHON" - <<'PY'
from pathlib import Path
import os, shutil
from huggingface_hub import snapshot_download
root=Path(os.environ.get('MW_ROOT','/data/mw_finetune'))
if shutil.disk_usage(root).free/1e9 < 60:
    raise SystemExit('Need room for raw dataset plus prepared224 data; free-space floor is5GB locally')
snapshot_download('lerobot/metaworld_mt50', repo_type='dataset', revision='b46a1c185a2368660650a82ac5a5083a56722f48', local_dir=root/'datasets/metaworld_mt50', max_workers=4)
PY
