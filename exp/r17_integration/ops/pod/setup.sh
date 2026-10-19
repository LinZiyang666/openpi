#!/bin/bash
# Install an R17 model-server environment on a jupyter pod (jupyter-ziyang10, -02).
#
# The pods have no root and no /data, and cannot mount. R17 server commands use
# absolute /data/... paths, so every launcher command on a pod runs through
# $BASE/bin/r17x, which runs it under proot with $DATA bound at /data.
# $BASE (proot, rsync, r17x) is on the shared NFS home; $DATA is on the pod's own
# disk (NFS takes ~6 MB/s for the venv's small files). Re-run after a pod restart.
#
# Usage (on the pod):  bash setup.sh <dp|fastwam> <rsync-url-prefix>
#   <rsync-url-prefix>: a temporary read-only rsync daemon on the H100 with modules
#   <prefix>_mw = /data/mw_finetune and <prefix>_xwam = /data/xwam, e.g.
#   rsync://149.165.153.233:23117/zljXXXX   (the H100 has a public IP; the pod pulls).
# Re-running is cheap: rsync only copies what changed.
set -euo pipefail
MODEL=${1:?dp or fastwam}
SRC=${2:?rsync url prefix}
BASE=/home/ziyang10/zl_r17
DATA=/tmp/zl_r17/data
BW=${BWLIMIT_KIBPS:-30000}
D=$DATA/mw_finetune
PY=cpython-3.12.3-linux-x86_64-gnu
mkdir -p "$BASE/bin" "$D/dp" "$D/envs" "$D/qualification" "$D/python" "$D/eval"

# 1. proot (static, checksum from the v5.5.0 release) and rsync (conda-forge).
if [ ! -x "$BASE/bin/proot" ]; then
  curl -sSL -m 120 -o "$BASE/bin/proot.part" https://github.com/proot-me/proot/releases/download/v5.5.0/proot
  echo "db185300a9daabbce86aca7632d8c7fdc4249f6f153171954013b70c5f491ec0  $BASE/bin/proot.part" | sha256sum -c
  chmod +x "$BASE/bin/proot.part" && mv "$BASE/bin/proot.part" "$BASE/bin/proot"
fi
[ -x "$BASE/tools/bin/rsync" ] || /opt/conda/bin/conda create -y -q -p "$BASE/tools" -c conda-forge --override-channels rsync

# 2. r17x: run argv with /data -> $DATA. It returns when argv exits; proot keeps
#    tracing any detached children (formal_process supervisors, servers) until they exit.
cat > "$BASE/bin/r17x.part" <<'EOF'
#!/bin/bash
BASE=/home/ziyang10/zl_r17
DATA=/tmp/zl_r17/data
export PATH=$BASE/tools/bin:$PATH
d=$(mktemp -d /tmp/r17x.XXXXXXXX)
R17X_DIR=$d setsid -w "$BASE/bin/proot" -b "$DATA:/data" -w "$PWD" /bin/sh -c \
  '"$@" >"$R17X_DIR/out" 2>"$R17X_DIR/err" </dev/null; echo $? >"$R17X_DIR/rc.part"; mv "$R17X_DIR/rc.part" "$R17X_DIR/rc"' \
  r17x "$@" </dev/null >/dev/null 2>"$d/proot.err" &
pid=$!
while [ ! -e "$d/rc" ]; do
  if ! kill -0 $pid 2>/dev/null && [ ! -e "$d/rc" ]; then sleep 0.2; [ -e "$d/rc" ] || { cat "$d/proot.err" >&2; rm -rf "$d"; exit 125; }; fi
  sleep 0.02
done
cat "$d/out"; cat "$d/err" >&2; rc=$(cat "$d/rc"); rm -rf "$d"; exit "$rc"
EOF
chmod +x "$BASE/bin/r17x.part" && mv "$BASE/bin/r17x.part" "$BASE/bin/r17x"

# 3. Python 3.12.3 (python-build-standalone via uv), at the path the venvs will name.
if [ ! -x "$D/python/$PY/bin/python3.12" ]; then
  UV_PYTHON_INSTALL_DIR=$D/python uv python install 3.12.3
  [ -x "$D/python/$PY/bin/python3.12" ] || { ls "$D/python"; echo "uv python layout differs"; exit 1; }
fi

pull() { "$BASE/tools/bin/rsync" -a --delete --bwlimit=$BW --timeout=120 "$SRC_MODULE/$1/" "$2/"; echo "pulled $1 ($(du -sh "$2" | cut -f1))"; }
relink() {  # venv copied from the H100 (base /usr/bin/python3.12) -> this Python
  for f in "$1/bin/python" "$1/bin/python3" "$1/bin/python3.12"; do
    if [ -L "$f" ] && readlink "$f" | grep -q '^/'; then ln -sfn /data/mw_finetune/python/$PY/bin/python3.12 "$f"; fi
  done
  sed -i "s#^home = .*#home = /data/mw_finetune/python/$PY/bin#; s#^executable = .*#executable = /data/mw_finetune/python/$PY/bin/python3.12#" "$1/pyvenv.cfg"
}

# 4. The server environment shared by DP and Fast-WAM: the Fast-WAM venv (torch 2.7.1+cu128),
#    DP's read-only venv on top of it, and the small R17 dependency island.
SRC_MODULE=${SRC}_mw
pull envs/fastwam "$D/envs/fastwam"; relink "$D/envs/fastwam"
pull qualification/python_deps312 "$D/qualification/python_deps312"

if [ "$MODEL" = dp ]; then
  pull dp/env "$D/dp/env"; relink "$D/dp/env"
  pull dp/libero_policy_100000 "$D/dp/libero_policy_100000"
  pull dp/mt50_policy_100000 "$D/dp/mt50_policy_100000"
  pull dp/clip_text "$D/dp/clip_text"
  # CUDA-graph self-check frames that formal_batch_server reads for LIBERO (256x256 wire
  # observations; the H100 has no copy). Copied once from weilandserver
  # /data/mw_finetune/qualification/assets/dp_libero_gpu_fix1_v1/ to $BASE/assets (sha256 4c480ecb...).
  mkdir -p "$D/qualification/assets/dp_libero_gpu_fix1_v1"
  cp "$BASE/assets/dp_libero_gpu_fix1_v1/wire_observations.npz" "$D/qualification/assets/dp_libero_gpu_fix1_v1/"
  PROBE=/data/mw_finetune/dp/env/bin/python
elif [ "$MODEL" = fastwam ]; then
  mkdir -p "$D/models" "$D/vendor" "$D/qualification/observations" "$DATA/xwam"
  pull models/fastwam_libero "$D/models/fastwam_libero"
  pull vendor/lerobot "$D/vendor/lerobot"
  pull vendor/FastWAM "$D/vendor/FastWAM"
  pull qualification/observations/libero_original256_fix2_v1 "$D/qualification/observations/libero_original256_fix2_v1"
  SRC_MODULE=${SRC}_xwam pull wan22_5b "$DATA/xwam/wan22_5b"
  PROBE=/data/mw_finetune/envs/fastwam/bin/python
else
  echo "unknown model $MODEL"; exit 2
fi

# 5. Probe under r17x: imports and the MIG slice.
"$BASE/bin/r17x" $PROBE -c 'import torch,sys; print(sys.version.split()[0], torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), [round(x/2**30,2) for x in torch.cuda.mem_get_info()])'
echo "setup $MODEL done: $(du -sh "$DATA" | cut -f1) under $DATA"
