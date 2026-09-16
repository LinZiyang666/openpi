#!/usr/bin/env bash
# Direct LIBERO clients for one (suite, k): S shard processes x 10 workers (timan107).
#
# usage: launch_clients.sh <suite> <k> <host> <ports csv> <shards S> <apool_dir> <filter_dir> <out_dir> <resize> [tag]
#
# Shard s connects to ports[s % nports], renders on EGL device NFE_GPUS[s % len]
# (default: every GPU on the box; set NFE_GPUS=3,4,5,6,7 to keep off cards
# another line's workers already fill) and runs
# the episodes of shard_<s>.json (make_shards.py) over all ten tasks with one
# worker per task. Results: <out_dir>/<suite>_k<k>_s<s>.json; log
# /tmp/nfe/<session>.log ending in NFECLI_EXIT=<code>. The exit marker uses a
# POSIX redirect because tmux on timan107 runs the command under dash.
set -eu
SUITE=${1:?suite}
K=${2:?k}
HOST=${3:?server host}
PORTS=${4:?ports csv}
S=${5:?shards}
APOOL=${6:?apool dir}
FILTER=${7:?filter dir}
OUT=${8:?out dir}
RESIZE=${9:?resize size}
TAG=${10:-run}
REPO=${NFE_CLIENT_REPO:-/scratch/zixuans8/openpi_lg}
PY=${NFE_CLIENT_PY:-/scratch/zixuans8/libero_sim/bin/python}
WORKERS=${NFE_WORKERS:-10}
TRIALS=${NFE_TRIALS:-50}
export HOME=${NFE_HOME:-/home/zixuans8}
export LIBERO_CONFIG_PATH=$HOME/.libero
export CONDA_PREFIX=$(dirname "$(dirname "$PY")")
export PATH=$CONDA_PREFIX/bin:$PATH
export MUJOCO_GL=egl
# timan108 stages its own NVIDIA EGL user-space under the conda prefix (activate.d hook);
# we call bin/python directly, so apply the hook by hand where it exists.
HOOK=$CONDA_PREFIX/etc/conda/activate.d/nvidia_egl.sh
EGLENV=""
if [ -f "$HOOK" ]; then
  . "$HOOK"
  # An empty __EGL_VENDOR_LIBRARY_DIRS makes glvnd find zero EGL devices, so only
  # forward the two variables when the hook actually set them.
  EGLENV="__EGL_VENDOR_LIBRARY_DIRS=$__EGL_VENDOR_LIBRARY_DIRS LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
fi
[ -d "$APOOL" ] || { echo "missing apool $APOOL"; exit 1; }
[ -f "$FILTER/shard_0.json" ] || { echo "missing filters under $FILTER"; exit 1; }
mkdir -p "$OUT" /tmp/nfe
GPUS=${NFE_GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')}
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
IFS=',' read -r -a PORT_ARR <<< "$PORTS"
NP=${#PORT_ARR[@]}
for s in $(seq 0 $((S-1))); do tmux kill-session -t "nfecli_${TAG}_${SUITE}_k${K}_s$s" 2>/dev/null || true; done
sleep 1
for s in $(seq 0 $((S-1))); do
  PORT=${PORT_ARR[$((s % NP))]}
  G=${GPU_ARR[$((s % NGPU))]}
  NAME="nfecli_${TAG}_${SUITE}_k${K}_s$s"
  RES="$OUT/${SUITE}_k${K}_s$s.json"
  [ -f "$RES" ] && { echo "$RES exists; refusing to overwrite (move it first)"; exit 1; }
  tmux new -s "$NAME" -d "cd $REPO && $EGLENV MUJOCO_EGL_DEVICE_ID=$G PYTHONPATH=$REPO/packages/openpi-client/src:$REPO:$REPO/src \
    $PY examples/libero/main.py --host $HOST --port $PORT --task-suite-name $SUITE \
    --num-trials-per-task $TRIALS --num-workers $WORKERS --init-states-dir $APOOL \
    --episode-filter $FILTER/shard_$s.json --replan-steps 5 --resize-size $RESIZE \
    --save-episode-results --episode-results-path $RES > /tmp/nfe/$NAME.log 2>&1; \
    echo NFECLI_EXIT=\$? >> /tmp/nfe/$NAME.log"
  sleep 6
done
sleep 3
echo "client sessions: $(tmux ls 2>/dev/null | grep -c "^nfecli_${TAG}_${SUITE}_k${K}_")"
