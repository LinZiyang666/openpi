#!/bin/bash
# timan108: one X-WAM robocasa client = one task (rank i of 24) x <n evals>. usage: run_client_xwam.sh <task rank> <n evals> <broker host> <frontend port> <gpu> [tag] [out root]
set -u
RANK=${1:?rank}; NE=${2:?n evals}; BH=${3:?broker host}; FP=${4:?port}; GPU=${5:?gpu}; TAG=${6:-run}; OUT=${7:-/scratch/zixuans8/xwam/results_$TAG}
CODE=/scratch/zixuans8/xwam/X-WAM
NAME=xwam_${TAG}_r${RANK}
mkdir -p /tmp/xwam $OUT
tmux has-session -t $NAME 2>/dev/null && { echo "$NAME running"; exit 0; }
EGLV=""; [ -d /usr/share/glvnd/egl_vendor.d ] && EGLV="export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d;"
tmux new -s $NAME -d "export HOME=/home/zixuans8 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$GPU CUDA_VISIBLE_DEVICES=$GPU; $EGLV cd $CODE && .venv/bin/python evaluation/robocasa_client.py --env_global_rank $RANK --world_size 24 --num_evals_per_worker $NE --server_addr $BH --server_port $FP --save_root_dir $OUT > /tmp/xwam/$NAME.log 2>&1; echo CLIENT_EXIT=\$? >> /tmp/xwam/$NAME.log"
echo "started $NAME -> $OUT"
