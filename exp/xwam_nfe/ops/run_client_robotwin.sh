#!/bin/bash
# weilandserver / h100 / timan107 (paths by hostname): one X-WAM RoboTwin 2.0 client = one task x <n evals> (demo_randomized, unseen
# instructions, seed 0 -> the same seed sequence for every k).
# usage: run_client_robotwin.sh <task> <n evals> <broker host> <frontend port> [gpu=0|auto] [tag] [out root]
# Result: <out root>/<task>/_result.txt (success rate); log /tmp/robotwin/<session>.log. Idempotent (has-session).
# Start-ups are serialised 25 s apart through a flock, the Vulkan ICD is pinned to NVIDIA, every client gets its own NVIDIA
# shader disk cache (__GL_SHADER_DISK_CACHE_PATH), python -u -X faulthandler so a crash leaves a trace, env dumped for forensics.
# XWAM_NO_VIDEO=1 (default): the patched deploy_policy.py skips the 8 mp4-only 3-camera renders per query (policy inputs
# unchanged; the per-episode mp4 is not written) -- rendering was the client bottleneck.
set -u
TASK=${1:?task}; NE=${2:?n evals}; BH=${3:?broker host}; FP=${4:?port}; GPU=${5:-0}; TAG=${6:-run}
case "$(hostname)" in
  weilandserver*) ROOT=/home/weiland/robotwin; DATA=/data/robotwin; UHOME=/home/weiland ;;
  timan107*) ROOT=/srv/local/zixuans8/robotwin; DATA=/srv/local/zixuans8/robotwin; UHOME=/home/zixuans8 ;;
  *) ROOT=/data/xwam/robotwin; DATA=/data/xwam/robotwin_data; UHOME=/home/exouser ;;
esac
# GPU "auto": the card with the least memory in use (timan107: 8x GTX 1080 8 GB, one client per card)
[ "$GPU" = auto ] && GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
OUT=${7:-$DATA/results_$TAG}
ENV=$ROOT/mamba/envs/rt; CODE=$ROOT/X-WAM
NAME=rtw_${TAG}_${TASK}
mkdir -p /tmp/robotwin "$OUT" /tmp/robotwin/glcache/$NAME
tmux has-session -t "=$NAME" 2>/dev/null && { echo "$NAME running"; exit 0; }   # "=" forces an exact match: plain -t prefix-matched rtw_k1_shake_bottle_horizontally for shake_bottle
tmux new -s $NAME -d "export HOME=$UHOME PATH=$ENV/bin:/usr/local/bin:/usr/bin:/bin CUDA_VISIBLE_DEVICES=$GPU CUDA_HOME=$ENV HF_HOME=$DATA/cache/hf XDG_CACHE_HOME=$DATA/cache/xdg VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json __GL_SHADER_DISK_CACHE_PATH=/tmp/robotwin/glcache/$NAME XWAM_NO_VIDEO=${XWAM_NO_VIDEO:-1} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; mkdir -p $DATA/cache/xdg; env > /tmp/robotwin/env_$NAME.txt; cd $CODE && flock /tmp/robotwin/launch.lock -c \"sleep 25\" && $ENV/bin/python -u -X faulthandler evaluation/robotwin_client.py --task_name $TASK --task_config demo_randomized --num_evals_per_worker $NE --server_addr $BH --server_port $FP --save_root_dir $OUT > /tmp/robotwin/$NAME.log 2>&1; echo CLIENT_EXIT=\$? >> /tmp/robotwin/$NAME.log"
echo "started $NAME -> $OUT/$TASK on $(hostname)"
