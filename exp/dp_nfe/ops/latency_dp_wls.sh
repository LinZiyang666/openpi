#!/bin/bash
# weilandserver: batch-1 predict_action latency of every DP ladder cell on an idle 4090 (the ladder's own numbers were
# taken while the X-WAM server shared the GPU). Writes /data/dp/results/<task>/<sched><steps>/latency.json.
# usage: latency_dp_wls.sh
set -u
export HOME=/home/weiland
D=/home/weiland/dp; DATA=/data/dp; RES=$DATA/results; LOG=/tmp/dp/latency.log
export PATH=$D/env/bin:/usr/local/bin:/usr/bin:/bin MUJOCO_PY_MUJOCO_PATH=$D/mujoco210 LD_LIBRARY_PATH=$D/mujoco210/bin:$D/env/lib:/usr/lib/nvidia MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0
say() { echo "$(date +%H:%M:%S) $*" | tee -a $LOG; }
cd $D/diffusion_policy
say "DP LATENCY UP gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
for T in square_mh can_mh tool_hang_ph transport_mh pusht; do
  CK=$(ls $DATA/ckpts/$T/epoch=*.ckpt 2>/dev/null | head -1); [ -z "$CK" ] && continue
  for cfg in ddpm:100 ddim:10 ddim:4 ddim:2 ddim:1 ddpm:10; do
    S=${cfg%%:*}; K=${cfg##*:}; OUT=$RES/$T/$S$K; mkdir -p $OUT
    python /tmp/dp/eval_dp_steps.py -c "$CK" -o $OUT --steps $K --scheduler $S --latency_only > $OUT.latency.log 2>&1 || say "DP $T $S$K LATENCY FAIL"
    grep -h "DPSTEPS LATENCY DONE" $OUT.latency.log | sed "s/^/$T /" | tee -a $LOG
  done
done
say "DP LATENCY FINISHED"
