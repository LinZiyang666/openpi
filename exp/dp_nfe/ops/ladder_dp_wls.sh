#!/bin/bash
# weilandserver: Diffusion Policy step ladder over the official checkpoints. For each task x (scheduler, steps):
# eval_dp_steps.py with n_test seeded episodes, results under /data/dp/results/<task>/<sched><steps>/summary.json (skipped
# when present, so the ladder resumes). usage: ladder_dp_wls.sh [n_test=50] [n_envs=25]
set -u
NT=${1:-50}; NENV=${2:-25}
export HOME=/home/weiland
D=/home/weiland/dp; DATA=/data/dp; RES=$DATA/results; LOG=/tmp/dp/ladder.log
export PATH=$D/env/bin:/usr/local/bin:/usr/bin:/bin MUJOCO_PY_MUJOCO_PATH=$D/mujoco210 LD_LIBRARY_PATH=$D/mujoco210/bin:$D/env/lib:/usr/lib/nvidia MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0
mkdir -p $RES /tmp/dp
say() { echo "$(date +%H:%M:%S) $*" | tee -a $LOG; }
ds() {  # dataset (env metadata) for a robomimic task
  case "$1" in
    square_mh) echo $DATA/data/robomimic/datasets/square/mh/low_dim_abs.hdf5 ;;
    can_mh) echo $DATA/data/robomimic/datasets/can/mh/low_dim_abs.hdf5 ;;
    transport_mh) echo $DATA/data/robomimic/datasets/transport/mh/low_dim_abs.hdf5 ;;
    tool_hang_ph) echo $DATA/data/robomimic/datasets/tool_hang/ph/low_dim_abs.hdf5 ;;
    *) echo "" ;;
  esac
}
cd $D/diffusion_policy
for T in square_mh can_mh tool_hang_ph transport_mh pusht; do
  CK=$(ls $DATA/ckpts/$T/epoch=*.ckpt 2>/dev/null | head -1); [ -z "$CK" ] && { say "DP $T: no checkpoint"; continue; }
  DSP=$(ds $T); if [ -n "$DSP" ] && [ ! -f "$DSP" ]; then say "DP $T: dataset $DSP missing"; continue; fi
  for cfg in ddpm:100 ddim:10 ddim:4 ddim:2 ddim:1 ddpm:10; do
    S=${cfg%%:*}; K=${cfg##*:}; OUT=$RES/$T/$S$K
    [ -f $OUT/summary.json ] && { say "DP $T $S$K: present"; continue; }
    say "DP $T $S$K UP"
    mkdir -p "$(dirname $OUT)"
    python /tmp/dp/eval_dp_steps.py -c "$CK" -o $OUT --steps $K --scheduler $S --n_test $NT --n_envs $NENV ${DSP:+--dataset_path $DSP} > $OUT.log 2>&1
    rc=$?
    grep -h "DPSTEPS" $OUT.log | tail -3 | tee -a $LOG
    [ $rc -ne 0 ] && say "DP $T $S$K FAIL rc=$rc (see $OUT.log)"
  done
done
say "DP LADDER FINISHED"
