#!/bin/bash
# Start the X-WAM policy broker (once) and one policy server on this host (h100 or weilandserver; paths by hostname).
# usage: [XWAM_CKPT=robotwin_sft] serve_xwam.sh <action_denoise_steps> [denoise_steps=50] [gpu=0] [frontend port=23270] [backend port=23271] [server idx=0]
# The broker's frontend port is what the robocasa clients connect to. Idempotent (has-session).
set -u
AK=${1:?action denoise steps}; VK=${2:-50}; GPU=${3:-0}; FP=${4:-23270}; BP=${5:-23271}; IDX=${6:-0}
case "$(hostname)" in
  weilandserver*) CODE=/home/weiland/x-wam; DATA=/data/xwam; UHOME=/home/weiland ;;
  *) CODE=/data/xwam/X-WAM; DATA=/data/xwam; UHOME=/home/exouser ;;
esac
export HOME=$UHOME; export PATH=$UHOME/.local/bin:/usr/local/bin:/usr/bin:/bin
BA=${BROKER_ADDR:-localhost}   # a server on another host joins the h100 broker: BROKER_ADDR=149.165.153.233
CK=${XWAM_CKPT:-robocasa_sft}  # fine-tuned checkpoint dir under $DATA/checkpoints (robocasa_sft | robotwin_sft)
mkdir -p /tmp/xwam
[ "$BA" = "localhost" ] && { tmux has-session -t xwam_broker 2>/dev/null || tmux new -s xwam_broker -d "cd $CODE && .venv/bin/python evaluation/policy_broker.py --frontend_port $FP --backend_port $BP 2>&1 | tee -a /tmp/xwam/broker.log"; }
N=xwam_srv${IDX}
tmux has-session -t $N 2>/dev/null && { echo "$N exists"; exit 0; }
tmux new -s $N -d "cd $CODE && export HF_HOME=$DATA/hf_home CUDA_VISIBLE_DEVICES=$GPU && .venv/bin/python evaluation/policy_server.py --exp_path $DATA/checkpoints/$CK --wan_checkpoint_dir $DATA/wan22_5b --broker_addr $BA --broker_port $BP --denoise_steps $VK --action_denoise_steps $AK 2>&1 | tee /tmp/xwam/srv$IDX.log"
echo "started $N (ckpt=$CK action_steps=$AK video_steps=$VK gpu=$GPU broker $FP/$BP) on $(hostname)"
