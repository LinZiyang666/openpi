#!/bin/bash
# Start one Cosmos Policy RoboCasa-2024 server replica on <port> (h100 or weilandserver; host paths by hostname).
# usage: serve_rc24.sh <port> [default k]   Idempotent (mkdir lock + has-session; h100 tether exec double-runs).
set -u
PORT=${1:?port}; KDEF=${2:-5}
case "$(hostname)" in
  weilandserver*) REPO=/home/weiland/cosmos-policy; HF=/data/cosmos/hf_home; HOMEDIR=/home/weiland/cosmos_home; EXP=/home/weiland/cosmos_exp; UHOME=/home/weiland ;;
  *) REPO=/data/cosmos/cosmos-policy; HF=/data/cosmos/hf_home; HOMEDIR=/home/exouser; EXP=/data/cosmos/openpi_exp; UHOME=/home/exouser ;;
esac
NV=$REPO/.venv/lib/python3.10/site-packages/nvidia
export HOME=$UHOME; export PATH=$UHOME/.local/bin:/usr/local/bin:/usr/bin:/bin
ENV="export HOME=$HOMEDIR PATH=$PATH HF_HOME=$HF CUDA_HOME=$NV CUDNN_HOME=$NV/cudnn CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$EXP"
mkdir -p /tmp/cosmos
if ! mkdir "/tmp/cosmos/lock_srv$PORT" 2>/dev/null; then echo "srv$PORT claimed; leaving it"; exit 0; fi
if tmux has-session -t "cosmos_srv$PORT" 2>/dev/null; then echo "cosmos_srv$PORT exists"; exit 0; fi
tmux new -s "cosmos_srv$PORT" -d "$ENV; cd $REPO && .venv/bin/python -m exp.cosmos_nfe.serve_cosmos --bench robocasa --port $PORT --no-future-decode 1 --cuda-graph 1 \
    --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
    --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True --num_wrist_images 1 --use_proprio True --normalize_proprio True --unnormalize_actions True \
    --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
    --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
    --trained_with_image_aug True --chunk_size 32 --num_open_loop_steps 16 \
    --task_name TurnOffMicrowave --num_trials_per_task 50 --seed 195 --randomize_seed False --deterministic True \
    --use_variance_scale False --use_jpeg_compression True --flip_images True \
    --num_denoising_steps_action $KDEF --num_denoising_steps_future_state 1 --num_denoising_steps_value 1 --data_collection False \
    2>&1 | tee /tmp/cosmos/srv$PORT.log"
echo "started cosmos_srv$PORT (robocasa, default k=$KDEF) on $(hostname)"
