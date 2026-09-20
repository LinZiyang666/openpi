#!/bin/bash
# h100: start the Cosmos Policy LIBERO server on <port> (default 23240). Idempotent (h100 double-exec: mkdir lock + has-session).
set -u
PORT=${1:-23240}
export HOME=/home/exouser; export PATH=/home/exouser/.local/bin:/usr/local/bin:/usr/bin:/bin
REPO=/data/cosmos/cosmos-policy
NV=$REPO/.venv/lib/python3.10/site-packages/nvidia
ENV="export HOME=/home/exouser PATH=$PATH HF_HOME=/data/cosmos/hf_home CUDA_HOME=$NV CUDNN_HOME=$NV/cudnn CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/cosmos/openpi_exp"
mkdir -p /tmp/cosmos
if ! mkdir "/tmp/cosmos/lock_srv$PORT" 2>/dev/null; then echo "srv$PORT claimed; leaving it"; exit 0; fi
if tmux has-session -t "cosmos_srv$PORT" 2>/dev/null; then echo "cosmos_srv$PORT exists"; exit 0; fi
tmux new -s "cosmos_srv$PORT" -d "$ENV; cd $REPO && .venv/bin/python -m exp.cosmos_nfe.serve_cosmos --port $PORT \
    --config cosmos_predict2_2b_480p_libero__inference_only \
    --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True --use_proprio True --normalize_proprio True --unnormalize_actions True \
    --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
    --t5_text_embeddings_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
    --trained_with_image_aug True --chunk_size 16 --num_open_loop_steps 16 \
    --task_suite_name libero_10 --randomize_seed False --seed 195 --use_variance_scale False --deterministic True \
    --ar_future_prediction False --ar_value_prediction False --use_jpeg_compression True --flip_images True \
    --num_denoising_steps_action 5 --num_denoising_steps_future_state 1 --num_denoising_steps_value 1 \
    2>&1 | tee /tmp/cosmos/srv$PORT.log"
echo "started cosmos_srv$PORT"
