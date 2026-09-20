#!/bin/bash
# One shard of the Cosmos Policy LIBERO ladder on weilandserver: <suite> <k> <shard idx> <task ids csv> [seed] [trials]
# Official eval command (LIBERO.md) with the task subset + results JSON from exp/cosmos_nfe/run_libero_shard.py.
set -u
SUITE=${1:?suite}; K=${2:?k}; S=${3:?shard idx}; TIDS=${4:?task ids}; SEED=${5:-195}; TRIALS=${6:-50}
export HOME=/home/weiland/cosmos_home; export PATH=/home/weiland/.local/bin:/usr/local/bin:/usr/bin:/bin   # private HOME on the SSD: own ~/.libero config + assets, HF token; not the py3.8 libero_sim one
export UV_CACHE_DIR=/home/weiland/.cache/uv HF_HOME=/data/cosmos/hf_home   # code+venv on SSD, model weights on /data
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/weiland/cosmos_exp
REPO=/home/weiland/cosmos-policy
ROOT=${COSMOS_RESULTS_ROOT:-/data/cosmos/results}; TAG=${COSMOS_TAG:-}
OUT=$ROOT/${SUITE}/k${K}/shard${S}.json
LOG=/tmp/cosmos/${SUITE}_k${K}_s${S}${TAG}.log
NAME=cosmos_${SUITE}_k${K}_s${S}${TAG}
mkdir -p /tmp/cosmos $ROOT/${SUITE}/k${K} /data/cosmos/logs $HOME
if [ -f "$OUT" ] && grep -q '"complete": true' "$OUT"; then echo "$NAME already complete; skip"; exit 0; fi
if tmux has-session -t "$NAME" 2>/dev/null; then echo "$NAME already running"; exit 0; fi
# Env is passed inside the tmux command: a tmux server started by another session does not inherit this shell.
ENV="export HOME=$HOME PATH=$PATH HF_HOME=$HF_HOME MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH"
tmux new -s "$NAME" -d "$ENV; cd $REPO && .venv/bin/python -m exp.cosmos_nfe.run_libero_shard --task-ids $TIDS --results-json $OUT \
    --config cosmos_predict2_2b_480p_libero__inference_only \
    --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True --use_proprio True --normalize_proprio True --unnormalize_actions True \
    --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
    --t5_text_embeddings_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
    --trained_with_image_aug True --chunk_size 16 --num_open_loop_steps 16 \
    --task_suite_name $SUITE --num_trials_per_task $TRIALS \
    --local_log_dir /data/cosmos/logs --randomize_seed False --data_collection False \
    --available_gpus 0 --seed $SEED --use_variance_scale False --deterministic True \
    --run_id_note k${K}-s${S}-seed${SEED}${TAG} \
    --ar_future_prediction False --ar_value_prediction False --use_jpeg_compression True --flip_images True \
    --num_denoising_steps_action $K --num_denoising_steps_future_state 1 --num_denoising_steps_value 1 \
    > $LOG 2>&1; echo SHARD_EXIT=\$? >> $LOG"
echo "started $NAME tasks=$TIDS -> $OUT"
