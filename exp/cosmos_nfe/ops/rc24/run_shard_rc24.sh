#!/bin/bash
# timan107/108: one RoboCasa-2024 client shard: <task> <k> <trials a-b> <gpu id> <ws url> [out root] [seed]
# Simulator + upstream eval loop run here; every policy query goes to a serve_cosmos --bench robocasa replica.
set -u
TASK=${1:?task}; K=${2:?k}; TR=${3:?trials a-b}; GPU=${4:?gpu}; REMOTE=${5:?ws url}; ROOT=${6:-/scratch/zixuans8/cosmos/results_rc24}; SEED=${7:-195}
REPO=/scratch/zixuans8/cosmos/cosmos-policy
NV=$REPO/.venv/lib/python3.10/site-packages/nvidia
HOMEDIR=/scratch/zixuans8/cosmos/home
OUT=$ROOT/k${K}/${TASK}_t${TR}.json
NAME=rc24_k${K}_${TASK}_t${TR}
LOG=/tmp/cosmos/$NAME.log
mkdir -p /tmp/cosmos "$ROOT/k${K}" $HOMEDIR /scratch/zixuans8/cosmos/logs_rc24
if [ -f "$OUT" ] && grep -q '"complete": true' "$OUT"; then echo "$NAME already complete; skip"; exit 0; fi
if tmux has-session -t "$NAME" 2>/dev/null; then echo "$NAME already running"; exit 0; fi
EGLV=""; [ -d /usr/share/glvnd/egl_vendor.d ] && EGLV="export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d;"
ENV="export HOME=$HOMEDIR PATH=/home/zixuans8/.local/bin:/usr/local/bin:/usr/bin:/bin HF_HOME=/scratch/zixuans8/cosmos/hf_home CUDA_HOME=$NV CUDNN_HOME=$NV/cudnn MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$GPU CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=/scratch/zixuans8/cosmos/openpi_exp; $EGLV"
tmux new -s "$NAME" -d "$ENV cd $REPO && .venv/bin/python -m exp.cosmos_nfe.run_robocasa_shard --trials $TR --results-json $OUT --remote $REMOTE \
    --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
    --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True --num_wrist_images 1 --use_proprio True --normalize_proprio True --unnormalize_actions True \
    --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
    --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
    --trained_with_image_aug True --chunk_size 32 --num_open_loop_steps 16 \
    --task_name $TASK --num_trials_per_task 50 --seed $SEED --randomize_seed False --deterministic True \
    --use_variance_scale False --use_jpeg_compression True --flip_images True \
    --local_log_dir /scratch/zixuans8/cosmos/logs_rc24 --run_id_note k${K}-t${TR}-remote \
    --num_denoising_steps_action $K --num_denoising_steps_future_state 1 --num_denoising_steps_value 1 --data_collection False \
    > $LOG 2>&1; echo SHARD_EXIT=\$? >> $LOG"
echo "started $NAME gpu=$GPU -> $OUT"
