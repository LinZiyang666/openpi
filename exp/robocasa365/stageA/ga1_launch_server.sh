#!/bin/bash
# G-A1 (plan W3): teacher-only GR00T server at k denoise steps, weilandserver.
# Same shape as the W8 floor-arm server (--concurrent, port 23160, same ckpt),
# started through the isolated ksweep wrapper from the /tmp clone.
# Usage: ga1_launch_server.sh <k>
set -u
K=${1:?k}
export HOME=/home/weiland
ROOT=/tmp/openpi-stageA
CKPT=/home/weiland/ckpt_n15_robocasa_tp/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000
if ss -tlnH "sport = :23160" | grep -q .; then echo "port 23160 busy; refusing"; ss -tlnp "sport = :23160"; exit 1; fi
tmux kill-session -t ga1srv 2>/dev/null
tmux new-session -d -s ga1srv "cd $ROOT && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
  PYTHONPATH=/home/weiland/gr00t_n15:$ROOT/src:$ROOT \
  /home/weiland/gr00t_n15_venv/.venv/bin/python exp/robocasa365/serve_groot_n15_ksweep.py \
  --denoising-steps $K --port 23160 --concurrent --checkpoint $CKPT 2>&1 | tee /tmp/stageA/ga1_srv_k${K}.log"
echo "ga1srv spawned: k=$K on :23160 (log /tmp/stageA/ga1_srv_k${K}.log)"
