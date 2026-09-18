#!/bin/bash
# Download the official DP datasets the x0 experiment still needs (kitchen, block pushing; image robomimic square on h100).
# usage: download_dp_data_x0.sh [lowdim|image]   (data root from dp_x0_env.sh)
set -u
source "$(dirname "$0")/dp_x0_env.sh"
what=${1:-lowdim}
cd $DP_DATA/data
if [ "$what" = lowdim ]; then
  [ -d kitchen/kitchen_demos_multitask ] || { wget -q --show-progress https://diffusion-policy.cs.columbia.edu/data/training/kitchen.zip && unzip -q kitchen.zip && rm kitchen.zip; }
  [ -d block_pushing ] || { wget -q --show-progress https://diffusion-policy.cs.columbia.edu/data/training/block_pushing.zip && unzip -q block_pushing.zip && rm block_pushing.zip; }
  ls kitchen/kitchen_demos_multitask block_pushing 2>&1 | head
else
  mkdir -p robomimic/datasets/square/mh robomimic/datasets/square/ph
  for s in mh ph; do [ -f robomimic/datasets/square/$s/image_abs.hdf5 ] || wget -q --show-progress -O robomimic/datasets/square/$s/image_abs.hdf5 https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image/square/$s/image_abs.hdf5; done
  ls -la robomimic/datasets/square/*/image_abs.hdf5
fi
echo "DOWNLOAD_X0_DONE $what"
