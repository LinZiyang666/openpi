#!/bin/bash
# Build the frozen U/M/held-out/training-pool subsets of the four core tasks (plan §3, §8 entry 1) inside the DP env.
# usage: build_subsets.sh [task ...]   (default: pusht blockpush kitchen square_mh); square image export only when
# image_abs.hdf5 exists on this host. Manifests: $X0_DATA/subsets/<task>_subset_manifest.json.
set -u
source "$(dirname "$0")/dp_x0_env.sh"
cd $X0_CODE
D=$DP_DATA/data
say() { echo "== $(date +%H:%M:%S) $*"; }
for t in ${@:-pusht blockpush kitchen square_mh}; do
  case $t in
    pusht)     args="--src $D/pusht/pusht_cchi_v7_replay.zarr --image-src $D/pusht/pusht_cchi_v7_replay.zarr" ;;
    blockpush) args="--src $D/block_pushing/multimodal_push_seed_abs.zarr" ;;
    kitchen)   args="--src $D/kitchen/kitchen_demos_multitask" ;;
    square_mh) args="--src $D/robomimic/datasets/square/mh/low_dim_abs.hdf5"; [ -f $D/robomimic/datasets/square/mh/image_abs.hdf5 ] && args="$args --image-src $D/robomimic/datasets/square/mh/image_abs.hdf5" ;;
  esac
  say "$t: $args"
  python -m exp.dp_nfe.mode_filter_datasets --task $t $args --out $X0_DATA/subsets 2>&1 | tail -3
  say "$t rc=${PIPESTATUS[0]}"
done
ls $X0_DATA/subsets
echo "SUBSETS_DONE $(date +%H:%M:%S)"
