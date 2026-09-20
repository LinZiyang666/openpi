#!/bin/bash
# Download the official DP datasets the x0 experiment needs into $DP_DATA/data (layout of the DP README: unzip inside
# data/). Idempotent: a dataset that is already present is skipped; zips are deleted after extraction; sha256 of every
# hdf5/npy/zarr root is appended to $DP_DATA/data/x0_data_sha256.txt.
#   usage: download_dp_data_x0.sh lowdim   # kitchen (778 MB), block_pushing (11 MB), pusht (31 MB), robomimic_lowdim (1.9 GB)
#          download_dp_data_x0.sh image    # robomimic_image.zip (84.7 GB, all tasks) -> keeps only square/{mh,ph}/image_abs.hdf5
set -u -o pipefail
source "$(dirname "$0")/dp_x0_env.sh"
what=${1:-lowdim}
URL=https://diffusion-policy.cs.columbia.edu/data/training
cd $DP_DATA/data || exit 1
say() { echo "== $(date +%H:%M:%S) $*"; }
get() {  # get <zip name> <present test path>
  local z=$1 probe=$2
  if [ -e "$probe" ]; then say "$z: present ($probe)"; return 0; fi
  say "$z: download"; wget -q -O "$z" "$URL/$z" || { echo "DOWNLOAD_FAIL $z"; return 1; }
  say "$z: unzip"; unzip -q -o "$z" || { echo "UNZIP_FAIL $z"; return 1; }
  rm "$z"; return 0
}
if [ "$what" = lowdim ]; then
  get kitchen.zip kitchen/kitchen_demos_multitask/observations_seq.npy || exit 1
  get block_pushing.zip block_pushing/multimodal_push_seed_abs.zarr || exit 1
  get pusht.zip pusht/pusht_cchi_v7_replay.zarr || exit 1
  get robomimic_lowdim.zip robomimic/datasets/square/ph/low_dim_abs.hdf5 || exit 1
  ls kitchen/kitchen_demos_multitask block_pushing pusht robomimic/datasets/*/* 2>&1 | head -40
elif [ "$what" = image ]; then
  if [ -f robomimic/datasets/square/mh/image_abs.hdf5 ] && [ -f robomimic/datasets/square/ph/image_abs.hdf5 ]; then say "square image present"; else
    [ -f robomimic_image.zip ] || { say "robomimic_image.zip: download (84.7 GB)"; wget -q -O robomimic_image.zip "$URL/robomimic_image.zip" || { echo "DOWNLOAD_FAIL robomimic_image.zip"; exit 1; }; }
    say "members:"; unzip -l robomimic_image.zip | grep -E "square/(mh|ph)/image_abs.hdf5" | head
    members=$(unzip -l robomimic_image.zip | grep -oE "[^ ]*square/(mh|ph)/image_abs.hdf5")
    say "extract square only"; unzip -q -o robomimic_image.zip $members || { echo "UNZIP_FAIL robomimic_image.zip"; exit 1; }
    ls -la robomimic/datasets/square/*/image_abs.hdf5 && rm robomimic_image.zip
  fi
fi
say "sha256 ledger"
{ for f in kitchen/kitchen_demos_multitask/observations_seq.npy kitchen/kitchen_demos_multitask/actions_seq.npy kitchen/kitchen_demos_multitask/existence_mask.npy robomimic/datasets/*/*/low_dim_abs.hdf5 robomimic/datasets/square/*/image_abs.hdf5; do [ -f "$f" ] && sha256sum "$f"; done; } 2>/dev/null | sort -k2 > x0_data_sha256.txt
for z in pusht/pusht_cchi_v7_replay.zarr block_pushing/multimodal_push_seed_abs.zarr; do [ -d "$z" ] && echo "$(find "$z" -type f | sort | xargs cat | sha256sum | cut -d' ' -f1)  $z (dir content)" >> x0_data_sha256.txt; done
wc -l x0_data_sha256.txt; df -h $DP_DATA | tail -1
echo "DOWNLOAD_X0_DONE $what $(date +%H:%M:%S)"
