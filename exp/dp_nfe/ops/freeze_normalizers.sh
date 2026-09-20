#!/bin/bash
# Fit + freeze the shared training-pool normaliser of every (task, modality) whose subsets exist on this host
# (plan §3.2, §8 entry 2). Idempotent: an existing sidecar whose source dataset_sha256 equals the manifest's trainpool
# hash is kept. usage: freeze_normalizers.sh [lowdim|image|all]
set -u
source "$(dirname "$0")/dp_x0_env.sh"
cd $X0_CODE
S=$X0_DATA/subsets; what=${1:-all}
say() { echo "== $(date +%H:%M:%S) $*"; }
freeze() {  # freeze <task> <modality> <workspace cfg> <dp task> <path key> <trainpool path>
  local t=$1 m=$2 ws=$3 task=$4 key=$5 pool=$6 out=$S/${1}_${2}_normalizer.pt
  [ -f $S/${t}_subset_manifest.json ] || { say "$t: no subset manifest, skip"; return; }
  python -c "import json,sys;sys.exit(0 if json.load(open('$S/${t}_subset_manifest.json'))['selection']['usable'] else 1)" || { say "$t: subsets unusable (no U/M), skip"; return; }
  [ -e "$pool" ] || { say "$t/$m: trainpool $pool missing, skip"; return; }
  want=$(python -c "import json;m=json.load(open('$S/${t}_subset_manifest.json'));print((m['image_exports'] if '$m'=='image' else m['exports']).get('trainpool',''))")
  have=$(python -c "import json,os;p='$out.json';print(json.load(open(p))['source'].get('dataset_sha256','') if os.path.exists(p) else '')")
  if [ -n "$want" ] && [ "$want" = "$have" ]; then say "$t/$m: frozen already ($want)"; return; fi
  say "$t/$m: fit on $pool"
  python -m exp.dp_nfe.x0_normalizer --dp-root $DP_ROOT --workspace-config $ws --task $task --override "$key=$pool" --out $out 2>&1 | grep -E "NORMALIZER|Error|error" | tail -2
}
if [ "$what" != image ]; then
  freeze pusht     lowdim train_diffusion_unet_lowdim_workspace pusht_lowdim          task.dataset.zarr_path    $S/pusht_trainpool.zarr
  freeze blockpush lowdim train_diffusion_unet_lowdim_workspace blockpush_lowdim_seed_abs task.dataset.zarr_path $S/blockpush_trainpool.zarr
  freeze kitchen   lowdim train_diffusion_unet_lowdim_workspace kitchen_lowdim        task.dataset.dataset_dir  $S/kitchen_trainpool
  freeze square_mh lowdim train_diffusion_unet_lowdim_workspace square_lowdim_abs     task.dataset.dataset_path $S/square_mh_trainpool.hdf5
fi
if [ "$what" != lowdim ]; then
  freeze pusht     image  train_diffusion_unet_hybrid_workspace pusht_image           task.dataset.zarr_path    $S/pusht_trainpool.zarr
  freeze square_mh image  train_diffusion_unet_hybrid_workspace square_image_abs      task.dataset.dataset_path $S/square_mh_image_trainpool.hdf5
fi
ls $S/*_normalizer.pt* 2>/dev/null
echo "NORMALIZERS_DONE $what $(date +%H:%M:%S)"
