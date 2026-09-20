#!/bin/bash
# Host-local preparation chain of the x0 experiment after the subsets exist (plan §8 steps 2-3): freeze normalisers,
# generate the temporary pilot matrix, smoke train/eval/runner on real data, pilots for the requested cell patterns.
# usage: prep_x0.sh <pilot pattern> [pilot pattern ...]      (patterns are cell_id regexes over $X0_DATA/cells_pilot)
set -u
PATS=("$@")   # pilot patterns (saved before the runner loop below reuses `set --`)
source "$(dirname "$0")/dp_x0_env.sh"
cd $X0_CODE
OPS=$X0_CODE/exp/dp_nfe/ops
say() { echo "== $(date +%H:%M:%S) PREP $*"; }
say "normalizers"
bash $OPS/freeze_normalizers.sh all 2>&1 | grep -E "^==|NORMALIZER|_DONE|Error"
say "pilot matrix"
python -m exp.dp_nfe.x0_cells --tasks exp/dp_nfe/config/x0_multimodal/tasks.yaml --subsets $X0_DATA/subsets --out $X0_DATA/cells_pilot --budget-lowdim 50000 --budget-image 20000 --explore --image 2>&1 | tail -2
S=$X0_DATA/smoke; mkdir -p $S
if [ ! -f $S/train_lowdim/checkpoints/final.done ]; then
  say "smoke train lowdim (pusht U eps, 20 updates)"
  python -m exp.dp_nfe.smoke_x0 train --dp-root $DP_ROOT --cell $X0_DATA/cells_pilot/pusht_lowdim_U_epsilon_s42_B50k.yaml --out $S/train_lowdim --steps 20 2>&1 | grep -E "SMOKE|Error|Traceback|assert" | tail -3
fi
if [ -f $X0_DATA/cells_pilot/pusht_image_U_sample_s42_B20k.yaml ] && [ ! -f $S/train_image/checkpoints/final.done ]; then
  say "smoke train image (pusht U x0, 5 updates)"
  python -m exp.dp_nfe.smoke_x0 train --dp-root $DP_ROOT --cell $X0_DATA/cells_pilot/pusht_image_U_sample_s42_B20k.yaml --out $S/train_image --steps 5 2>&1 | grep -E "SMOKE|Error|Traceback|assert" | tail -3
fi
say "smoke eval (lowdim final, 2 eps x 3 samplers)"
python -m exp.dp_nfe.smoke_x0 eval --dp-root $DP_ROOT --ckpt $S/train_lowdim/checkpoints/final.ckpt --out $S/eval_lowdim 2>&1 | grep -E "SMOKE|Error|Traceback|assert" | tail -4
for rt in "kitchen_lowdim_abs task.env_runner.dataset_dir=$DP_DATA/data/kitchen" "blockpush_lowdim_seed_abs" "square_lowdim_abs task.env_runner.dataset_path=$DP_DATA/data/robomimic/datasets/square/mh/low_dim_abs.hdf5"; do
  set -- $rt; ro=""; [ -n "${2:-}" ] && ro="--runner-override $2"
  say "smoke runner $1"
  python -m exp.dp_nfe.smoke_x0 runner --dp-root $DP_ROOT --task $1 $ro > $S/runner_$1.log 2>&1
  grep -E "SMOKE|Error|Traceback|assert" $S/runner_$1.log | tail -2
done
for pat in "${PATS[@]}"; do
  say "pilot $pat"
  bash $OPS/pilot_x0.sh "$pat" 2>&1 | grep -E "^==|^PILOT|Error|Traceback"
done
echo "PREP_DONE $(date +%H:%M:%S)"
