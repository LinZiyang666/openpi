#!/bin/bash
# weilandserver: P0 of the x0 experiment (plan §2.1) -- the official eps checkpoints of square_mh / can_mh / pusht under the
# trailing_v1 protocol: DDIM k in {1,2,4,10,100} and the DDPM-100 anchor, 50 test episodes each (seeds 100000+), identity
# bound to the payload (variant=official). One lane per task; results at
# $X0_DATA/results_trailing/official_<task>/test_<sampler>_<k>/summary.json (verified summaries are skipped).
# usage: p0_official_wls.sh <task: square_mh|can_mh|pusht>
set -u
T=${1:?task}
source "$(dirname "$0")/dp_x0_env.sh"
export CUDA_VISIBLE_DEVICES=0
CK=$(ls $DP_DATA/ckpts/$T/epoch=*.ckpt | head -1)
case "$T" in
  square_mh) RO="--runner-override task.env_runner.dataset_path=$DP_DATA/data/robomimic/datasets/square/mh/low_dim_abs.hdf5" ;;
  can_mh)    RO="--runner-override task.env_runner.dataset_path=$DP_DATA/data/robomimic/datasets/can/mh/low_dim_abs.hdf5" ;;
  *)         RO="" ;;
esac
say() { echo "== $(date +%H:%M:%S) P0 $T $*"; }
say "ckpt=$CK"
for arm in "ddim 100" "ddpm 100" "ddim 10" "ddim 4" "ddim 2" "ddim 1"; do
  set -- $arm; OUT=$X0_DATA/results_trailing/official_$T/test_$1_$2
  if [ -f $OUT/summary.json ] && python - "$OUT/summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); sys.exit(0 if d.get("complete") is True and len(d.get("episodes", {})) == 50 else 1)
PY
  then say "$1-$2 present"; continue; fi
  mkdir -p $OUT
  python -m exp.dp_nfe.eval_dp_steps_v2 --dp-root $DP_ROOT -c "$CK" -o $OUT --sampler $1 --k $2 --split test --n-test 50 --n-envs 25 $RO > $OUT.log 2>&1
  say "$1-$2 rc=$? $(grep -ao 'DPSTEPS2 [A-Z]* .*' $OUT.log | tail -1 | cut -c1-160)"
done
echo "P0_DONE $T $(date +%H:%M:%S)"
