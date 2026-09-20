#!/bin/bash
# Plan §4.3 conditional-distribution diagnostics for every *finished* run matching a cell regex: the held-out / training-
# pool dataset configs are derived from the run's resolved_config.yaml + identity.json (paths of this host), the frozen
# normaliser from the subsets directory, and `dispersion_index sample` writes $X0_DATA/diagnostics/<cell>_ddim_<k>.json
# (skipped when present). eps / x0 and U / M cells share the history set and noise keys via --sampling-seed.
# usage: x0_dispersion.sh <cell regex> [k list=100,1] [n-histories=64] [n-samples=64]   (OUT_DIR overrides diagnostics/)
set -u
PAT=${1:?cell regex}; KS=${2:-100,1}; NH=${3:-64}; NS=${4:-64}
source "$(dirname "$0")/dp_x0_env.sh"
cd $X0_CODE
OUT=${OUT_DIR:-$X0_DATA/diagnostics}; mkdir -p $OUT
say() { echo "== $(date +%H:%M:%S) DISP $*"; }
n=0; fail=0
for run in $(ls -d $X0_DATA/runs/*/ | sed 's:/$::'); do
  cell=$(basename $run)
  [[ $cell =~ $PAT ]] || continue
  [ -f $run/checkpoints/final.done ] || { say "$cell: not finished, skip"; continue; }
  cfgs=$(python - "$run" "$X0_DATA/subsets" <<'PY'
import json, pathlib, sys, yaml
run, subsets = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
cfg = yaml.safe_load((run / "resolved_config.yaml").read_text()); ident = json.loads((run / "identity.json").read_text())
held = dict(cfg["x0"]["heldout_dataset"]); held["val_ratio"] = 0
pool = dict(cfg["task"]["dataset"]); pool["val_ratio"] = 0
key = next(k for k in ("zarr_path", "dataset_dir", "dataset_path") if pool.get(k))
sp = pathlib.Path(ident["subset_path"]); name = sp.name
for tag in ("_U.", "_M.", "_U", "_M"):
    if tag in name:
        name = name.replace(tag, "_trainpool." if tag.endswith(".") else "_trainpool", 1); break
pool[key] = str(sp.with_name(name))
norm = subsets / f"{ident['task_name']}_{ident['modality']}_normalizer.pt"
print(json.dumps(held)); print(json.dumps(pool)); print(norm); print(pool[key])
PY
) || { say "$cell: config derivation failed"; fail=$((fail+1)); continue; }
  held=$(sed -n 1p <<<"$cfgs"); pool=$(sed -n 2p <<<"$cfgs"); norm=$(sed -n 3p <<<"$cfgs"); poolpath=$(sed -n 4p <<<"$cfgs")
  [ -e "$poolpath" ] || { say "$cell: trainpool $poolpath missing"; fail=$((fail+1)); continue; }
  [ -f "$norm" ] || { say "$cell: normalizer $norm missing"; fail=$((fail+1)); continue; }
  for k in ${KS//,/ }; do
    out=$OUT/${cell}_ddim_$k.json
    [ -f $out ] && { say "$cell k=$k: exists"; continue; }
    say "$cell k=$k"
    python -m exp.dp_nfe.analysis.dispersion_index sample --dp-root $DP_ROOT -c $run/checkpoints/final.ckpt \
      --heldout-cfg "$held" --trainpool-cfg "$pool" --normalizer "$norm" -o $out --n-histories $NH --n-samples $NS \
      --sampler ddim --k $k --sampling-seed 0 2>&1 | grep -E "DISPERSION|Error|Traceback|assert" | tail -3
    [ -f $out ] && n=$((n+1)) || fail=$((fail+1))
  done
done
echo "DISPERSION_DONE written=$n failed=$fail $(date +%H:%M:%S)"
