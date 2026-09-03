#!/bin/bash
# build_k3_tables.sh — rebuild both suites' tau1 shadow tables with the extra 0.5 warm tier (y_tau5).
set -uo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
export HOME=/home/weiland
R=/data/openpi_dispatch
CKPT=/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
PY=/home/weiland/openpi/.venv/bin/python
cd $R
export PYTHONPATH=$R/src:$R
step() { echo "=== [$(date +%H:%M:%S)] $*"; }
SUITE=$1
for SUITE in $SUITE; do
  B=/tmp/dsp_shared/rit_pareto/$SUITE
  PKL=/home/weiland/openpi/exp/common/data/cache_artifacts/$SUITE/cp1_spatial_pool_16.pkl
  step table k3 $SUITE
  $PY -m exp.dispatch_surface.build_dispatch_table --query-h5-dir $B/h5 --library-pkl $PKL --split-manifest $B/shadow_manifest.json --cache-yaml $B/calibration_retrieval.yaml --config-name pi05_libero --checkpoint-dir $CKPT --ref-mode tau1 --top-k 5 --h-exec 5 --extra-warm-tiers 0.5 --out-jsonl $B/table_tau1_k3.jsonl 2>&1 | grep -v -iE "warning|pynvml" || { echo K3_TABLE_FAILED $SUITE; exit 1; }
  step guard $SUITE
  $PY - "$B/table_tau1_k3.jsonl" "$B/table_tau1.jsonl" <<'PYG' || { echo K3_TABLE_FAILED guard $SUITE; exit 1; }
import json, sys
import numpy as np
new=[json.loads(l) for l in open(sys.argv[1])]; old=[json.loads(l) for l in open(sys.argv[2])]
n=len(new); eps=set(r["episode_id"] for r in new); tasks=set(r["task_id"] for r in new)
y5=np.array([r["y_tau5"] for r in new],float); y7=np.array([r["y_tau7"] for r in new],float); y10=np.array([r["y_tau10"] for r in new],float); s=np.array([r["s"] for r in new],float)
keff=set(r["k_eff"] for r in new); rm=set(r["ref_mode"] for r in new)
# parity with the K2 table on shared columns (same code path, same seeds -> identical s / y7 / y10)
o={(r["episode_id"],r["step_idx"]):r for r in old}
ds=max(abs(r["s"]-o[(r["episode_id"],r["step_idx"])]["s"]) for r in new); d7=max(abs(r["y_tau7"]-o[(r["episode_id"],r["step_idx"])]["y_tau7"]) for r in new); d10=max(abs(r["y_tau10"]-o[(r["episode_id"],r["step_idx"])]["y_tau10"]) for r in new)
print(f"rows={n} (old {len(old)}) episodes={len(eps)} tasks={len(tasks)} k_eff={sorted(keff)} ref_mode={sorted(rm)} parity max|ds|={ds:.2e} max|d7|={d7:.2e} max|d10|={d10:.2e}")
print(f"y5: min {y5.min():.3f} med {np.median(y5):.3f} p95 {np.quantile(y5,.95):.3f} max {y5.max():.3f} | y7 med {np.median(y7):.3f} | y10 med {np.median(y10):.3f}")
print(f"frac y5<=y7 {np.mean(y5<=y7):.3f}  y7<=y10 {np.mean(y7<=y10):.3f}")
from scipy.stats import spearmanr
print(f"spearman y5~s {spearmanr(y5,s).statistic:.3f} y7~s {spearmanr(y7,s).statistic:.3f} y10~s {spearmanr(y10,s).statistic:.3f}")
ok = (n==len(old) and len(eps)==150 and len(tasks)==10 and keff=={5} and rm=={"tau1"} and np.isfinite(y5).all() and ds<1e-9 and d7<1e-6 and d10<1e-6)
print("GUARD", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PYG
done
echo K3_TABLE_DONE_$1
