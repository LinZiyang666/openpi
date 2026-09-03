#!/bin/bash
# postshadow.sh <suite>  — verify cohort -> tables (tau1 primary, uncoupled control) -> export -> emit
set -uo pipefail
SUITE=$1
export PATH=/usr/local/bin:/usr/bin:/bin
export HOME=/home/weiland
R=/data/openpi_dispatch
B=/tmp/dsp_shared/rit_pareto/$SUITE
PKL=/home/weiland/openpi/exp/common/data/cache_artifacts/$SUITE/cp1_spatial_pool_16.pkl
CKPT=/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
PY=/home/weiland/openpi/.venv/bin/python
cd $R
export PYTHONPATH=$R/src:$R
step() { echo "=== [$(date +%H:%M:%S)] $*"; }
step verify
$PY -m exp.dispatch_surface.collect_query_cohort verify --plan $B/cohort_plan.json --h5-dir $B/h5 --out $B/cohort_manifest.json || { echo POSTSHADOW_FAILED verify; exit 1; }
for MODE in tau1; do
  step table $MODE
  $PY -m exp.dispatch_surface.build_dispatch_table --query-h5-dir $B/h5 --library-pkl $PKL --split-manifest $B/shadow_manifest.json --cache-yaml $B/calibration_retrieval.yaml --config-name pi05_libero --checkpoint-dir $CKPT --ref-mode $MODE --top-k 5 --h-exec 5 --out-jsonl $B/table_$MODE.jsonl 2>&1 | grep -v -iE "warning|pynvml" || { echo POSTSHADOW_FAILED table $MODE; exit 1; }
  step guard $MODE
  $PY - "$B/table_$MODE.jsonl" "$MODE" <<'PYG' || { echo POSTSHADOW_FAILED guard; exit 1; }
import json, sys, math
import numpy as np
rows=[json.loads(l) for l in open(sys.argv[1])]
mode=sys.argv[2]
n=len(rows); eps=set(r["episode_id"] for r in rows); tasks=set(r["task_id"] for r in rows)
s_none=sum(r["s"] is None for r in rows); v_none=sum(r["v"] is None for r in rows)
keff=set(r["k_eff"] for r in rows); rm=set(r["ref_mode"] for r in rows)
y7=np.array([r["y_tau7"] for r in rows],float); y10=np.array([r["y_tau10"] for r in rows],float); s=np.array([r["s"] for r in rows],float)
splits={sp: len(set(r["episode_id"] for r in rows if r["split"]==sp)) for sp in ("fit","cal")}
print(f"rows={n} episodes={len(eps)} tasks={len(tasks)} splits={splits} s_none={s_none} v_none={v_none} k_eff={sorted(keff)} ref_mode={sorted(rm)}")
print(f"s: min {s.min():.4f} p10 {np.quantile(s,.1):.4f} med {np.median(s):.4f} p90 {np.quantile(s,.9):.4f} max {s.max():.4f}")
print(f"y7 : min {y7.min():.3f} med {np.median(y7):.3f} p95 {np.quantile(y7,.95):.3f} max {y7.max():.3f} | y10: min {y10.min():.3f} med {np.median(y10):.3f} p95 {np.quantile(y10,.95):.3f} max {y10.max():.3f}")
from scipy.stats import spearmanr
print(f"spearman y10~s {spearmanr(y10,s).statistic:.3f}  y7~s {spearmanr(y7,s).statistic:.3f}")
ok = (len(eps)==150 and len(tasks)==10 and s_none==0 and v_none==0 and keff=={5} and rm=={mode} and np.isfinite(y7).all() and np.isfinite(y10).all() and splits=={"fit":50,"cal":100})
print("GUARD", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PYG
done
step export tau1
rm -rf $B/export_tau1
$PY -m exp.rit_pareto.export_rit fit --suite $SUITE --table $B/table_tau1.jsonl --weights-npz $B/table_tau1.jsonl.weights.npz --cache-yaml $B/calibration_retrieval.yaml --library-pkl $PKL --checkpoint-dir $CKPT --ref-mode tau1 --out-dir $B/export_tau1 2>&1 | grep -v -iE "warning|pynvml" || { echo POSTSHADOW_FAILED export; exit 1; }
step emit
rm -rf $B/arms
$PY -m exp.rit_pareto.emit_arms --export-record $B/export_tau1/export_record.json --suite $SUITE --library-pkl $PKL --out-dir $B/arms 2>&1 | grep -v -iE "warning|pynvml" || { echo POSTSHADOW_FAILED emit; exit 1; }
step package
cd /tmp/dsp_shared/rit_pareto && tar -czf /tmp/dsp_shared/rit_pareto/${SUITE}_arms.tgz $SUITE/export_tau1 $SUITE/arms && sha256sum /tmp/dsp_shared/rit_pareto/${SUITE}_arms.tgz
echo POSTSHADOW_DONE
