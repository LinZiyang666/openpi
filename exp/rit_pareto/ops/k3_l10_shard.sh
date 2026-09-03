#!/bin/bash
# k3_l10_shard.sh split|finalize — parallelise the remaining l10 K3 table build over 4 GPU processes.
set -uo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
export HOME=/home/weiland
R=/data/openpi_dispatch
B=/tmp/dsp_shared/rit_pareto/libero_10
PKL=/home/weiland/openpi/exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl
CKPT=/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
PY=/home/weiland/openpi/.venv/bin/python
N=4
cd $R; export PYTHONPATH=$R/src:$R
case "$1" in
split)
  tmux kill-session -t rit_k3_libero_10 2>/dev/null; P=$(pgrep -f "[b]uild_dispatch_table.*libero_10"); [ -n "$P" ] && kill $P; sleep 4
  $PY - <<'PYS'
import json, os, pathlib, h5py
B=pathlib.Path("/tmp/dsp_shared/rit_pareto/libero_10")
paths=sorted((B/"h5").rglob("*.h5"))
rows=[json.loads(l) for l in open(B/"table_tau1_k3.jsonl") if l.strip()]
per={}
for r in rows: per.setdefault(r["episode_id"],[]).append(r)
done_rows=[]; remaining=[]
for p in paths:
    with h5py.File(p,"r") as f: n=sum(1 for k in f.keys() if k.startswith("step_"))
    got=per.get(p.stem,[])
    if len(got)==n and n>0: done_rows.extend(got)
    else: remaining.append(p)
(B/"table_tau1_k3.done.jsonl").write_text("".join(json.dumps(r)+"\n" for r in done_rows))
shards=B/"h5_shards"
if shards.exists():
    import shutil; shutil.rmtree(shards)
for i in range(4): (shards/f"s{i}").mkdir(parents=True)
for j,p in enumerate(remaining):
    os.symlink(p.resolve(), shards/f"s{j%4}"/p.name)
print(f"kept {len(done_rows)} rows from {len(per)-len([p for p in remaining if p.stem in per])} complete episodes; remaining {len(remaining)} episodes -> 4 shards")
PYS
  rm -f $B/table_tau1_k3.part*.jsonl* /tmp/rit_k3_l10_part*.log
  for i in 0 1 2 3; do
    tmux new -s rit_k3_l10_p$i -d "cd $R && export HOME=/home/weiland PYTHONPATH=$R/src:$R && $PY -m exp.dispatch_surface.build_dispatch_table --query-h5-dir $B/h5_shards/s$i --library-pkl $PKL --split-manifest $B/shadow_manifest.json --cache-yaml $B/calibration_retrieval.yaml --config-name pi05_libero --checkpoint-dir $CKPT --ref-mode tau1 --top-k 5 --h-exec 5 --extra-warm-tiers 0.5 --out-jsonl $B/table_tau1_k3.part$i.jsonl 2>&1 | grep -v -iE 'warning|pynvml' | tee /tmp/rit_k3_l10_part$i.log; echo PART_EXIT=\${PIPESTATUS[0]} | tee -a /tmp/rit_k3_l10_part$i.log"
  done
  echo "shards launched"
  ;;
finalize)
  for i in 0 1 2 3; do grep -q "PART_EXIT=0" /tmp/rit_k3_l10_part$i.log 2>/dev/null || { echo "part $i not finished cleanly"; exit 1; }; done
  $PY - <<'PYF'
import json, pathlib
B=pathlib.Path("/tmp/dsp_shared/rit_pareto/libero_10")
order={p.stem:i for i,p in enumerate(sorted((B/"h5").rglob("*.h5")))}
rows=[]
for f in [B/"table_tau1_k3.done.jsonl"]+[B/f"table_tau1_k3.part{i}.jsonl" for i in range(4)]:
    rows += [json.loads(l) for l in open(f) if l.strip()]
rows.sort(key=lambda r:(order[r["episode_id"]], r["step_idx"]))
(B/"table_tau1_k3.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
import shutil; shutil.copy(B/"table_tau1_k3.part0.jsonl.weights.npz", B/"table_tau1_k3.jsonl.weights.npz")
print("merged rows", len(rows), "episodes", len(set(r["episode_id"] for r in rows)))
PYF
  echo "=== [$(date +%H:%M:%S)] guard libero_10" | tee -a /tmp/rit_k3_libero_10.log
  $PY - "$B/table_tau1_k3.jsonl" "$B/table_tau1.jsonl" <<'PYG' 2>&1 | tee -a /tmp/rit_k3_libero_10.log
import json, sys
import numpy as np
new=[json.loads(l) for l in open(sys.argv[1])]; old=[json.loads(l) for l in open(sys.argv[2])]
n=len(new); eps=set(r["episode_id"] for r in new); tasks=set(r["task_id"] for r in new)
y5=np.array([r["y_tau5"] for r in new],float); y7=np.array([r["y_tau7"] for r in new],float); y10=np.array([r["y_tau10"] for r in new],float); s=np.array([r["s"] for r in new],float)
keff=set(r["k_eff"] for r in new); rm=set(r["ref_mode"] for r in new)
o={(r["episode_id"],r["step_idx"]):r for r in old}
ds=max(abs(r["s"]-o[(r["episode_id"],r["step_idx"])]["s"]) for r in new); d7=max(abs(r["y_tau7"]-o[(r["episode_id"],r["step_idx"])]["y_tau7"]) for r in new); d10=max(abs(r["y_tau10"]-o[(r["episode_id"],r["step_idx"])]["y_tau10"]) for r in new)
keys_new=[(r["episode_id"],r["step_idx"]) for r in new]; same_order = keys_new == [(r["episode_id"],r["step_idx"]) for r in old]
print(f"rows={n} (old {len(old)}) episodes={len(eps)} tasks={len(tasks)} k_eff={sorted(keff)} ref_mode={sorted(rm)} parity max|ds|={ds:.2e} max|d7|={d7:.2e} max|d10|={d10:.2e} same_row_order={same_order}")
print(f"y5: min {y5.min():.3f} med {np.median(y5):.3f} p95 {np.quantile(y5,.95):.3f} max {y5.max():.3f} | y7 med {np.median(y7):.3f} | y10 med {np.median(y10):.3f}")
print(f"frac y5<=y7 {np.mean(y5<=y7):.3f}  y7<=y10 {np.mean(y7<=y10):.3f}")
from scipy.stats import spearmanr
print(f"spearman y5~s {spearmanr(y5,s).statistic:.3f} y7~s {spearmanr(y7,s).statistic:.3f} y10~s {spearmanr(y10,s).statistic:.3f}")
ok = (n==len(old) and len(eps)==150 and len(tasks)==10 and keff=={5} and rm=={"tau1"} and np.isfinite(y5).all() and ds<1e-9 and d7<1e-6 and d10<1e-6 and same_order and len(set(keys_new))==n)
print("GUARD", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
PYG
  st=${PIPESTATUS[0]}
  if [ "$st" -eq 0 ]; then echo K3_TABLE_DONE_libero_10 | tee -a /tmp/rit_k3_libero_10.log; else echo K3_TABLE_FAILED guard | tee -a /tmp/rit_k3_libero_10.log; exit 1; fi
  ;;
*) echo "usage: split|finalize"; exit 2 ;;
esac
