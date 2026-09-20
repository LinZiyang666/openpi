#!/bin/bash
# Pilot runs (plan §5.2): 100 fixed updates per cell from a *temporary* matrix ($X0_DATA/cells_pilot, default budgets),
# written to $X0_DATA/pilot/<cell_id> (identity carries budget_override=100, never a formal start). Records per cell:
# updates/s from train_log.jsonl (elapsed_s of update 100 - update 20, warm-up excluded), save time, peak GPU memory
# (nvidia-smi sampled every 2 s), peak RSS, load time; summary at $X0_DATA/pilot/pilot_summary.json.
# usage: pilot_x0.sh <cell_id regex>   e.g. 'pusht_lowdim_(U|M)_(epsilon|sample)_s42_' or 'pusht_image_'
set -u
PAT=${1:?cell id regex}
source "$(dirname "$0")/dp_x0_env.sh"
cd $X0_CODE
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
P=$X0_DATA/pilot; mkdir -p $P
say() { echo "== $(date +%H:%M:%S) PILOT $*"; }
for y in $(ls $X0_DATA/cells_pilot/*.yaml | grep -E "$PAT"); do
  cid=$(basename $y .yaml); out=$P/$cid
  if [ -f $out/checkpoints/final.done ]; then say "$cid present"; continue; fi
  mkdir -p $out
  ( while sleep 2; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; done > $out/gpu_mem.txt ) &
  smp=$!
  t0=$(date +%s.%N)
  /usr/bin/time -f "%M" -o $out/rss_kb.txt python -m exp.dp_nfe.train_x0 --dp-root $DP_ROOT --cell $y --out $out --budget 100 > $out.log 2>&1
  rc=$?; t1=$(date +%s.%N); kill $smp 2>/dev/null; wait $smp 2>/dev/null
  say "$cid rc=$rc wall=$(python -c "print(round($t1-$t0,1))")s $(grep -ao 'FIXEDSTEP DONE.*' $out.log | cut -c1-80)"
done
python - "$P" "$PAT" <<'PY'
import json, pathlib, re, sys
P = pathlib.Path(sys.argv[1]); pat = re.compile(sys.argv[2])
summary = json.loads((P / "pilot_summary.json").read_text()) if (P / "pilot_summary.json").exists() else {}
for d in sorted(P.iterdir()):
    if not d.is_dir() or not pat.search(d.name) or not (d / "train_log.jsonl").exists():
        continue
    log = [json.loads(l) for l in (d / "train_log.jsonl").read_text().splitlines()]
    by = {r["step"]: r for r in log}
    rate = (100 - 20) / (by[100]["elapsed_s"] - by[20]["elapsed_s"]) if 100 in by and 20 in by else None
    gpu = [int(x) for x in (d / "gpu_mem.txt").read_text().split()] if (d / "gpu_mem.txt").exists() else []
    rss = (d / "rss_kb.txt").read_text().split()[-1] if (d / "rss_kb.txt").exists() else None
    ident = json.loads((d / "identity.json").read_text()) if (d / "identity.json").exists() else {}
    summary[d.name] = {"updates_per_s": rate, "peak_gpu_mib": max(gpu) if gpu else None, "peak_rss_mib": int(rss) // 1024 if rss else None,
                       "n_windows": ident.get("n_windows"), "elapsed_100": by.get(100, {}).get("elapsed_s"),
                       "final_ckpt_mib": round((d / "checkpoints" / "final.ckpt").stat().st_size / 2**20, 1) if (d / "checkpoints" / "final.ckpt").exists() else None}
    if rate:
        summary[d.name]["hours_per_budget"] = {b: round(b / rate / 3600, 2) for b in (10000, 20000, 40000, 50000, 100000)}
(P / "pilot_summary.json").write_text(json.dumps(summary, indent=1))
for k, v in summary.items():
    if pat.search(k):
        print(f"PILOT {k}: {v['updates_per_s'] and round(v['updates_per_s'], 2)} upd/s gpu={v['peak_gpu_mib']}MiB rss={v['peak_rss_mib']}MiB ckpt={v['final_ckpt_mib']}MiB h/B={v.get('hours_per_budget')}")
PY
echo "PILOT_DONE $PAT $(date +%H:%M:%S)"
