"""summarize.py -> runs/summary.json + printed table: per group integrity + outcome + volume."""
import json, pathlib, collections, os
B = pathlib.Path("/home/weiland/trace_runs/dual_20260923/runs")
ARMS = ["tr_pi05_sp_inf", "tr_pi05_sp_cache", "tr_pi05_l10_inf", "tr_pi05_l10_cache",
        "tr_groot_sp_inf", "tr_groot_sp_cache", "tr_groot_l10_inf", "tr_groot_l10_cache"]
out = {}
for arm in ARMS:
    d = B / arm
    jf = d / "client" / "journal.jsonl"; cf = d / "check_trace.jsonl"
    if not jf.exists() or not cf.exists():
        out[arm] = {"status": "missing", "journal": jf.exists(), "check": cf.exists()}; continue
    rows = [json.loads(l) for l in jf.open() if l.strip()]
    term = {}
    for r in rows:
        if r.get("accepted") and r.get("status") in ("done", "failed") and not r.get("error"):
            term[r["task_uid"]] = r
    files = [json.loads(l) for l in cf.open() if l.startswith("{")]
    summ = [l for l in cf.open() if l.startswith("SUMMARY")]
    tuid = collections.Counter(f["uid"] for f in files if f["terminal"] and f["closed"] and f["werr"] == 0)
    succ_j = sum(1 for r in term.values() if r.get("success"))
    succ_t = sum(1 for f in files if f["uid"] in term and f["success"])
    per_task = collections.Counter(); per_task_n = collections.Counter()
    for u, r in term.items():
        t = int(u.split(":")[2]); per_task_n[t] += 1; per_task[t] += bool(r.get("success"))
    dec = [f["n"] for f in files]
    size = sum(p.stat().st_size for p in (d / "trace").rglob("*") if p.is_file())
    out[arm] = {
        "episodes_terminal": len(term), "journal_rows": len(rows), "success": succ_j,
        "success_rate": round(succ_j / max(len(term), 1), 4),
        "trace_files": len(files), "trace_uids_match_journal": set(tuid) == set(term),
        "trace_dup_uids": sum(1 for v in tuid.values() if v > 1), "trace_success_agrees": succ_t == succ_j,
        "decisions_total": sum(dec), "decisions_mean": round(sum(dec) / max(len(dec), 1), 2),
        "trace_bytes": size, "trace_GiB": round(size / 2**30, 1),
        "MiB_per_decision": round(size / 2**20 / max(sum(dec), 1), 3),
        "per_task_success": {t: f"{per_task[t]}/{per_task_n[t]}" for t in sorted(per_task_n)},
        "check_summary": summ[-1].strip() if summ else None,
    }
(B / "summary.json").write_text(json.dumps(out, indent=2))
print(f"{'arm':22s} {'eps':>4s} {'succ':>5s} {'SR':>6s} {'dec':>7s} {'dec/ep':>7s} {'GiB':>6s} {'MiB/d':>6s} uid_match")
for a, r in out.items():
    if r.get("status") == "missing": print(a, "missing", r); continue
    print(f"{a:22s} {r['episodes_terminal']:4d} {r['success']:5d} {r['success_rate']:6.3f} {r['decisions_total']:7d} {r['decisions_mean']:7.1f} {r['trace_GiB']:6.1f} {r['MiB_per_decision']:6.2f} {r['trace_uids_match_journal']} dup={r['trace_dup_uids']}")
