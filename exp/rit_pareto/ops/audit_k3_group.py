#!/usr/bin/env python
"""audit_k3_group.py <suite> <rule> <total> [--per-arm 500]  (runs on timan108)

Integrity audit for one K=3 group (handoff §6): terminal journal rows == unique uid == total,
no duplicate terminal uid, per-arm counts, failed episodes with suspiciously few per_step rows,
per_step (uid, attempt) set == journal (uid, attempt) set.  Exit 0 == OK, 1 == ANOMALY.
"""
import collections
import json
import sys

suite, rule, total = sys.argv[1], sys.argv[2], int(sys.argv[3])
per_arm = int(sys.argv[sys.argv.index("--per-arm") + 1]) if "--per-arm" in sys.argv else 500
out = f"/tmp/dsp_precheck/rit_pareto/{suite}_k3_{rule}"
min_steps = 42 if suite == "libero_spatial" else 100

term, dups, failed, arm_counts = {}, [], set(), collections.Counter()
with open(f"{out}/journal.jsonl") as fh:
    for line in fh:
        r = json.loads(line)
        if r.get("status") not in ("done", "failed"):
            continue
        uid = r["task_uid"]
        if uid in term:
            dups.append(uid)
        term[uid] = r
for uid, r in term.items():
    arm_counts[r["yaml_id"]] += 1
    if r["status"] == "failed":
        failed.add(uid)
j_pairs = {(u, r["attempt"]) for u, r in term.items()}

ps_rows = collections.Counter()
ps_pairs, hit_counts, timing_steps = set(), collections.Counter(), {}
with open(f"{out}/per_step.jsonl") as fh:
    for line in fh:
        r = json.loads(line)
        key = (r["task_uid"], r["attempt"])
        ps_pairs.add(key)
        if key in j_pairs:
            if r.get("_kind") == "client_timing" or "hit_type" not in r:
                hit_counts["<%s row>" % r.get("_kind", "no hit_type")] += 1
                if r.get("_kind") == "client_timing":
                    timing_steps[r["task_uid"]] = r.get("steps")
                continue
            ps_rows[r["task_uid"]] += 1
            ht = r["hit_type"] or "MISS"
            if ht == "WARM_START":
                ht = f"WARM_START@{r.get('start_t')}"
            hit_counts[ht] += 1

short_failed = sorted(u for u in failed if ps_rows[u] < min_steps)
# A failed episode that ended before the step cap never hit LIBERO's success-only
# ``done``: it was truncated by a client-side exception (e.g. websocket 1011 keepalive
# close, 2026-09-02 group 4) and must be excised + re-run, not counted as a failure.
cap = 500 if suite == "libero_10" else 200
truncated_failed = sorted(u for u in failed if timing_steps.get(u) is not None and timing_steps[u] < cap)
bad_arms = {a: n for a, n in arm_counts.items() if n != per_arm}
missing_ps = sorted(u for u, _ in (j_pairs - ps_pairs))
extra_ps = sorted(u for u, _ in (ps_pairs - j_pairs))
summary = {
    "suite": suite, "rule": rule, "total": total,
    "terminal_rows": len(term) + len(dups), "unique_uid": len(term), "dup_terminal_uid": len(dups),
    "arms": len(arm_counts), "arms_not_full": bad_arms,
    "failed": len(failed), "failed_short_per_step(<%d)" % min_steps: short_failed[:20],
    "truncated_failed(steps<%d)" % cap: len(truncated_failed), "truncated_failed_uids": truncated_failed,
    "journal_pairs_without_per_step": len(missing_ps), "per_step_pairs_not_terminal": len(extra_ps),
    "hit_counts": dict(hit_counts),
}
ok = (
    len(term) == total and not dups and not bad_arms and not short_failed
    and not missing_ps and not extra_ps and not truncated_failed
)
print(json.dumps(summary, indent=1))
print("AUDIT OK" if ok else "AUDIT ANOMALY")
sys.exit(0 if ok else 1)
