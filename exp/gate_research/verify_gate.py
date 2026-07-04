"""Verify a gate_research suite: journal SR + per-config verdict tally + dedup.

Usage: python3 verify_gate.py <journal.jsonl> <gate_rows.jsonl> <total_ep>
"""
import sys, json, collections

J, G, TOTAL = sys.argv[1], sys.argv[2], int(sys.argv[3])

dj = [json.loads(l) for l in open(J)]
term = [r for r in dj if r.get("status") in ("done", "failed")]
succ = sum(1 for r in term if r.get("success"))
print("journal: rows=%d terminal=%d/%d success=%d (SR %.1f%%)"
      % (len(dj), len(term), TOTAL, succ, 100.0 * succ / max(len(term), 1)))

rows = [json.loads(l) for l in open(G)]
kinds = collections.Counter(r.get("_kind", "per_step") for r in rows)
print("gate_rows total=%d kinds=%s" % (len(rows), dict(kinds)))

perc = collections.defaultdict(collections.Counter)
sr = collections.defaultdict(lambda: [0, 0])
for r in rows:
    if r.get("_kind") == "episode_summary":
        y = r["yaml_id"]
        sr[y][0] += 1 if r.get("success") else 0
        sr[y][1] += 1
    else:
        perc[r["yaml_id"]][r["hit_type"]] += 1

print("per-config (verdict tally + episode SR):")
for y in sorted(perc):
    c = perc[y]
    tot = sum(c.values())
    fh, ws, ms = c.get("FULL_HIT", 0), c.get("WARM_START", 0), c.get("MISS", 0)
    ok, n = sr[y]
    short = y.split("__d1__")[-1]
    # inference ratio proxy: FULL_HIT skips stage2, WS/MISS incur inference
    infr = (ws + ms) / max(tot, 1)
    print("  %-22s SR=%3.0f%% (%d/%d)  verdicts=%d FH=%d WS=%d MISS=%d  inf_ratio~%.3f"
          % (short, 100.0 * ok / max(n, 1), ok, n, tot, fh, ws, ms, infr))

# crash-safety dedup check: (task_uid, step_idx) duplicates only appear if a
# process death re-ran an in-flight episode (append checkpoint keeps both).
seen = collections.Counter(
    (r.get("task_uid"), r.get("step_idx")) for r in rows if r.get("_kind") != "episode_summary"
)
dups = sum(1 for k, v in seen.items() if v > 1)
print("dup (task_uid,step_idx) per-step keys: %d (0 = no crash-resume duplicates)" % dups)

# robot_state sanity
ps = [r for r in rows if r.get("_kind") != "episode_summary"]
with_rs = sum(1 for r in ps if r.get("collect", {}).get("robot_state"))
print("per-step rows with robot_state: %d/%d" % (with_rs, len(ps)))
