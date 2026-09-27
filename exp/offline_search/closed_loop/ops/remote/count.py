"""count.py <journal.jsonl> -> "<accepted_terminal> <success> <rows>": unique task_uids whose journal row is
accepted AND status in {done, failed} AND has no error (a normal task failure counts as complete), and how many of
them succeeded. Plain python3, no deps (runs on timan107 outside the venv)."""
import json
import sys

u = {}
rows = 0
try:
    for line in open(sys.argv[1]):
        try:
            r = json.loads(line)
        except Exception:
            continue
        rows += 1
        if r.get("accepted") and r.get("status") in ("done", "failed") and not r.get("error"):
            u[r["task_uid"]] = r
except FileNotFoundError:
    pass
print(len(u), sum(1 for r in u.values() if r.get("success")), rows)
