#!/usr/bin/env python3
"""Drop terminal journal records of episodes whose client loop ended on an exception, so the driver re-runs them.

examples/libero/main.py:_run_episode catches generic exceptions (a websocket keepalive timeout on an overloaded client
host, an offscreen-render error), breaks out and returns success=False; the conductor then journals that as an ordinary
failed outcome. Such an episode never ran the controller to its end, so it is not an outcome. The per-step file carries
the client's own verdict: its `client_timing` row has termination_reason == "exception" and the same (task_uid, run_id,
attempt) as the journal record. Matching records are appended to purged_exceptions.jsonl and removed from the journal
(atomic rewrite); on resume the driver re-runs those uids. The stale per-step rows stay; readers select by the journal.

Usage: purge_exc.py <arm_dir>  -> prints the number of purged records.
"""
import json
import os
import sys

d = sys.argv[1]
jp, pp = os.path.join(d, 'journal.jsonl'), os.path.join(d, 'per_step.jsonl')
if not (os.path.exists(jp) and os.path.exists(pp)):
    print(0)
    sys.exit(0)
exc = set()
with open(pp) as f:
    for line in f:
        if '"client_timing"' not in line or '"exception"' not in line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get('_kind') == 'client_timing' and r.get('termination_reason') == 'exception':
            exc.add((r.get('task_uid'), r.get('run_id'), r.get('attempt')))
keep, drop = [], []
with open(jp) as f:
    for line in f:
        try:
            r = json.loads(line)
        except ValueError:
            keep.append(line)
            continue
        key = (r.get('task_uid'), r.get('run_id'), r.get('attempt'))
        if r.get('status') in ('done', 'failed') and key in exc:
            drop.append(line)
        else:
            keep.append(line)
if drop:
    with open(os.path.join(d, 'purged_exceptions.jsonl'), 'a') as f:
        f.writelines(drop)
    tmp = jp + '.purge_tmp'
    with open(tmp, 'w') as f:
        f.writelines(keep)
    os.replace(tmp, jp)
print(len(drop))
