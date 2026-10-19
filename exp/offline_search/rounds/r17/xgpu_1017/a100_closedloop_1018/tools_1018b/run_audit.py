"""A100-side read-only audit of completed closed-loop runs (stdlib only).

Per run: RUN receipt, owned-PID cleanup, private MPS closure, accepted-UID
population, driver receipt, and server_timing.clock_unit / segment-head
fields of every served decision (driver per_step rows and worker socket traces).
"""
import collections, glob, json, os, sys
from pathlib import Path

import os
ROOT = Path(os.environ.get('XGPU_AUDIT_ROOT', '/home/exouser/xgpu_1017/cl18'))


def rows(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def audit(run):
    d = ROOT / 'runs' / run
    r = json.loads((d / 'RUN.json').read_text())
    model, arm = r['model'], r['arm']
    uids = set(json.loads((ROOT / model / arm / 'UIDS.json').read_text()))
    out = dict(run=run, status=r['status'], episodes_planned=r.get('episodes'),
               gpu=r.get('gpu'), started=r.get('started'), ended=r.get('ended'),
               mps_daemon_closed=(r.get('mps_cleanup') or {}).get('daemon_closed'),
               owned_children_reaped=all(c.get('returncode') is not None for c in r.get('cleanup', [])),
               owned_children=len(r.get('cleanup', [])))
    acc = {}
    errors = 0
    for x in rows(d / 'driver/journal.jsonl'):
        if x.get('accepted') is True:
            if x.get('error'):
                errors += 1
            acc[x['task_uid']] = int(bool(x.get('success')))
    out.update(accepted=len(acc), accepted_errors=errors, accepted_equals_plan=set(acc) == uids,
               successes=sum(acc.values()))
    rec = d / 'driver/DRIVER_RECEIPT.json'
    if rec.exists():
        rr = json.loads(rec.read_text())
        out['driver_receipt_status'] = rr.get('status')
    att = json.loads((d / 'SERVER_ATTESTATIONS.json').read_text())
    out['residents'] = len(att)
    ps = collections.Counter(); cu = collections.Counter(); cursor = collections.Counter()
    head = collections.Counter(); ch = collections.Counter(); keys = collections.Counter()
    src = collections.Counter(); n = 0
    for x in rows(d / 'driver/per_step.jsonl'):
        n += 1
        st = x.get('server_timing') or {}
        keys[tuple(sorted(st))] += 1
        cu[str(st.get('clock_unit'))] += 1
        head[str(st.get('segment_head'))] += 1
        cursor[str(x.get('cursor'))] += 1
        ch[str(x.get('commit_horizon'))] += 1
        src[str(x.get('source'))] += 1
    out.update(per_step_rows=n, server_timing_clock_unit=dict(cu), server_timing_segment_head=dict(head),
               server_timing_key_sets={'|'.join(k): v for k, v in keys.items()},
               cursor=dict(cursor), commit_horizon=dict(ch), source=dict(src))
    tc = collections.Counter(); tcur = collections.Counter(); tn = 0; texec = collections.Counter()
    for p in glob.glob(str(d / 'workers/*/trace/*.jsonl')):
        if p.endswith('.executed.jsonl'):
            for x in rows(p):
                texec[str(x.get('executed_count'))] += 1
            continue
        for x in rows(p):
            if 'cache_decision' not in x:
                continue
            tn += 1
            cd = x.get('cache_decision') or {}
            tc[str(cd.get('clock_unit'))] += 1
    out.update(trace_queries=tn, trace_clock_unit=dict(tc), trace_executed_count=dict(texec))
    return out


if __name__ == '__main__':
    runs = sys.argv[1:] or sorted(p.name for p in (ROOT / 'runs').iterdir() if (p / 'RUN.json').exists())
    print(json.dumps([audit(r) for r in runs], indent=1, sort_keys=True))
