#!/usr/bin/env python3
"""R6 data-quality repair audit (ANALYSIS.md section 1).

For every arm under os_closed_loop/<run>/runs/<arm>/client that has journal.jsonl and per_step.jsonl:
- exception keys = client_timing rows with termination_reason == "exception", keyed (task_uid, run_id, attempt);
- residual = accepted terminal journal records (status done/failed) whose key is an exception key (must be 0);
- coverage = accepted terminal records that have any client_timing row with a termination_reason;
- purged = records in client/purged_exceptions.jsonl (written by closed_loop/ops/remote/purge_exc.py).
For purged arms, the pre-repair outcome of a purged (task, init) is the purged record (success=False); the post-repair
outcome is the current accepted record. Also reports the 7 ledger/client decision-count tolerance flags.
Writes analysis_r6/repair_*.json|csv. Read-only on run roots.
"""
import csv
import glob
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402


def scan(client_dir):
    jp, pp = os.path.join(client_dir, 'journal.jsonl'), os.path.join(client_dir, 'per_step.jsonl')
    arm_dir = os.path.dirname(client_dir)
    run = os.path.basename(os.path.dirname(os.path.dirname(arm_dir)))
    arm = os.path.basename(arm_dir)
    res = dict(run=run, arm=arm, has_per_step=os.path.exists(pp))
    exc, reasons = set(), {}
    if res['has_per_step']:
        out = subprocess.run(['grep', '-F', '"client_timing"', pp], capture_output=True, text=True).stdout
        for line in out.splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('_kind') != 'client_timing':
                continue
            k = (r.get('task_uid'), r.get('run_id'), r.get('attempt'))
            reasons[k] = r.get('termination_reason')
            if r.get('termination_reason') == 'exception':
                exc.add(k)
    acc, resid, covered, term = 0, 0, 0, Counter()
    with open(jp) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('status') not in ('done', 'failed') or ':eval:' not in (r.get('task_uid') or ''):
                continue
            acc += 1
            k = (r.get('task_uid'), r.get('run_id'), r.get('attempt'))
            if k in exc:
                resid += 1
            if reasons.get(k) is not None:
                covered += 1
                term[reasons[k]] += 1
    res.update(accepted=acc, residual_exceptions=resid, covered=covered, termination=dict(term),
               exception_rows=len(exc))
    pf = os.path.join(client_dir, 'purged_exceptions.jsonl')
    if os.path.exists(pf):
        recs = [json.loads(l) for l in open(pf) if l.strip()]
        keys = [(r.get('task_uid'), r.get('run_id'), r.get('attempt')) for r in recs]
        res.update(purged_records=len(recs), purged_unique_uids=len({r['task_uid'] for r in recs}),
                   purged_all_failed=all(r.get('success') is False for r in recs),
                   purged_all_match_exception_rows=all(k in exc for k in keys),
                   purged_uids=sorted({r['task_uid'] for r in recs}))
    return res


def main():
    dirs = sorted(d for d in glob.glob(f'{cm.ROOT}/*/runs/*/client') if os.path.exists(os.path.join(d, 'journal.jsonl')))
    with Pool(4) as p:
        rows = p.map(scan, dirs, chunksize=4)
    by_run = defaultdict(lambda: Counter())
    for r in rows:
        c = by_run[r['run']]
        c['arms'] += 1
        c['accepted'] += r['accepted']
        c['covered'] += r['covered']
        c['residual'] += r['residual_exceptions']
        c['purged_arms'] += 'purged_records' in r
        c['purged_records'] += r.get('purged_records', 0)
        c['purged_uids'] += r.get('purged_unique_uids', 0)
        c['exception_rows'] += r['exception_rows']
    # before / after for purged arms
    ba = []
    for r in rows:
        if 'purged_records' not in r:
            continue
        spec = f"{r['run']}:{r['arm']}"
        cur = cm.journal(spec)
        keys = {tuple(int(x) for x in u.split(':eval:')[1].split(':')[:2]) for u in r['purged_uids']}
        before = dict(cur)
        for k in keys:
            before[k] = False
        rerun_success = sum(cur[k] for k in keys if k in cur)
        s = cm.summary(spec)
        ba.append(dict(run=r['run'], arm=r['arm'], n=len(cur), complete=set(cur) == cm.EXPECTED,
                       purged_records=r['purged_records'], purged_pairs=len(keys), rerun_success=rerun_success,
                       sr_before=cm.sr(before), sr_after=cm.sr(cur), delta_pp=100 * (cm.sr(cur) - cm.sr(before)),
                       summary_success_matches=s.get('success') == sum(cur.values()),
                       purged_all_failed=r['purged_all_failed'],
                       purged_all_match_exception_rows=r['purged_all_match_exception_rows']))
    ba.sort(key=lambda x: (x['run'], x['arm']))
    # ledger tolerance flags
    flags = json.load(open(os.path.join(cm.R6, 'frontier_final', 'ledger_tolerance_flags.json')))
    tol = []
    for arm, f in flags.items():
        cand = glob.glob(f'{cm.ROOT}/*/runs/{arm}/summary.json')
        spec = f"{cand[0].split('/')[-4]}:{arm}"
        s = cm.summary(spec)
        N, V, M, L = cm.ledger(spec)
        cd = int(s['client_decisions'])
        c1 = cm.C1[s['model']]
        ir = (c1 * V + (1 - c1) * M) / N
        d = cd - N
        lo = (c1 * V + (1 - c1) * M) / cd if d > 0 else (c1 * max(V + d, 0) + (1 - c1) * max(M + d, 0)) / cd
        hi = (c1 * (V + max(d, 0)) + (1 - c1) * (M + max(d, 0))) / cd if d > 0 else (c1 * V + (1 - c1) * M) / cd
        tol.append(dict(arm=spec, ledger_N=N, client_N=cd, rel_diff=(N - cd) / cd, owner_IR=ir,
                        IR_bound_lo=min(lo, hi), IR_bound_hi=max(lo, hi), purged=any(b['arm'] == arm for b in ba)))
    cm.dump('repair_scan_by_run.json', {k: dict(v) for k, v in sorted(by_run.items())})
    cm.dump('repair_scan_arms.json', rows)
    cm.dump('repair_before_after.json', ba)
    cm.dump('repair_ledger_tolerance.json', tol)
    with open(os.path.join(cm.OUT, 'repair_before_after.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(ba[0]))
        w.writeheader()
        w.writerows(ba)
    tot = Counter()
    for v in by_run.values():
        tot.update(v)
    print(json.dumps(dict(total=dict(tot), purged_arms=len(ba), purged_records=sum(b['purged_records'] for b in ba),
                          purged_pairs=sum(b['purged_pairs'] for b in ba),
                          runs_with_purges=sorted({b['run'] for b in ba}),
                          residual_any=[(r['run'], r['arm'], r['residual_exceptions']) for r in rows if r['residual_exceptions']],
                          uncovered=[(r['run'], r['arm'], r['accepted'] - r['covered']) for r in rows
                                     if r['accepted'] != r['covered']][:40]), indent=1))


if __name__ == '__main__':
    main()
