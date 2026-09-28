"""Audit only sufficient historical fields; never synthesize missing keys."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from exp.offline_search.rounds.r04.k7_guard.evidence import HERE, method

RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
CASES = [('r03_mx','r3mx_p_l10_g500'), ('r04_blind','r4b3_p_l10_500_b0g'),
         ('r04_blind','r4b3_p_l10_500_ph2g')]
m=method('pi05_l10',500)
result=[]
for run, arm in CASES:
    root=RUNS/run/'runs'/arm
    journal=[json.loads(x) for x in (root/'client/journal.jsonl').read_text().splitlines()]
    accepted={r['task_uid']:r for r in journal if r.get('accepted') and r.get('status') in ('done','failed')}
    eps=defaultdict(dict); startup=[]; filemeta=[]; field_counts=Counter()
    for path in sorted(root.glob('server_*/decisions_*.jsonl')):
        filemeta.append(dict(path=str(path),bytes_at_read=path.stat().st_size))
        with path.open() as f:
            for line in f:
                r=json.loads(line)
                if r['ev']=='startup':
                    startup.append({k:r.get(k) for k in ('method_spec','kwargs','log_inputs')})
                if r['ev']!='dec' or r.get('uid') not in accepted:continue
                if r.get('attempt')!=accepted[r['uid']]['attempt']:continue
                selected={k:r.get(k) for k in ('step','task_id','vision','robot_state','hit','extras')}
                for k in ('robot_state','key_v0','key_v1'):
                    field_counts[k]+=int(k in r)
                field_counts['vself']+=int('vself' in (r.get('extras') or {}))
                eps[r['uid']][r['step']]=selected
    counts=Counter(); original=Counter(); projected=Counter(); rederived_stuck=0
    verified=0; missing=0; raw_low=0; dense_mismatch=0; unsupported=0
    for uid, bystep in eps.items():
        ordered=[bystep[k] for k in sorted(bystep)]
        assert [x['step'] for x in ordered]==list(range(len(ordered))),uid
        sc=dc=0; prev=None
        for r in ordered:
            ex=r['extras'] or {};counts['decisions']+=1
            counts['vision']+=r['vision'] is not False;counts['miss']+=r['hit'] is False
            flags=int(ex.get('os_flags',0));reason=int(ex.get('os_reason',0))
            original[reason]+=1;counts['stuck_flag']+=bool(flags&1)
            if run=='r03_mx':
                motion=ex.get('motion',float('nan'))
            elif prev is not None:
                now=np.asarray(r['robot_state'],np.float32)[:8]
                before=np.asarray(prev['robot_state'],np.float32)[:8]
                motion=float(np.linalg.norm(now-before))
                scale=m.base.state_scale_by_task[int(r['task_id'])]
                dense=float(np.sqrt(np.mean(((now-before)/scale)**2)))
                dc=dc+1 if dense<m.base.motion10[int(r['task_id'])] else 0
                if r['vision'] is not False:
                    dense_mismatch+=int(dc!=ex.get('stuck_n'))
            else:motion=float('nan')
            raw_low+=int(motion<m.m_thr)
            if r['step'] and 'vself' not in ex:
                unsupported+=1
            if arm.endswith('ph2g'):
                # At gaps the necessary anchor-pair cosine is absent. Even a
                # later adjacent cosine cannot reconstruct the earlier run.
                prev=r;continue
            if not r['step']:
                sc=0
            elif 'vself' in ex:
                sc=sc+1 if motion<m.m_thr and ex['vself']>=m.c_thr else 0
            else:
                missing+=1;prev=r;continue
            verified+=1; rederived_stuck+=int(sc>=m.stuck_thr)
            if run=='r03_mx':
                counts['stock_count_mismatches']+=int(sc!=ex['stuck_n'])
            # Only stuck/overtime depend on this count; retain all other logged
            # flags. This is a same-path diagnostic, not new HIT/MISS execution.
            newflags=flags&~5
            if sc>=m.stuck_thr:newflags|=1
            if ex.get('overtime',0)>1 and ex.get('lag',0)>m.lag_thr and sc>=1:newflags|=4
            newreason=(newflags&-newflags).bit_length() if newflags else 0
            projected[newreason]+=1
            prev=r
    result.append(dict(run=run,arm=arm,episodes=len(eps),success=sum(bool(v['success']) for v in accepted.values()),
        counts=dict(counts),original_reason_counts=dict(original),projected_reason_counts=dict(projected),
        stock_stuck_rederived=rederived_stuck if verified else None,verified_decisions=verified,
        missing_required_adjacent_cosines=missing,unavailable_vself=unsupported,raw_low_motion_decisions=raw_low,
        dense_count_mismatches=dense_mismatch,fields=dict(field_counts),startup=startup,files=filemeta,
        note='No counterfactual SR/confidence claim; gap confirmation unavailable without anchor keys/cosine.'))
(HERE/'results/historical.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
