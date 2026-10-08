"""Accepted B-dev decisions only; pricing arithmetic and observable eligibility."""
from .common import *
install()
import numpy as np

rows=[]
for root in sorted(DEV-{'r06_c_cal'}):
    for run in sorted((RUNS/root/'runs').glob('*')):
        if not (run/'summary.json').exists():continue
        summary=json.loads((run/'summary.json').read_text())
        contract=json.loads((run/'pool_contract.json').read_text())
        assert contract['init_pool']=='B' and contract['dev']
        accepted={}
        for line in (run/'client/journal.jsonl').read_text().splitlines():
            a=json.loads(line)
            if a.get('accepted') and a.get('status') in ('done','failed'):
                assert a['init_pool']=='B'
                accepted[a['task_uid']]=a
        dec={}
        for path in sorted(run.glob('server_*/decisions_*.jsonl')):
            with path.open() as f:
                for line in f:
                    a=json.loads(line)
                    if a.get('ev')!='dec' or a.get('uid') not in accepted:continue
                    if a.get('attempt',1)!=accepted[a['uid']].get('attempt',1):continue
                    k=(a['uid'],a['step'])
                    if k in dec:assert dec[k]==a
                    dec[k]=a
        N=len(dec);V=sum(a.get('vision',True) for a in dec.values());M=sum(not a['hit'] for a in dec.values())
        ledger=summary['cost_ledger']
        assert (N,V,M)==(ledger['decisions'],ledger['vision_decisions'],ledger['misses']), (run.name,N,V,M,ledger['decisions'],ledger['vision_decisions'],ledger['misses'],len(accepted))
        assert len(accepted)==summary['complete']
        model=summary['model'];cv=.152 if model=='pi05' else .148
        # Eligible attempts use only prior full/non-stall history. Knob-due
        # information is pre-look for periodic; distance calls remain unknown.
        E=EC=0; per_ep=[]; refresh_counts=[]
        for uid,a in accepted.items():
            seq=[dec[(uid,s)] for s in range(sum(k[0]==uid for k in dec))]
            last_full=-999;prev_span=0.;counts=dict(N=len(seq),V=0,M=0,E=0,EC=0)
            for d in seq:
                if not d.get('vision',True):continue
                s=d['step'];counts['V']+=1;counts['M']+=int(not d['hit'])
                ex=d.get('extras',{})
                due=bool(ex.get('os_r11_knob_call',0)) if summary.get('kwargs',{}).get('method')=='periodic' else False
                eligible=(s>=2 and s-last_full==2 and prev_span==0 and not due and d.get('look_reason') not in (6,7,8))
                if eligible:
                    counts['E']+=1;counts['EC']+=int(not d['hit']);E+=1;EC+=int(not d['hit'])
                    # Worst-case no-refresh after a successful half. Completion
                    # on a policy call makes a genuine current full observation.
                    if not d['hit']:last_full=s
                else:last_full=s
                prev_span=float(ex.get('noprog_span',0))
            per_ep.append(dict(uid=uid,success=bool(a['success']),**counts))
            # An explicit new planner may recognize the completed policy tail
            # as a valid full-camera memory source. Never bypass other lifecycle
            # failures merely because their integer code is also 6.
            last_full=-999;prev_span=0.;er=ec=0
            for d in seq:
                if not d.get('vision',True):continue
                s=d['step'];ex=d.get('extras',{})
                due=bool(ex.get('os_r11_knob_call',0)) if summary.get('kwargs',{}).get('method')=='periodic' else False
                tail_ok=(s>=2 and not seq[s-2]['hit'] and seq[s-2].get('vision',True)
                         and seq[s-1].get('source',seq[s-1].get('src'))=='policy_tail')
                lifecycle=(d.get('look_reason') not in (6,7,8)) or (d.get('look_reason')==6 and tail_ok)
                eligible=(s>=4 and s-last_full==2 and prev_span==0 and not due and lifecycle)
                if eligible:
                    er+=1;ec+=int(not d['hit'])
                    if not d['hit']:last_full=s
                else:last_full=s
                prev_span=float(ex.get('noprog_span',0))
            refresh_counts.append(dict(E=er,EC=ec))
        out=dict(root=root,arm=run.name,model=model,suite={'libero_10':'l10','libero_spatial':'spatial'}.get(contract['suite'],contract['suite']),suite_full=contract['suite'],
                 size='current' if root=='r11_dev_current' else (200 if root=='r11_dev_size200' else 50),
                 method=summary.get('kwargs',{}).get('method','off'),episodes=len(accepted),
                 sr=sum(a['success'] for a in accepted.values())/len(accepted),N=N,V=V,M=M,v=V/N,m=M/N,
                 owner_ir=(cv*V+(1-cv)*M)/N,eligible=E,eligible_calls=EC,
                 eligible_share_looks=E/V,eligible_call_fraction=EC/max(E,1),per_episode=per_ep,
                 refreshed_eligible=sum(x['E'] for x in refresh_counts),
                 refreshed_eligible_calls=sum(x['EC'] for x in refresh_counts),
                 has_query_keys=False)
        if model=='pi05':
            out.update(all_half_ceiling_stock=(.152-.0646)*V/N-.0502*M/N,
                       half_noncall_ceiling_stock=(.152-.0646)*(V-M)/N,
                       alternate_eligible_saving_stock=(.152-.0646)*E/N-.0502*EC/N,
                       alternate_eligible_saving_request=(.1656-.0646)*E/N-.0136*V/N-.0502*EC/N,
                       no_policy_all_half_ceiling=.5*(.152-.0646),
                       no_policy_alternate_ceiling=.25*(.152-.0646),
                       all_policy_all_half_ceiling=.5*(.152-.0646-.0502))
        rows.append(out)
dump(HERE/'dev_cost.json',dict(timestamp=now(),arms=rows,prices=dict(stock_full=.152,request_full=.1656,wrist=.0646,completion=.0502),
    note='Counterfactual accounting keeps baseline decisions/calls fixed. Eligibility is only an exposure bound; it is not a changed-policy rollout.'))
dump(HERE/'dev_cost_reads.json',sorted(READS))
for r in rows:
    print(r['arm'],round(r['sr'],3),round(r['owner_ir'],5),round(r['eligible_share_looks'],3),
          round(r.get('alternate_eligible_saving_request',0),5))
