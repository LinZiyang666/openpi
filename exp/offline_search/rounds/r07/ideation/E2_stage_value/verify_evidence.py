"""Independent accounting checks, not a policy test or closed-loop evaluation."""
import ast
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_stage import OUT,TMP,RUN,FRONT

lot=pd.read_pickle(TMP/'lottery.pkl')
pf=pd.read_pickle(TMP/'profile.pkl')
lk=pd.read_pickle(TMP/'look.pkl')
checks=[]
for (run,arm),g in lot.groupby(['run','arm']):
    s=json.loads((RUN/run/'runs'/arm/'summary.json').read_text())['cost_ledger']
    delta_v=len(g)-s['vision_decisions'];delta_m=int(g.Z.sum())-s['misses']
    assert np.array_equal(g.Z.to_numpy(),(g.coin<g.p).astype(int).to_numpy())
    assert g.uid.nunique()==500
    checks.append(dict(run=run,arm=arm,anchors=len(g),calls=int(g.Z.sum()),
                       ledger_match=delta_v==delta_m==0,delta_vision=delta_v,delta_misses=delta_m))
assert len(checks)==23
discrepancies=[r for r in checks if not r['ledger_match']]
assert len(discrepancies)==1,discrepancies
assert discrepancies[0]['arm']=='r6q2_groot_l10_50_risk_rho0p45',discrepancies
# Independent accepted client timing identifies the two shorter repair episodes.
arm=discrepancies[0]['arm'];root=RUN/'r06_frontier/runs'/arm
accepted={}
for line in (root/'client/journal.jsonl').open():
    r=json.loads(line)
    if r.get('accepted') and r.get('status') in ['done','failed'] and not r.get('error'):
        accepted[r['task_uid']]=r
timings=[]
for line in (root/'client/per_step.jsonl').open():
    r=json.loads(line);uid=r.get('task_uid','')
    if r.get('_kind')=='client_timing' and uid in accepted and r.get('run_id')==accepted[uid].get('run_id'):
        if uid.endswith((':6:2',':6:12')):
            g=lot[lot.arm.eq(arm)&lot.uid.eq(uid)]
            assert len(g)==(r['infers']+1)//2
            assert int(g.step.max())==r['infers']-1
            timings.append(dict(uid=uid,accepted_run_id=r['run_id'],client_infers=r['infers'],retained_anchors=len(g)))
assert len(timings)==2
assert not lot.duplicated(['run','arm','uid','step']).any()
assert not pf.duplicated(['campaign','arm','uid','step']).any()
assert not lk.duplicated(['campaign','arm','uid','step']).any()
assert len(pf[pf.campaign.eq('r06_p3_pilot')])==10691
assert len(pf[pf.campaign.eq('r06_c_cal')])==1817
assert len(lk[lk.campaign.eq('r06_p3_pilot')])==10472
assert len(lk[lk.campaign.eq('r06_c_cal')])==1789
effects=pd.read_csv(OUT/'lottery_effects.csv')
for (dataset,cell,sampling),g in effects[effects.cell.ne('POOLED')].groupby(['dataset','cell','sampling']):
    if sampling!='all_anchors':continue
    n=int(g[g.feature.eq('all')].n.iloc[0])
    for field in ['phase','event','grip_state','nonadvance','post_event','state_deviation','milestone']:
        assert g[g.feature.eq(field)].n.sum()==n,(cell,field)
for file in OUT.glob('*.py'):ast.parse(file.read_text())
proposal=(OUT/'PROPOSAL.md').read_text()
assert len(proposal.splitlines())<=400
for link in re.findall(r'\]\(([^)]+)\)',proposal):
    if not link.startswith(('http:','https:')):
        assert (OUT/link).exists(),link
report=dict(arms=checks,ledger_discrepancies=discrepancies,repair_client_crosscheck=timings,profile_anchors=len(pf),blind_probes=len(lk),
            unique_lottery_episode_runs=lot[['run','arm','uid']].drop_duplicates().shape[0],
            lottery_anchors=len(lot),supported_lottery_anchors=int(((lot.p>0)&(lot.p<1)).sum()),
            proposal_lines=len(proposal.splitlines()),checks='PASS',
            caveat='Accounting/partition checks only; not inference validation or causal identification of LOOK.')
(OUT/'verification.json').write_text(json.dumps(report,indent=2))
hashes={str(f.name):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(OUT.iterdir())
        if f.is_file() and f.name!='file_hashes.json'}
(OUT/'file_hashes.json').write_text(json.dumps(hashes,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='arms'},indent=2))
