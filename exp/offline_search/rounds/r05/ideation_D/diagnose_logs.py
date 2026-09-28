"""Read-only R5-D closed-loop extraction. Run with documented CPU affinity/env."""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/weiland/trace_runs/os_closed_loop')
STORE = Path('/home/weiland/trace_runs/offline_search_store')
FEATURES = ['os_flags','os_reason','stuck_n','noprog_span','noprog_n','lag','overtime',
            'disp','dnn','vis','vself','top1_prog','pred_err','vote','gexec','gprop',
            'displacement_residual','motion','os_force_miss']

def read_lines(path):
    with path.open() as f:
        for line in f:
            try: yield json.loads(line)
            except json.JSONDecodeError: continue

def extract(root, arm):
    base = ROOT / root / 'runs' / arm
    comp = {r['task_uid']: r for r in read_lines(base/'client/journal.jsonl')
            if r.get('accepted') and r.get('status') in ('done','failed') and not r.get('error')}
    records = {}
    for path in sorted(base.glob('server_*/decisions_*.jsonl')):
        for r in read_lines(path):
            if r.get('ev') != 'dec' or r.get('uid') not in comp: continue
            c = comp[r['uid']]
            if c.get('attempt') is not None and r.get('attempt') is not None and int(c['attempt']) != int(r['attempt']): continue
            records[r['uid'],int(r['step'])] = r
    uids = sorted(comp)
    ui = {u:i for i,u in enumerate(uids)}
    rs = sorted(records.values(), key=lambda d:(d['uid'],d['step']))
    n = len(rs)
    out = dict(ep=np.array([ui[d['uid']] for d in rs],np.int32),
               step=np.array([d['step'] for d in rs],np.int32),
               task=np.array([int(d['uid'].split(':')[-2]) for d in rs],np.int16),
               init=np.array([int(d['uid'].split(':')[-1]) for d in rs],np.int16),
               success=np.array([bool(comp[d['uid']].get('success')) for d in rs]),
               hit=np.array([d.get('hit',True) for d in rs]),
               vision=np.array([d.get('vision',True) for d in rs]),
               state=np.full((n,8),np.nan,np.float32),action=np.full((n,5,7),np.nan,np.float32),
               rows=np.full((n,16),-1,np.int32),weights=np.full((n,16),np.nan,np.float32),
               top1=np.array([d.get('top1',-1) for d in rs],np.int32),
               feats=np.full((n,len(FEATURES)),np.nan,np.float32))
    for i,d in enumerate(rs):
        if d.get('robot_state') is not None: out['state'][i] = d['robot_state'][:8]
        a = d.get('served_head') or d.get('a_exec')
        if a is not None: out['action'][i] = np.array(a)[:5,:7]
        rr,ww = d.get('rows',[]), d.get('weights',[])
        if rr and ww:
            k = min(len(rr),16); out['rows'][i,:k]=rr[:k]; out['weights'][i,:k]=ww[:k]
        ex = d.get('extras',{})
        for j,f in enumerate(FEATURES):
            if ex.get(f) is not None: out['feats'][i,j] = ex[f]
    np.savez_compressed(HERE/f'{arm}.npz',**out)
    unique = np.r_[True,np.diff(out['ep']) != 0]
    report = dict(root=root,arm=arm,n_journal=len(comp),n_logged_episodes=int(unique.sum()),
        n_decisions=n,success=int(out['success'][unique].sum()),
        sr=float(out['success'][unique].mean()),miss=int((~out['hit']).sum()),
        vision=int(out['vision'].sum()),ir_pi=float((.152*out['vision']+.848*(~out['hit'])).mean()),
        finite_state=int(np.isfinite(out['state']).all(1).sum()),
        finite_action=int(np.isfinite(out['action']).all((1,2)).sum()),
        tasks={str(t):dict(n=int((unique&(out['task']==t)).sum()),
                         success=int((unique&(out['task']==t)&out['success']).sum())) for t in np.unique(out['task'])})
    return report

if __name__ == '__main__':
    inventory=[]
    for p in sorted(ROOT.glob('r04_*/runs/*/summary.json')):
        d=json.loads(p.read_text())
        inventory.append(dict(root=p.parents[2].name,arm=p.parent.name,
            **{k:d.get(k) for k in ['model','suite','complete','success','sr','per_task']}))
    (HERE/'inventory.json').write_text(json.dumps(inventory,indent=2))
    reports=[]
    selected = [('r04_k7',p.parent.name) for p in sorted((ROOT/'r04_k7/runs').glob('*/summary.json'))]
    selected += [('r04_cost',f'r4f_p_{s}_inf_s1001') for s in ['sp','l10']]
    for root,arm in selected:
        r=extract(root,arm); reports.append(r); print(json.dumps(r),flush=True)
    (HERE/'logs_summary.json').write_text(json.dumps(reports,indent=2))
