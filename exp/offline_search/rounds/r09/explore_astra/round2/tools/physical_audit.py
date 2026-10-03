"""Strict episode-addressed physical audit on evaluation inits 20..29 only.

Episode paths are obtained from the discovery-only compact decision IDs. We do
NOT enumerate/read a mixed-init journal or reader.episodes(). Privileged goal
predicates are diagnostic labels, never online inputs or fitted task switches.
"""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from .data import HERE, COMPACT, RUN, CELLS, dataset, subset, validate_identity, dump


def episode_addresses(cell, variant, inits):
    if cell not in CELLS or variant != 'A' or not set(inits) <= set(range(30)):
        raise ValueError('forbidden population')
    arm=f'r8_{cell}_{variant}'
    with np.load(COMPACT/f'{arm}.npz',allow_pickle=False) as z:
        t,i=z['task'],z['init'];validate_identity(t,i)
        mask=np.isin(i,inits)  # SELECT identity before reading any raw episode.
        ids=z['decision_id'][mask]
        addresses={}
        for tt,ii,did in zip(t[mask],i[mask],ids):
            key=(int(tt),int(ii));ep=str(did).split(':')[0]
            if key in addresses and addresses[key] != ep:raise ValueError('conflicting accepted episode')
            addresses[key]=ep
    return arm,addresses


def one(cell):
    arm,addresses=episode_addresses(cell,'A',range(20,30))
    d=dataset(cell);d=subset(d,d['init']>=20)
    records=[]
    for (task,init),key in sorted(addresses.items()):
        directory=RUN/'runs'/arm/'debug/client'/key
        meta=json.loads((directory/'episode.json').read_text())
        if meta['task_uid'] != f'{arm}:eval:{task}:{init}': raise ValueError('identity mismatch')
        chunks=[]
        for path in sorted(directory.glob('controls_*.npz')):
            with np.load(path,allow_pickle=False) as z:
                # Per-episode file has already been admitted by its trusted UID.
                chunks.append({k:z[k] for k in ['predicates','decision_seq','is_settle','eef_pos']})
        controls={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        if len(controls['predicates'])!=meta['n_controls']: raise ValueError('incomplete controls')
        ok=~controls['is_settle'];p=controls['predicates'][ok]
        if not np.isfinite(p).all():raise ValueError('missing predicates')
        truth=p>.5; progress=truth.sum(1); regressed=np.any(truth[:-1]&~truth[1:],axis=1)
        row=dict(task=task,init=init,success=bool(meta['success']),goals=truth.shape[1],
            initial_goals=int(progress[0]),
            max_goals=int(progress.max()),final_goals=int(progress[-1]),
            any_progress=bool((progress>0).any()),any_regression=bool(regressed.any()),
            newly_achieved_goal=bool((truth & ~truth[0]).any()),
            controls=int(ok.sum()),first_progress_control=int(np.flatnonzero(progress>0)[0]) if (progress>0).any() else None)
        take=(d['task']==task)&(d['init']==init)
        seq=controls['decision_seq'][ok]
        gp=((d['base'][take,:,6]>=0)!=(d['teacher'][take,:,6]>=0)).mean(1)
        # Match label at the start of each fresh observation; never imply causal onset.
        starts=np.searchsorted(seq,d['seq'][take],side='left')
        before=progress[np.maximum(starts-1,0)]==0
        row['grip_error_before_progress']=float(gp[before].mean()) if before.any() else None
        row['grip_error_after_progress']=float(gp[~before].mean()) if (~before).any() else None
        records.append(row)
    def summarize(rr):
        if not rr:return dict(n=0)
        return dict(n=len(rr), any_progress=float(np.mean([r['any_progress'] for r in rr])),
            newly_achieved_goal=float(np.mean([r['newly_achieved_goal'] for r in rr])),
            initially_satisfied=float(np.mean([r['initial_goals']>0 for r in rr])),
            any_regression=float(np.mean([r['any_regression'] for r in rr])),
            ends_below_best=float(np.mean([r['max_goals']>r['final_goals'] for r in rr])),
            mean_controls=float(np.mean([r['controls'] for r in rr])))
    result=dict(cell=cell,eval_inits=list(range(20,30)),success=summarize([r for r in records if r['success']]),
        failure=summarize([r for r in records if not r['success']]),records=records)
    dump(HERE/'results'/f'physical_{cell}.json',result)
    print(cell,'failure',result['failure'],flush=True)
    return result


def main():
    out=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(one,c) for c in CELLS]):out.append(f.result())
    dump(HERE/'results/physical_audit.json',sorted(out,key=lambda x:x['cell']))


if __name__=='__main__':main()
