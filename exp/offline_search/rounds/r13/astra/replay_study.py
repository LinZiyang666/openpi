"""B-only controller calibration and out-of-episode/state-path checks."""
from common import *
fence()
import numpy as np
from controller import replay, fit_bias, score

DOSES=(.25,.32,.40,.45)
WIDTHS=(.5,1.,2.,4.,8.)

def npz(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}

def take(p,mask):
    return {k:(v[:,mask] if isinstance(v,np.ndarray) and v.ndim==2 else
               v[mask] if isinstance(v,np.ndarray) and v.ndim==1 else v) for k,v in p.items()}

def split(keys):
    # One deterministic identity-only split, shared by roots/methods/models.
    # It is an offline split, never a task-specific serving parameter.
    return np.array([int.from_bytes(hashlib.sha256(f'R13-cal:{int(t)}:{int(i)}'.encode()).digest()[:8],'big')%2==0
                     for t,i in keys])

def trace_pack(meta):
    p=HERE/'data'/f'{meta["root"]}__{meta["arm"]}.npz'; tr=npz(p)
    looks=tr['vision'].astype(bool)
    if not np.isfinite(tr['distance'][looks]).all(): return None
    E=len(tr['keys']); H=int(tr['epV'].max())
    out={k:np.zeros((H,E),dtype=bool if k in ('alive','guard') else float) for k in
         ('alive','guard','step','distance','risk')}
    out['risk'][:]=np.nan
    for j in range(E):
        ids=np.flatnonzero(looks&(tr['ep']==j)); L=len(ids)
        out['alive'][:L,j]=True
        for k in ('guard','step','distance','risk'): out[k][:L,j]=tr[k][ids]
        assert np.array_equal(out['step'][:L,j],np.arange(0,tr['epN'][j],2)), 'non-nominal cadence: use explicit fallback study'
    out['N']=tr['epN']; out['cal']=split(tr['keys'])
    out['task']=tr['keys'][:,0]; out['init']=tr['keys'][:,1]
    return out

def library_pack(cell):
    d=R11/'astra/data'/cell
    a=npz(d/'signals.npz'); b=npz(d/'pack.npz'); prov=read(d/'provenance.json')
    assert sha(d/'signals.npz')==prov['signals_sha256']
    assert sha(HERE.parents[1]/'r10/subsets'/f'{cell}.npy')==prov['subset_sha256']
    assert sha(HERE.parents[1]/'r10/artifacts'/f'r10_{cell}_G.pkl')==prov['source_fit_sha256']
    assert sha(HERE.parents[1]/'r10/astra/cv'/cell/'rows.npz')==prov['r10_cv_sha256']
    for f in prov['folds']: assert not set(f['train_episodes'])&set(f['heldout_episodes'])
    idx=b['idx'][::2]; alive=idx>=0; idx=np.maximum(idx,0)
    prog=b['prog'][::2]; den=b['den'][::2]
    guard=np.zeros_like(alive)
    guard[1:]=((prog[1:]-prog[:-1])*den[1:]<=.5)&alive[1:]
    E=idx.shape[1]
    out=dict(alive=alive,guard=guard,step=np.broadcast_to(2*np.arange(len(idx))[:,None],idx.shape).copy(),
             distance=a['distance'][idx],risk=a['risk'][idx],N=b['length'],
             task=a['task'][idx[0]],init=b['episodes'],cal=np.arange(E)%2==0)
    return out

def scale(p):
    valid=p['alive']&~p['guard']
    lo,hi=np.quantile(np.log1p(p['distance'][valid]),[.1,.9])
    return dict(lo=float(lo),hi=float(max(hi,lo+.05)))

def metrics(p,cfg,kind='feedback'):
    r=replay(p,**cfg,kind=kind)
    null=replay(p,**cfg,kind=kind,constant=True)
    shuffled=replay(p,**cfg,kind=kind,shuffle=True)
    eligible=p['alive']&~p['guard']
    a=r['calls']&eligible; b=null['calls']&eligible; sh=shuffled['calls']&eligible
    union=(a|b).sum()
    return dict(ir=r['ir'],error=r['ir']-cfg['target'],interior=r['interior'],knobs=r['knobs'],
        placement_change=float((a!=b).sum()/max(union,1)),
        shuffled_change=float((a!=sh).sum()/max((a|sh).sum(),1)),
        capture=r['capture'],max_cache_controls=int(r['maxrun'].max()*10),
        median_max_cache_controls=float(np.median(r['maxrun'])*10),
        ep_ir_p10=float(np.quantile(r['ep_ir'],.1)),ep_ir_p90=float(np.quantile(r['ep_ir'],.9)),
        debt_min=float(r['terminal_debt'].min()),debt_max=float(r['terminal_debt'].max()),
        n=len(p['N']),N=int(p['N'].sum()),guard_floor=float((COSTS[cfg['model']][0]*p['alive'].sum()+
           COSTS[cfg['model']][1]*p['guard'].sum())/p['N'].sum()))

def main():
    metas=[read(p) for p in sorted((HERE/'data').glob('*.json'))]
    packs={}; library={}
    for meta in metas:
        if meta['cell'] not in CELLS: continue
        p=trace_pack(meta)
        if p is not None: packs[(meta['cell'],meta['root'],meta['method'])]=p
    rows=[]; fits=[]; sources={}
    for cell in CELLS:
        model,suite,sz=cell.split('_')
        library[cell]=library_pack(cell)
        root=('r12_devlooks_50L' if suite=='l10' else 'r12_devlooks_50S') if sz=='50' else 'r12_devlooks_200'
        source=(cell,root,'off')
        if source not in packs:
            source=(cell,'library','heldout'); base=library[cell]
        else: base=packs[source]
        sources[cell]=list(source)
        cal=take(base,base['cal']); val=take(base,~base['cal']); scales=scale(cal)
        evals=[('same_path_holdout',val)]
        for key,p in packs.items():
            if key[0]==cell and key!=source:
                evals.append((f'{key[1]}:{key[2]}:heldout',take(p,~p['cal'])))
        # Test geometry shift without using any observed A data.
        for multiplier in (.5, .75, 1.25, 1.5):
            stress={**val,'distance':val['distance']*multiplier}
            evals.append((f'synthetic_distance_x{multiplier}',stress))
        for target in DOSES:
            for kind, widths in [('feedback',WIDTHS),('static',(1.,))]:
                for width in widths:
                    common=dict(model=model,target=target,width=width,**scales)
                    fitted=fit_bias(cal,**common,kind=kind)
                    cfg={**common,'bias':fitted['bias']}
                    fits.append(dict(cell=cell,kind=kind,source=':'.join(source[1:]),**{**cfg,**fitted}))
                    rows.append(dict(cell=cell,kind=kind,width=width,target=target,
                        evaluation='calibration',**metrics(cal,cfg,kind)))
                    for label,p in evals:
                        if len(p['N']): rows.append(dict(cell=cell,kind=kind,width=width,target=target,
                            evaluation=label,**metrics(p,cfg,kind)))
        print(cell,'calibration',source[1:],'episodes',len(cal['N']),flush=True)
    csvout(HERE/'replay.csv',rows);dump(HERE/'candidate_fits.json',fits);dump(HERE/'calibration_sources.json',sources)
    summary=[]
    for width in WIDTHS:
        rr=[r for r in rows if r['kind']=='feedback' and r['width']==width and
            r['evaluation']!='calibration' and 'synthetic' not in r['evaluation'] and sources[r['cell']][1]!='library']
        summary.append(dict(width=width,checks=len(rr),max_abs_error=max(abs(r['error']) for r in rr),
            mean_abs_error=float(np.mean([abs(r['error']) for r in rr])),
            passes=sum(abs(r['error'])<=.02 for r in rr),
            min_interior=min(r['interior'] for r in rr),mean_interior=float(np.mean([r['interior'] for r in rr])),
            min_placement_change=min(r['placement_change'] for r in rr),
            mean_placement_change=float(np.mean([r['placement_change'] for r in rr]))))
    dump(HERE/'replay_summary.json',summary);provenance('replay')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
