"""Unchanged R11 periodic scalars plus paired B power calculations."""
from common import *
fence()
import numpy as np
from scipy.stats import norm
from exp.offline_search.rounds.r11.knob.calibration import tables
from exp.offline_search.rounds.r11.knob.recipe import uniform
from replay_study import DOSES, trace_pack, take

def periodic(p,model,K):
    cv,cm=COSTS[model]; H,E=p['alive'].shape
    run=np.zeros(E,int); last=np.full(E,-1,int); M=np.zeros(E); maxrun=np.zeros(E,int)
    calls=np.zeros((H,E),bool)
    for j in range(E):
        for i in np.flatnonzero(p['alive'][:,j]):
            u=uniform('R11-gap-v1',0,int(p['task'][j]),int(p['init'][j]),int(last[j]),'cap')
            cap=int(np.floor(K))+int(u<K-np.floor(K))
            call=bool(p['guard'][i,j] or run[j]>=cap)
            calls[i,j]=call
            if call: M[j]+=1;last[j]=int(p['step'][i,j]);run[j]=0
            else: run[j]+=1;maxrun[j]=max(maxrun[j],run[j])
    V=p['alive'].sum(0); N=p['N'];cost=cv*V+cm*M
    return dict(ir=float(cost.sum()/N.sum()),calls=calls,max_cache_controls=int(maxrun.max()*10))

def main():
    fits=[]; checks=[]
    for cell in CELLS:
        model,suite,n=cell.split('_')
        a,b,eps,sched=tables(model,suite,int(n))
        for target in DOSES:
            setting,pred,status=sched.solve('periodic',target); K=round(setting,4)
            fits.append(dict(cell=cell,model=model,suite=suite,size=int(n),target=target,setting=K,
                unrounded_setting=setting,pred_ir_library=sched.ir('periodic',K),status=status))
        print(cell,'periodic calibrated',flush=True)
    for mp in sorted((HERE/'data').glob('*.json')):
        meta=read(mp)
        if meta['cell'] not in CELLS: continue
        p=trace_pack(meta)
        if p is None: continue
        val=take(p,~p['cal'])
        for fit in [x for x in fits if x['cell']==meta['cell']]:
            r=periodic(val,meta['model'],fit['setting'])
            checks.append(dict(cell=meta['cell'],root=meta['root'],method=meta['method'],target=fit['target'],
                ir=r['ir'],error=r['ir']-fit['target'],max_cache_controls=r['max_cache_controls']))
    dump(HERE/'periodic_fits.json',fits);csvout(HERE/'periodic_replay.csv',checks)
    # Existing same-dose B arms have unequal realized budgets. Only their paired
    # discordance is used as a planning variance scenario, never as a matched-IR effect.
    outcomes={}
    for mp in sorted((HERE/'data').glob('r11_devknob_50__*.json')):
        m=read(mp)
        if m['method'] not in ('periodic','distance'): continue
        with np.load(mp.with_suffix('.npz'),allow_pickle=False) as z:
            outcomes[(m['cell'],m['method'])]={tuple(k):float(y) for k,y in zip(z['keys'],z['success'])}
    diffs={}; discord=[]
    for cell in [c for c in CELLS if c.endswith('_50')]:
        P=outcomes[(cell,'periodic')];D=outcomes[(cell,'distance')];assert P.keys()==D.keys()
        d=np.array([P[k]-D[k] for k in sorted(P)])
        discord.append(dict(cell=cell,n=len(d),discordance=float(np.mean(d*d)),difference=float(d.mean())))
        for k in P:
            suite=cell.split('_')[1];diffs.setdefault((suite,*k),[]).append(P[k]-D[k])
    # Average models per init; keep all repeated doses/sizes/replicates perfectly
    # correlated in the conservative scenario. Do not count arms as independent.
    grouped={s:np.array([np.mean(v) for k,v in diffs.items() if k[0]==s]) for s in ('l10','spatial')}
    se=np.sqrt(sum(np.var(v,ddof=1)/500 for v in grouped.values())/4)
    zsum=norm.ppf(.975)+norm.ppf(.8)
    # Optimistic scenario: independent replicate noise, same original inits.
    out=dict(variance_source='paired B periodic versus frozen distance at nominal .32; budgets differ',
        episodes_per_arm=500,replicates=2,discordance=discord,
        pooled_se_conservative=float(se),mde80_conservative_pp=float(100*zsum*se),
        mde80_independent_replicate_noise_pp=float(100*zsum*se/np.sqrt(2)),
        perfect_dependence_warning='Extra doses and libraries do not automatically increase effective sample size.',
        arbitrary_discordance_scenarios=[dict(discordance=q,mde80_pp=float(100*zsum*np.sqrt(q/1000))) for q in (.05,.10,.20)])
    dump(HERE/'power.json',out);provenance('periodic_power');print(json.dumps(out,indent=2))

if __name__=='__main__': main()
