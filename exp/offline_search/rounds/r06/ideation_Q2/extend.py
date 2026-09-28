"""Budget response fits, K5 identification check and reproducible candidate specifications."""
from pathlib import Path
import json, math
import numpy as np
from scipy.optimize import minimize_scalar, linprog
from scipy.stats import spearmanr
from analyze import R,D,QD,AA,BB,PP,PA,CELLS,pool,avg,cost,paired,write,md

O=Path(__file__).resolve().parent
K=json.loads((O/'key_contrasts.json').read_text())
T=json.loads((O/'task_budgets.json').read_text())

def causal():
    result=[]
    for scale in [50,500]:
        source=O.parents[1]/f'r05/q3_callvalue/results/audited_g{scale}_episodes.json'
        data=json.loads(source.read_text())['episodes'];groups={}
        for r in data:
            # Compare the old exported episode totals with this turn's independent raw audit.
            e=next(e for e in D[r['arm']]['episodes'] if (e['task'],e['init'])==(r['task_id'],r['init']))
            assert [e[k] for k in ['Y','N','M']]==[r[k] for k in ['Y','N','M']]
            groups.setdefault((r['task_id'],r['init']),{})[r['assigned_treatment']]=r
        keys=sorted(groups);delta=np.array([[groups[k]['CALL'][x]-groups[k]['CACHE'][x] for x in ['Y','N','M']] for k in keys])
        w=np.random.default_rng(20260928).multinomial(len(keys),np.full(len(keys),1/len(keys)),10000)
        bs=w@delta/len(keys)
        res=dict(scale=scale,source=str(source),episodes=len(data),clusters=len(keys),
                 exposed=sum(r['exposed'] for r in data),effect=dict(zip(['SR','N','M'],delta.mean(0).tolist())),
                 ci={x:np.quantile(bs[:,i],[.025,.975]).tolist() for i,x in enumerate(['SR','N','M'])})
        res['treatments']={tr:{'sr':float(np.mean([r['Y'] for r in data if r['assigned_treatment']==tr])),
                                 'm':sum(r['M'] for r in data if r['assigned_treatment']==tr)/sum(r['N'] for r in data if r['assigned_treatment']==tr)} for tr in ['CALL','CACHE']}
        result.append(res)
    write('causal.json',result)

def fit_gamma(rows):
    usable=[r for r in rows if r['gap']>0]
    def loss(g):return np.mean([(r['srA']+r['gap']*(1-(1-2*r['mB'])**g)-r['srB'])**2 for r in usable])
    return float(minimize_scalar(loss,bounds=(0,20),method='bounded',options={'xatol':1e-10}).x)

def response():
    g=fit_gamma(K);cv=[]
    for cell in CELLS:
        gg=fit_gamma([r for r in K if r['cell']!=cell])
        for r in K:
            if r['cell']!=cell:continue
            pred=r['srA']+max(r['gap'],0)*(1-(1-2*r['mB'])**gg)
            cv.append(dict(cell=cell,lib=r['lib'],gamma=gg,pred=pred,actual=r['srB'],error=pred-r['srB']))
    alloc=[]
    for cell in CELLS:
        c=.152 if cell.startswith('pi05') else .148
        gg=fit_gamma([r for r in K if r['cell']!=cell])
        for lib in ['50','500']:
            qs=[QD[cell,lib,t] for t in range(10)];h=np.array([q['mean_decisions'] for q in qs]);anchors=np.array([q['mean_anchors_b2'] for q in qs]);d=np.array([q['state_loeo_distance'] for q in qs]);d=d/d.mean()
            for method in ['uniform','library_weighted']:
                target=np.clip((.15*h.sum()-c*anchors.sum())/((1-c)*anchors.sum()),0,1);weights=np.ones(10) if method=='uniform' else d
                lo,hi=0.,1/min(weights)
                for _ in range(80):
                    mid=(lo+hi)/2
                    if np.dot(anchors,np.minimum(1,mid*weights))/anchors.sum()<target:lo=mid
                    else:hi=mid
                p=np.minimum(1,hi*weights)
                ts=[t for t in T if t['cell']==cell and t['lib']==lib]
                # Allow signed task gaps here: increasing policy rate can reduce task SR.
                sr=float(np.mean([t['srA']+t['gap']*(1-(1-p[t['task']])**gg) for t in ts]))
                # Estimate finite-endpoint vision fraction from A and rates on A's lengths.
                a=pool([AA[cell,lib]]);N=np.array([avg(a,t)[1] for t in range(10)]);V=np.array([avg(a,t)[2] for t in range(10)])
                M=V*p;ir=float((c*V.sum()+(1-c)*M.sum())/N.sum());m=float(M.sum()/N.sum())
                alloc.append(dict(cell=cell,lib=lib,method=method,target_ir=.15,p=p.tolist(),gamma=gg,predicted_sr=sr,
                                  predicted_ir=ir,predicted_m=m,predicted_calls_per_episode=float(M.mean()),
                                  assumption='A task lengths/vision counts retained; held-cell gamma; observed task gaps, NOT library-only SR calibration'))
    write('response_model.json',dict(gamma=g,cv=cv,mae=float(np.mean([abs(x['error']) for x in cv])),
                                     max_error=float(max(abs(x['error']) for x in cv)),allocations=alloc))
    return alloc

def extras():
    rows=[]
    for r in K:
        ts=[t for t in T if t['cell']==r['cell'] and t['lib']==r['lib']];br=D[r['B']]
        ep=br['episodes'];pos=[t for t in ts if t['gap']>0]
        rows.append(dict(cell=r['cell'],lib=r['lib'],nonpositive_gap_tasks=[t['task'] for t in ts if t['gap']<=0],
                         calls_in_nonpositive_gap_tasks=sum(t['M'] for t in ts if t['gap']<=0)/sum(t['M'] for t in ts),
                         calls_in_failed_episodes=sum(e['M'] for e in ep if not e['Y'])/sum(e['M'] for e in ep),
                         call_gap_spearman=float(spearmanr([t['gap'] for t in ts],[t['M'] for t in ts]).statistic),
                         gain_call_spearman=float(spearmanr([t['gain'] for t in ts],[t['M'] for t in ts]).statistic),
                         m_per_episode_mean=float(np.mean([e['M']/e['N'] for e in ep])),
                         calls_p90=float(np.quantile([e['M'] for e in ep],.9))))
    write('allocation_diagnostics.json',rows)
    latest=D['r6p1_c10_g_l10_500'];cell=latest['cell'];a=pool([AA[cell,'500']]);b=pool([latest]);g10=pool([BB[cell,'500']])
    write('r6_completed_guard.json',dict(arm=latest['arm'],sr=latest['sr'],ir=latest['ir'],m=latest['m'],M=latest['M'],N=latest['N'],
                                       vs_A=paired(b,a),vs_G10=paired(b,g10),vs_P=paired(b,PP[cell]),
                                       tasks=[dict(task=t,A=avg(a,t)[0],B=avg(b,t)[0],P=avg(PP[cell],t)[0],calls=avg(b,t)[3]) for t in range(10)]))
    # All-arm per-task realized counts, not only the A/B panel.
    all_tasks=[]
    for r in R:
        for t in sorted({e['task'] for e in r['episodes']}):
            ee=[e for e in r['episodes'] if e['task']==t];N=sum(e['N'] for e in ee);M=sum(e['M'] for e in ee)
            all_tasks.append(dict(arm=r['arm'],cell=r['cell'],lib=r['lib'],task=t,sr=np.mean([e['Y'] for e in ee]),
                                  N=N,M=M,m=M/N if N else None,calls_per_episode=M/len(ee),cost_valid=r['cost_valid']))
    write('all_task_counts.json',all_tasks)

def plans(alloc):
    # Four paired allocation tests, eight arms total. Include a strong-library negative-control cell.
    plans=[]
    for r in alloc:
        cell=r['cell'];lib=r['lib'];q=[QD[cell,lib,t] for t in range(10)]
        if (cell,lib) not in [('pi05_l10','50'),('pi05_l10','500'),('groot_l10','50'),('groot_spatial','500')]:continue
        method=r['method']
        plans.append(dict(name=f'q2_budget_{cell}_{lib}_{method}',model=cell.split('_')[0],suite=cell.split('_',1)[1],library=lib,
                          controller='A cache/policy commitment unchanged; Q2 randomized-phase scheduled paid anchors',
                          cache_method=AA[cell,lib]['spec']['method'],cache_kwargs=AA[cell,lib]['kwargs'],
                          stage1_mode='full',policy_committed_controls=10,cache_committed_controls=10,tail_blocks=1,
                          user_knob={'target_ir':.15},allocation=method,calibration='library-only LOEO standardized-state nearest-other-episode RMS; episode/task macro',
                          length_calibration='per-task mean library decisions and mean ceil(decisions/2), equally weighted tasks; solve exact library expected cost ratio',
                          p_call_per_anchor=r['p'],initial_phase='SHA256 UTF-8 seed:task_key:episode_key, first eight bytes big-endian / 2**64; task/episode keys exclude arm name',
                          seed=20260928,rule='call at anchor k iff floor(u+(k+1)*p)>floor(u+k*p), k starts at zero',
                          reset='reset k and phase each episode; no inherited guard overrides',
                          reference_A=AA[cell,lib]['arm'],reference_B=BB[cell,lib]['arm'],reference_P=[x['arm'] for x in PA[cell]],
                          predicted_sr=r['predicted_sr'],predicted_ir=r['predicted_ir'],predicted_m=r['predicted_m'],
                          paired_with=f'q2_budget_{cell}_{lib}_'+('uniform' if method=='library_weighted' else 'library_weighted'),
                          candidate_only=True))
    write('test_plan.json',plans)

def oracle_frontier():
    results=[]
    for cell in CELLS:
        c=.152 if cell.startswith('pi05') else .148
        for lib in ['50','500']:
            modes={'A':pool([AA[cell,lib]]),'B':pool([BB[cell,lib]]),'P':PP[cell]}
            if cell=='groot_l10' and lib=='500':modes['guardB']=pool([D['r6p1_c10_g_l10_500']])
            names=list(modes);target=avg(PP[cell])[0]-.02
            uniform=[]
            for a in names:
                for b in names:
                    x,y=avg(modes[a]),avg(modes[b]);p=0 if x[0]>=target else (target-x[0])/(y[0]-x[0]) if y[0]>x[0] else 2
                    if 0<=p<=1:
                        z=(1-p)*x+p*y;uniform.append(dict(low=a,high=b,p=p,sr=z[0],ir=cost(z,cell),m5=z[3]/z[1],calls=z[3]))
            best=min(uniform,key=lambda r:r['ir'])
            # Fractional LP by bisection on IR; each task receives an independent episode lottery.
            stats=np.array([[avg(modes[n],t) for n in names] for t in range(10)]);Y=stats[:,:,0].flatten();N=stats[:,:,1].flatten();M=stats[:,:,3].flatten();C=(c*stats[:,:,2]+(1-c)*stats[:,:,3]).flatten()
            eq=np.zeros((10,Y.size))
            for t in range(10):eq[t,t*len(names):(t+1)*len(names)]=1
            lo,hi=0.,1.;last=None
            for _ in range(50):
                rho=(lo+hi)/2
                fit=linprog(C-rho*N,A_ub=-Y[None,:],b_ub=[-10*target],A_eq=eq,b_eq=np.ones(10),bounds=(0,1),method='highs',options={'threads':1})
                assert fit.success
                if fit.fun<=0:hi=rho;last=fit.x
                else:lo=rho
            results.append(dict(cell=cell,lib=lib,epsilon=.02,target=target,best_uniform=best,
                                task_oracle=dict(sr=float(last@Y/10),ir=float(last@C/(last@N)),m5=float(last@M/(last@N)),calls=float(last@M/10),
                                                 probabilities={str(t):dict(zip(names,last.reshape(10,len(names))[t].tolist())) for t in range(10)})))
    write('epsilon_frontier.json',results)

if __name__=='__main__':
    causal();alloc=response();extras();plans(alloc);oracle_frontier()
    print(md(['cell','lib','method','SR hypothesis','IR hypothesis','m','calls/ep'],[[r['cell'],r['lib'],r['method'],f"{r['predicted_sr']:.3f}",f"{r['predicted_ir']:.3f}",f"{r['predicted_m']:.3f}",f"{r['predicted_calls_per_episode']:.2f}"] for r in alloc]))
