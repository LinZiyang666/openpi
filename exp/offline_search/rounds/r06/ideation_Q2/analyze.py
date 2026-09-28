"""Curves, paired contrasts, proxy validation, and identifiable episode routing baselines."""
from pathlib import Path
import collections, csv, json, math
import numpy as np
from scipy.stats import spearmanr, binomtest
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

O=Path(__file__).resolve().parent
R=json.loads((O/'arms.json').read_text()); D={r['arm']:r for r in R}
Q=json.loads((O/'library_quality.json').read_text()); QD={(r['cell'],r['lib'],r['task']):r for r in Q}
CELLS=['pi05_l10','pi05_spatial','groot_l10','groot_spatial']

def write(name,x): (O/name).write_text(json.dumps(x,indent=1))
def md(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def pool(arms):
    arms=[D[a] if isinstance(a,str) else a for a in arms];p={}
    for r in arms:
        for e in r['episodes']:
            k=(e['task'],e['init']);p.setdefault(k,[]).append([e['Y'],e['N']*r['L']/5,e['V'],e['M']])
    return {k:np.mean(v,axis=0) for k,v in p.items()}
def paired(a,b):
    ks=sorted(set(a)&set(b));d=np.array([a[k][0]-b[k][0] for k in ks]);n=len(d)
    u,c=np.unique(d,return_counts=True);bs=np.random.default_rng(20260928).multinomial(n,c/n,10000)@u/n
    return dict(delta=float(d.mean()),ci=np.quantile(bs,[.025,.975]).tolist(),n=n,
                wins=int(sum(d>0)),losses=int(sum(d<0)),p=(float(binomtest(sum(d>0),sum(d!=0),.5).pvalue) if sum(d!=0) else 1.) if np.all(np.isin(d,[-1,0,1])) else None)
def avg(p,task=None):return np.mean([v for (t,i),v in p.items() if task is None or t==task],axis=0)
def cost(z,cell):
    c=.152 if cell.startswith('pi05') else .148
    return (c*z[2]+(1-c)*z[3])/z[1]

PA={c:[r for r in R if r['cell']==c and r['family']=='policy_L10'] for c in CELLS}
PP={c:pool(PA[c]) for c in CELLS}
AA={(r['cell'],r['lib']):r for r in R if r['family']=='A'}
BB={(r['cell'],r['lib']):r for r in R if (r['family']=='B' and r['cell'].startswith('pi05')) or r['family']=='G10'}

def key_table():
    rows=[];tasks=[]
    for cell in CELLS:
        for lib in ['50','500']:
            ar=AA[cell,lib];br=BB[cell,lib];a=pool([ar]);b=pool([br]);p=PP[cell]
            za,zb,zp=avg(a),avg(b),avg(p);gap=zp[0]-za[0];gain=zb[0]-za[0]
            gamma=math.log(1-gain/gap)/math.log(1-2*br['m5']) if gap>0 and 0<gain<gap else None
            row=dict(cell=cell,lib=lib,A=ar['arm'],B=br['arm'],P=[x['arm'] for x in PA[cell]],
                     srA=za[0],srB=zb[0],srP=zp[0],irA=cost(za,cell),irB=cost(zb,cell),irP=cost(zp,cell),
                     mB=br['m'],callsB=zb[3],gap=gap,gain=gain,fraction_closed=gain/gap if gap>0 else None,
                     slope_A_B=gain/br['m5'],slope_B_P=(zp[0]-zb[0])/(.5-br['m5']),gamma=gamma,
                     paired_B_A=paired(b,a),paired_P_A=paired(p,a))
            if gamma:
                row['m_for_gap_fractions']={str(f):.5*(1-(1-f)**(1/gamma)) for f in [.5,.8,.9]}
                row['m_for_epsilon_02']=.5*(1-(min(1,.02/gap))**(1/gamma))
            rows.append(row)
            for t in range(10):
                x,y,z=avg(a,t),avg(b,t),avg(p,t);ee=[e for e in br['episodes'] if e['task']==t]
                tasks.append(dict(cell=cell,lib=lib,task=t,srA=x[0],srB=y[0],srP=z[0],gap=z[0]-x[0],
                                  gain=y[0]-x[0],M=y[3],N=y[1],m=y[3]/y[1],ir=cost(y,cell),
                                  miss_median=float(np.median([e['M'] for e in ee])),miss_p90=float(np.quantile([e['M'] for e in ee],.9)),
                                  failure_miss_fraction=sum(e['M'] for e in ee if not e['Y'])/sum(e['M'] for e in ee) if sum(e['M'] for e in ee) else 0,
                                  **{k:QD[cell,lib,t][k] for k in ['state_loeo_distance','action_loeo','inverse_sqrt_episodes']}))
    write('key_contrasts.json',rows);write('task_budgets.json',tasks)
    with (O/'task_budgets.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(tasks[0]));w.writeheader();w.writerows(tasks)
    return rows,tasks

def slopes():
    groups={
      'pi05_l10_50_periodic':['oscl50_p_l10_cl2','r4_p_l10_per6_50','r3mx_p_l10_perk5','r3mx_p_l10_perk3'],
      'pi05_l10_500_periodic':['oscl500_p_l10_cl2','r4_p_l10_per12_500','r4_p_l10_per8_500'],
      'pi05_spatial_50_periodic':['oscl50_p_sp_cl2','r3mx_p_sp_perk3'],
      'pi05_spatial_500_periodic':['oscl500_p_sp_cl2','r4_p_sp_per12_500'],
      'pi05_l10_50_quantile':['r3mx_p_l10_awm_h70','r3mx_p_l10_awm_h50'],
    }; rr=[]
    for name,arms in groups.items():
        for low,high in zip(arms,arms[1:]):
            a,b=D[low],D[high];pa=paired(pool([b]),pool([a]));dm=b['m5']-a['m5']
            rr.append(dict(group=name,low=low,high=high,dm=dm,delta=pa['delta'],slope=pa['delta']/dm,
                           slope_ci=[x/dm for x in pa['ci']],paired=pa))
    write('marginal_returns.json',rr)
    return rr

def regression():
    records=[]
    for (cell,lib),arm in AA.items():
        a=pool([arm]);p=PP[cell]
        for t in range(10):records.append(dict(QD[cell,lib,t],gap=avg(p,t)[0]-avg(a,t)[0]))
    results=[]
    for feature in ['state_loeo_distance','action_loeo','state_residual_loeo','inverse_sqrt_episodes']:
        rr=[r for r in records if r.get(feature) is not None]
        X=np.array([[1.,r[feature]] for r in rr]);y=np.array([r['gap'] for r in rr])
        beta=np.linalg.lstsq(X,y,rcond=None)[0];pred=X@beta
        result=dict(feature=feature,n=len(y),beta=beta.tolist(),spearman=float(spearmanr(X[:,1],y).statistic),
                    r2=float(1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2)),cv={})
        for scheme in ['scene','cell','suite']:
            labels=[(r['cell'].split('_',1)[1],r['task']) if scheme=='scene' else (r['cell'] if scheme=='cell' else r['cell'].split('_',1)[1]) for r in rr]
            pp=np.zeros(len(y));base=np.zeros(len(y))
            for label in set(labels):
                test=np.array([v==label for v in labels]);b=np.linalg.lstsq(X[~test],y[~test],rcond=None)[0]
                pp[test]=X[test]@b;base[test]=y[~test].mean()
            result['cv'][scheme]=dict(mae=float(np.mean(abs(pp-y))),null_mae=float(np.mean(abs(base-y))),
                                     rmse=float(np.sqrt(np.mean((pp-y)**2))))
        results.append(result)
    # Validate across all applicable arms, while acknowledging repeated labels share a task.
    arm_tests=[]
    for fam in ['A','B','G10','CL2','periodic','guard','quantile']:
        rr=[]
        for r in R:
            if r['family']!=fam or not r['cost_valid'] or 'b0' in r['arm']:continue
            p=pool([r]);ref=PP[r['cell']]
            for t in range(10):
                q=QD.get((r['cell'],r['lib'],t))
                if q and avg(p,t).size:rr.append([q['state_loeo_distance'],avg(ref,t)[0]-avg(p,t)[0]])
        if rr:
            ar=np.array(rr);arm_tests.append(dict(family=fam,n_task_arms=len(rr),rho=float(spearmanr(ar[:,0],ar[:,1]).statistic)))
    write('quality_validation.json',dict(regressions=results,arm_tests=arm_tests,records=records))
    return results

def route(p,cell,lib):
    a=pool([AA[cell,lib]]);b=PP[cell];zz=[]
    for t in range(10):zz.append((1-p[t])*avg(a,t)+p[t]*avg(b,t))
    z=np.mean(zz,axis=0)
    return dict(sr=z[0],ir=cost(z,cell),m5=z[3]/z[1],calls=z[3],p=list(map(float,p)))

def predictions():
    rows=[]
    for cell in CELLS:
        c=.152 if cell.startswith('pi05') else .148
        for lib in ['50','500']:
            qs=[QD[cell,lib,t] for t in range(10)];h=np.array([q['mean_decisions'] for q in qs]);anchors=np.array([q['mean_anchors_b2'] for q in qs]);r=np.array([q['state_loeo_distance'] for q in qs]);r=r/r.mean()
            for rho in [.10,.15,.25,.40]:
                # Pure/A lottery; library predicts equal endpoint durations and anchor counts.
                target=np.clip((rho*h.sum()-c*anchors.sum())/((1-c)*anchors.sum()),0,1)
                for method in ['uniform','library_weighted']:
                    weights=np.ones(10) if method=='uniform' else r
                    lo,hi=0.,1/max(min(weights),1e-12)
                    for _ in range(80):
                        mid=(lo+hi)/2;p=np.minimum(1,mid*weights)
                        if np.dot(anchors,p)/anchors.sum()<target:lo=mid
                        else:hi=mid
                    p=np.minimum(1,hi*weights);ev=route(p,cell,lib)
                    rows.append(dict(cell=cell,lib=lib,target_ir=rho,method=method,library_m5=target*anchors.sum()/h.sum(),**ev))
            # In-sample target-loss frontier: an explicit descriptive interpolation, not a deployment guarantee.
            a=avg(pool([AA[cell,lib]]));b=avg(PP[cell]);gap=b[0]-a[0]
            for eps in [.01,.02,.05]:
                prob=0 if gap<=eps else 1-eps/gap
                rows.append(dict(cell=cell,lib=lib,epsilon=eps,method='measured_uniform_epsilon',**route(np.repeat(prob,10),cell,lib)))
    write('predictions.json',rows)
    # Sparse pilot: first five available initializations/task fit task gaps, held-out 45 evaluate; sensitivity over 10 disjoint folds.
    pilot=[]
    for cell in CELLS:
        for lib in ['50','500']:
            a=pool([AA[cell,lib]]);b=PP[cell]
            for fold in range(10):
                gaps=[]
                for t in range(10):
                    ks=[k for k in a if k[0]==t and k[1]//5==fold]
                    gaps.append(max(0,np.mean([b[k][0]-a[k][0] for k in ks])))
                gaps=np.array(gaps);p=np.zeros(10);need=max(0,gaps.mean()-.02)
                # Minimize expected policy calls using pilot lengths; fractional knapsack with 1/T mass.
                pc=np.array([np.mean([b[k][3] for k in b if k[0]==t and k[1]//5==fold]) for t in range(10)])
                for t in sorted(range(10),key=lambda t:(-gaps[t]/max(pc[t],1e-12),t)):
                    if gaps[t]>0:
                        p[t]=min(1,need*10/gaps[t]);need-=p[t]*gaps[t]/10
                zz=[];pure=[]
                for k in a:
                    if k[1]//5!=fold:zz.append((1-p[k[0]])*a[k]+p[k[0]]*b[k]);pure.append(b[k])
                z=np.mean(zz,axis=0);ps=np.mean(pure,axis=0)
                pilot.append(dict(cell=cell,lib=lib,fold=fold,epsilon=.02,sr=z[0],ir=cost(z,cell),loss=ps[0]-z[0],p=p.tolist()))
    write('pilot_validation.json',pilot)
    return rows,pilot

def plots():
    fams=['A','B','G10','CL2','periodic','guard','quantile','randomized_guard','policy_L10','policy_L5']
    colors=dict(zip(fams,['#187a3c','#bf3d2a','#ed8c00','#777777','#4352b5','#ba64b1','#047f9d','#93662c','#111111','#aaaaaa']))
    curves=[]
    for metric in ['m','m5','ir']:
        fig,axes=plt.subplots(4,2,figsize=(13,15),sharex=True,sharey='row')
        for i,cell in enumerate(CELLS):
            for j,lib in enumerate(['50','500']):
                ax=axes[i,j]
                rr=[r for r in R if r['cell']==cell and r['lib'] in (lib,'-') and r['n']==500 and r['cost_valid'] and 'k2' not in r['arm'] and r['stage1_mode']=='full']
                for fam in fams:
                    pts=[r for r in rr if r['family']==fam and 'b0' not in r['arm']]
                    if not pts:continue
                    pts.sort(key=lambda r:r[metric]);ax.scatter([r[metric] for r in pts],[r['sr'] for r in pts],c=colors[fam],s=38,label=fam,marker='s' if fam.startswith('policy') else 'o',zorder=3)
                    if fam=='periodic':ax.plot([r[metric] for r in pts],[r['sr'] for r in pts],c=colors[fam],alpha=.6)
                extra=[r for r in rr if r['family'] not in fams or 'b0' in r['arm']]
                ax.scatter([r[metric] for r in extra],[r['sr'] for r in extra],s=10,c='#bbbbbb',alpha=.4)
                # A→B→P is a descriptive arm-level curve, with cost denominator documented.
                base=[AA[cell,lib],BB[cell,lib]]
                x=[r[metric] for r in base]+[1 if metric=='m' else .5];y=[r['sr'] for r in base]+[avg(PP[cell])[0]]
                ax.plot(x,y,color='#222222',ls=':',lw=1.2)
                ax.set_title(f'{cell}, library {lib}');ax.grid(alpha=.2);ax.set_ylabel('success rate');ax.set_xlabel(metric);ax.set_xlim(-.02,1.03)
                if metric=='ir':
                    for r in sorted(rr,key=lambda r:r['ir']):
                        dominated=any(v['ir']<=r['ir'] and v['sr']>=r['sr'] and (v['ir']<r['ir'] or v['sr']>r['sr']) for v in rr)
                        if not dominated:curves.append({k:r[k] for k in ['cell','lib','arm','family','sr','m','m5','ir']})
        handles={}
        for ax in axes.flat:
            h,l=ax.get_legend_handles_labels();handles.update(zip(l,h))
        fig.legend(handles.values(),handles.keys(),loc='upper center',ncol=5);fig.tight_layout(rect=(0,0,1,.955));fig.savefig(O/f'sr_vs_{metric}.png',dpi=150);plt.close(fig)
    write('observed_frontiers.json',curves)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,cell in zip(axes,['pi05_l10','pi05_spatial']):
        rr=sorted([r for (c,l),r in AA.items() if c==cell],key=lambda r:int(r['lib']))
        ax.plot([int(r['lib']) for r in rr],[r['sr'] for r in rr],'o-');ax.axhline(avg(PP[cell])[0],color='k',ls='--',label='policy L10')
        ax.set(title=cell,xlabel='library episodes (nominal)',ylabel='SR at zero MISS');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(O/'library_size_curve.png',dpi=150);plt.close(fig)

def main():
    keys,tasks=key_table();ss=slopes();regs=regression();pred,pilot=predictions();plots()
    lines=['# Computed tables','',md(['Cell','Library','A SR','B/G10 SR','P10 SR','m(B)','IR(B)','Calls/ep','B−A pp','Gap closed','Slope SR/m'],[[r['cell'],r['lib'],f"{r['srA']:.3f}",f"{r['srB']:.3f}",f"{r['srP']:.3f}",f"{r['mB']:.4f}",f"{r['irB']:.4f}",f"{r['callsB']:.3f}",f"{100*r['gain']:.1f}",f"{r['fraction_closed']:.3f}" if r['fraction_closed'] is not None else 'n/a',f"{r['slope_A_B']:.3f}"] for r in keys]),'']
    lines+=[md(['Sequence','Low','High','Δm','ΔSR pp','SR/m','95% slope interval'],[[r['group'],r['low'],r['high'],f"{r['dm']:.4f}",f"{100*r['delta']:.1f}",f"{r['slope']:.3f}",str(np.round(r['slope_ci'],3))] for r in ss]),'']
    lines+=[md(['Proxy','N','Spearman','R²','Held-scene MAE/null','Held-cell MAE/null','Held-suite MAE/null'],[[r['feature'],r['n'],f"{r['spearman']:.3f}",f"{r['r2']:.3f}",*[f"{r['cv'][k]['mae']:.3f}/{r['cv'][k]['null_mae']:.3f}" for k in ['scene','cell','suite']]] for r in regs]),'']
    lines+=[md(['Cell','Lib','Library IR=.15: SR','Realized IR','m5','Uniform SR','Uniform IR'],[[r['cell'],r['lib'],f"{r['sr']:.4f}",f"{r['ir']:.4f}",f"{r['m5']:.4f}",*[f"{u[k]:.4f}" for k in ['sr','ir']]] for r in pred if r['method']=='library_weighted' and r['target_ir']==.15 for u in pred if u['method']=='uniform' and u['target_ir']==.15 and u['cell']==r['cell'] and u['lib']==r['lib']]),'']
    for cell in CELLS:
        for lib in ['50','500']:
            lines += [f'## {cell} {lib}',md(['Task','A','B','P10','P−A','B−A','MISS/ep','m','MISS p90','LOEO d'],[[r['task'],*[f"{r[k]:.3f}" for k in ['srA','srB','srP','gap','gain','M','m','miss_p90','state_loeo_distance']]] for r in tasks if r['cell']==cell and r['lib']==lib]),'']
    (O/'TABLES.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:8]))
    print('pilot target violations',sum(p['loss']>.02 for p in pilot),'/',len(pilot))

if __name__=='__main__':main()
