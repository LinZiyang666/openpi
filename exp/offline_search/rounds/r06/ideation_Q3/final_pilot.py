"""Summarize unchanged frozen outputs and project continuation from measured scores.

Projections are planning, not new confirmatory estimators. The separately labeled
guard-state contrasts are post-hoc; they never alter the frozen nomination rule.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/tables')
OUT=HERE/'final_pilot'
sys.path.insert(0,str(HERE))
from analyze_pilot import (CLUSTER, EPKEY, GATES, PRIMARY, call_effect, clean, expected_slots, sha)
from exp.offline_search.rounds.r06.p3_profiling.pilot_stats import icc


def save(name,x):
    (OUT/name).write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')


def projection(frame,value,label,kind):
    groups=list(frame.groupby(CLUSTER))
    ic=icc([g[value].to_numpy() for _,g in groups],[key[0] for key,_ in groups])
    mb,mw=ic.get('ms_between',np.nan),ic.get('ms_within',np.nan)
    # Balanced pilot: three repeated seed blocks in every task/init.
    assert all(len(g)==3 for _,g in groups)
    u=(mb-mw)/3
    base=dict(cell=frame.cell.iloc[0],contrast=label,kind=kind,N=len(frame),clusters=len(groups),
              raw_variance=float(frame[value].var(ddof=1)),raw_second_moment=float((frame[value]**2).mean()),
              within_task_variance=u+mw,icc_raw=ic.get('icc'),sigma_init_raw=u,sigma_within=mw,
              ms_between=mb,ms_within=mw,icc_planning=max(0,u)/(max(0,u)+mw) if max(0,u)+mw>0 else np.nan,
              projection_status='measured_components_conditional_on_stable_init_population' if max(0,u)+mw>0 else 'zero_pilot_variance_unresolved_not_zero_future_risk')
    rows=[]
    for stage in ('pilot','full','continuation'):
        for split in ('all','calibration','validation'):
            repeats={}
            for block in (0,1,2):
                for task,init in expected_slots(block,stage):
                    if split=='calibration' and init%5!=0:continue
                    if split=='validation' and init%5==0:continue
                    repeats[task,init]=repeats.get((task,init),0)+1
            rr=np.asarray(list(repeats.values()),float);n=rr.sum();k=len(rr);df=k-10
            # Negative measured covariance is retained in raw sensitivity, while
            # conservative planning clips only the between-init variance component.
            vr=u*(rr@rr)/(n*n)+mw/n
            vc=max(0,u)*(rr@rr)/(n*n)+mw/n
            variance_ref=max(0,u)+mw
            row={**base,'stage':stage,'split':split,'target_episodes':int(n),'target_clusters':k,'df':df,
                 'se_raw_covariance':np.sqrt(max(0,vr)),'se_conservative':np.sqrt(max(0,vc)),
                 'n_eff_conservative':variance_ref/vc if vc>0 else np.nan}
            for mult,key in ((1,'95'),(64,'sim64')):
                row['halfwidth_'+key]=t.ppf(1-.05/(2*mult),df)*np.sqrt(vc) if df>0 and vc>0 else np.nan
            row['mde80']=2.801621*np.sqrt(vc) if vc>0 else np.nan
            rows.append(row)
    return rows


def grouped_score(g,episodes,mask,sign):
    """Linearized episode totals, with zero for every unvisited episode slot."""
    tmp=g[EPKEY].copy()
    tmp['num']=np.asarray(mask,float)*np.asarray(sign,float)*g.Y.to_numpy(float)
    tmp['den']=np.asarray(mask,float)
    agg=tmp.groupby(EPKEY)[['num','den']].sum()
    result=episodes[EPKEY+['cell','task_id','init','block']].merge(agg,on=EPKEY,how='left',validate='one_to_one')
    result[['num','den']]=result[['num','den']].fillna(0)
    total=result.den.sum()
    if total==0:
        result['score']=np.nan
        return np.nan,result
    tau=result.num.sum()/total
    result['score']=(result.num-tau*result.den)/(total/len(episodes))
    assert abs(result.score.mean())<1e-10
    return tau,result


def main():
    OUT.mkdir(exist_ok=False)
    frozen=HERE/'pilot_all'
    fx=pd.read_csv(frozen/'effects.csv')
    primary=fx[fx.primary & fx.outcome.eq('Y')].copy()
    assert len(primary)==64
    primary.to_csv(OUT/'primary64.csv',index=False)
    nomination=[]
    for cell in sorted(primary.cell.unique()):
        for gate in GATES:
            names=[f'factorial:{gate}:high','coverage_enrichment' if gate=='coverage_high' else f'factorial:{gate}:enrichment']
            entry=dict(cell=cell,gate=gate)
            for label,name in zip(('inside','enrichment'),names):
                z=fx[fx.cell.eq(cell)&fx.split.eq('calibration')&fx.outcome.eq('Y')&fx.contrast.eq(name)]
                assert len(z)==1
                row=z.iloc[0]
                entry.update({f'{label}_{k}':row.get(k) for k in ['estimate','df','clusters','cluster_ess_a','cluster_ess_b','nomination_support','lo_selection','status']})
            nomination.append(entry)
    pd.DataFrame(nomination).to_csv(OUT/'nomination48_audit.csv',index=False)
    allplans=[];scores=[];eq=[];guarddiag=[];guardcausal=[];armowner=[];audit=[]
    source_cols=set(EPKEY+['step','Y','task_id','init','future_misses',
        'assignment.cohort','assignment.replicate','assignment.override','assignment.nominal_propensity','assignment.coin_call',
        'assignment.pre_guard_call','assignment.duration_choice','assignment.duration_probability','assignment.hold_choice','assignment.hold_probability',
        'assignment.scheduled_trigger','assignment.delay_choice','assignment.delay_probability','retrieval.d1_loeo_quantile',
        'guards.inputs_outputs.os_flags','guards.inputs_outputs.motion','guards.thresholds.m_thr',
        'calibration.statistics.stuck','calibration.statistics.progress','calibration.p_values.progress','calibration.p_values.coverage'])
    for cell in sorted(primary.cell.unique()):
        e=pd.read_csv(ROOT/cell/'episodes.csv')
        e['cell']=cell;e['block']=e['provenance.run_block'].astype(int)
        # Completed manifest names encode cohort unambiguously.
        prefix='r6p3v2_'+cell+'_'
        e['cohort']=e.arm.str.removeprefix(prefix).str.replace(r'_r[012]$','',regex=True)
        assert set(e.cohort)=={'A','B','P10','factorial','window','dose_mix','dose125','dose25','dose50'}
        c=.152 if cell.startswith('pi05') else .148
        for cohort,g in e.groupby('cohort'):
            v,m,n=g.anchors.sum(),g.misses.sum(),g.decisions.sum()
            armowner.append(dict(cell=cell,cohort=cohort,episodes=len(g),SR=g.Y.mean(),vision=int(v),misses=int(m),decisions=int(n),
                                 owner_IR_request=(c*v+(1-c)*m)/n,owner_IR_actual=(c*v+(1-c)*m)/(g.active_controls.sum()/5)))
        for left,right in [('B','A'),('B','P10'),('factorial','B'),('window','A'),('dose125','B'),('dose25','B'),('dose50','B')]:
            l=e[e.cohort.eq(left)];r=e[e.cohort.eq(right)]
            p=l.merge(r[['task_id','init','block','Y']],on=['task_id','init','block'],suffixes=('','_ref'),validate='one_to_one')
            assert len(p)==60
            p['difference']=p.Y-p.Y_ref
            allplans.extend(projection(p,'difference',f'{left}-{right}','paired_episode_SR'))
        # A second, narrow read supplies episode scores and post-hoc guard diagnostics.
        a=pd.read_csv(ROOT/cell/'anchors.csv',usecols=lambda col:col in source_cols,low_memory=False)
        a['cell']=cell;a['episode_weight']=1/60
        g=a[a['assignment.cohort'].eq('factorial') & a['assignment.override'].eq('coin') &
            a['assignment.nominal_propensity'].between(0,1,inclusive='neither')].copy()
        e_fac=e[e.cohort.eq('factorial')].copy()
        z=g['assignment.coin_call'].astype(int).to_numpy();p=g['assignment.nominal_propensity'].to_numpy()
        sign=z/p-(1-z)/(1-p)
        guard=g['assignment.pre_guard_call'].to_numpy(bool)
        cov=g['retrieval.d1_loeo_quantile'].to_numpy()>=2/3
        results={}
        for name,mask in [('factorial_call',np.ones(len(g))),('factorial_at_guard',guard),('factorial_pre_guard',~guard)]:
            results[name]=grouped_score(g,e_fac,mask,sign)
        for name,mask in [('coverage_enrichment',cov),('guard_enrichment',guard)]:
            hi,h=grouped_score(g,e_fac,mask,sign);lo,l=grouped_score(g,e_fac,~mask,sign)
            h['score']=h.score-l.score
            results[name]=(hi-lo,h)
        for name,col,pa,first,last in [('duration10_vs5','assignment.duration_choice','assignment.duration_probability',10,5),
                                      ('hold3_vs1','assignment.hold_choice','assignment.hold_probability',3,1)]:
            root=(g[col].eq(first).to_numpy(int)-g[col].eq(last).to_numpy(int))/g[pa].to_numpy()
            results[name]=grouped_score(g,e_fac,z,root)
        w=a[a['assignment.cohort'].eq('window') & a['assignment.scheduled_trigger']].copy()
        s=(w['assignment.delay_choice'].eq(0).to_numpy(int)-w['assignment.delay_choice'].eq(2).to_numpy(int))/w['assignment.delay_probability'].to_numpy()
        results['window_delay0_vs2']=grouped_score(w,e[e.cohort.eq('window')],np.ones(len(w)),s)
        for name,(tau,frame) in results.items():
            old=primary[primary.cell.eq(cell)&primary.contrast.eq(name)].iloc[0]
            assert np.isclose(tau,old.estimate,atol=1e-12), (cell,name,tau,old.estimate)
            eq.append(dict(cell=cell,contrast=name,reconstructed=tau,frozen=old.estimate,passed=True))
            frame['contrast']=name
            scores.append(frame[['cell','task_id','init','block','contrast','score']])
            allplans.extend(projection(frame,'score',name,'linearized_excursion_SR'))
        # These were NOT frozen: explanatory guard-specific interactions.
        flags=g['guards.inputs_outputs.os_flags'].fillna(0).astype(int)
        npmask=(flags & 8)>0
        still=g['calibration.statistics.stuck']>0
        risk=g['calibration.p_values.coverage']<=.2
        for stratum,mask in [('noprog_flag',npmask),('noprog_still',npmask&still),('noprog_not_still',npmask&~still),
                             ('noprog_error_high',npmask&risk),('noprog_error_not_high',npmask&~risk)]:
            rec,_=call_effect(g[mask],'Y')
            guardcausal.append(dict(cell=cell,stratum=stratum,exploratory=True,**rec))
        for cohort in ('B','factorial','A'):
            ac=a[a['assignment.cohort'].eq(cohort)]
            mask=(ac['guards.inputs_outputs.os_flags'].fillna(0).astype(int)&8)>0
            ss=ac[mask]
            guarddiag.append(dict(cell=cell,cohort=cohort,anchors=len(ac),noprog_flags=len(ss),
                                  frac_flag_with_still=float((ss['calibration.statistics.stuck']>0).mean()),
                                  frac_flag_with_error_p_le20=float((ss['calibration.p_values.coverage']<=.2).mean()),
                                  frac_flag_with_progress_p_le20=float((ss['calibration.p_values.progress']<=.2).mean()),
                                  mean_motion_over_threshold=float((ss['guards.inputs_outputs.motion']/ss['guards.thresholds.m_thr']).replace([np.inf,-np.inf],np.nan).mean())))
        # Joint rerun must exactly reproduce each already-produced per-cell result.
        previous=pd.read_csv(HERE/('pilot_'+cell)/'effects.csv')
        current=fx[fx.cell.eq(cell)].reset_index(drop=True)
        previous=previous.reset_index(drop=True)
        pd.testing.assert_frame_equal(current,previous,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
        audit.append(dict(cell=cell,per_cell_exact_reproduction=True,episodes=len(e),anchors=len(a),factorial_free_roots=len(g)))
        print(json.dumps(audit[-1]),flush=True)
    pd.DataFrame(allplans).to_csv(OUT/'measured_continuation_precision.csv',index=False)
    pd.concat(scores,ignore_index=True).to_csv(OUT/'episode_linearized_scores.csv',index=False)
    pd.DataFrame(eq).to_csv(OUT/'score_reconstruction64.csv',index=False)
    pd.DataFrame(armowner).to_csv(OUT/'pilot_owner_IR.csv',index=False)
    pd.DataFrame(guarddiag).to_csv(OUT/'POSTHOC_guard_state.csv',index=False)
    pd.DataFrame(guardcausal).to_csv(OUT/'POSTHOC_guard_call_effects.csv',index=False)
    save('audit.json',dict(cells=audit,primary_count=len(primary),primary_simultaneous_nonzero=int(((primary.lo_primary>0)|(primary.hi_primary<0)).sum()),
                            primary_status=primary.status.value_counts().to_dict(),
                            nominations=int(pd.read_csv(frozen/'nominations.csv').chosen.sum()),
                            analysis_sha256=sha(__file__),frozen_analysis_sha256=sha(HERE/'analyze_pilot.py'),
                            projection='observed task-fixed repeated-init MS components; raw ICC retained; negative init variance clipped for conservative planning',
                            assumptions='same risk-set visits/linearization and covariance under new inits; no guarantee; zero variance not safety'))


if __name__=='__main__':main()
