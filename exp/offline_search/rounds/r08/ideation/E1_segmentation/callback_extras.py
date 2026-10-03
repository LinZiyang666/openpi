"""Compact report tables from completed discovery-only E1 products."""
from collections import Counter
import json
import numpy as np
import pandas as pd
from callback_analysis import OUT,ROOT
from callback_summary import cluster_ci

e=pd.read_parquet(OUT/'all_episodes.parquet')
r=pd.read_csv(OUT/'risk_summary.csv')
b=pd.read_csv(OUT/'boundary_summary.csv')
assert e.init.max()<30 and e.arm.nunique()==36
lines=[]
def show(name,data):
    lines.append('\n'+name+'\n'+(data.to_string(index=False) if isinstance(data,pd.DataFrame) else str(data)))

show('COVERAGE',json.loads((OUT/'summary.json').read_text()))
show('SR',e.groupby(['arm','model','suite','variant'],dropna=False).agg(n=('success','size'),success=('success','sum'),onset=('onset','count')).reset_index())
rr=r[(r.definition=='known_onset_or_success') & (r.variant=='A')]
keep=['event_mass','gripper_r7','elapsed_fixed','stop_q25_d5','waypoint_x1.0','change_x1.0']
show('A RISK',rr[rr.candidate.isin(keep)][['arm','candidate','windows','positive_windows','flagged','true_positive','sensitivity','precision','auprc','onset_episodes','hit_episodes','actionable_hit_episodes','median_lead','delta_ci_lo','delta_ci_hi']])
ra=r[(r.definition=='known_onset_or_success') & r.candidate.isin(keep)]
show('RISK BY VARIANT equal-cell mean',ra.groupby(['variant','candidate']).agg(cells=('arm','nunique'),sensitivity=('sensitivity','mean'),catalog_delta=('delta_vs_catalog','mean'),positive_cells=('delta_vs_catalog',lambda x:(x>0).sum()),positive_CI_cells=('delta_ci_lo',lambda x:(x>0).sum())).reset_index())
ba=b[(b.variant=='A') & b.tolerance.eq(5)]
show('A BOUNDARY',ba.groupby(['candidate','baseline']).agg(cells=('arm','nunique'),f1=('micro_f1','mean'),predicted=('predicted','sum'),truth=('truth','sum'),matched=('matched','sum')).reset_index())
show('BOUNDARY BY VARIANT',b[b.tolerance.eq(5)&b.candidate.isin(keep)].groupby(['variant','candidate','baseline']).agg(cells=('arm','nunique'),f1=('micro_f1','mean')).reset_index())
show('RESET HASHES',e.groupby(['suite','task_id','init']).reset_hash.nunique().value_counts().to_dict())
show('ONSET LABELS A',e[e.variant.eq('A')].label.value_counts().to_dict())
show('STAGE COUNTS',[(v,dict(sum((Counter(json.loads(x)) for x in g.stage_counts),Counter()))) for v,g in e.groupby('variant')])
show('DUPLICATE RANKINGS',pd.read_csv(OUT/'identical_risk_rankings.csv').groupby(['a','b']).unequal_flags.agg(['sum','size']).reset_index())
age=pd.read_csv(OUT/'age_summary.csv')
age['variant']=age.arm.str.rsplit('_',n=1).str[-1]
show('AGE BY VARIANT',age.groupby(['variant','blind_age'])[['windows','positive_windows','unknown_onset_windows']].sum().reset_index())
show('SEMANTIC',pd.read_csv(OUT/'semantic_audit.csv').groupby('success').agg(episodes=('success','size'),with_losses=('predicate_losses',lambda x:(x>0).sum()),losses=('predicate_losses','sum'),with_long_losses=('predicate_losses_ge5',lambda x:(x>0).sum()),long_losses=('predicate_losses_ge5','sum'),release_controls=('release_controls','sum'),closed_release=('release_while_closed','sum'),completed_closed_release=('completed_closed_release','sum')).reset_index())

matched=[];coverage=[];rec=[]
for arm in sorted(e.arm.unique()):
    ds=pd.read_parquet(OUT/arm/'decisions.parquet')
    coverage.append(dict(arm=arm,decisions=len(ds),catalog_missing=int(ds.event_mass.isna().sum()),unknown_mass=int((ds.event_known_mass<.999999).sum()),cost_missing=int(ds.owner_cost.isna().sum()),owner_cost=ds.owner_cost.sum(),applied=ds.applied.sum()))
    if not arm.endswith('_A'):continue
    rec.append(pd.read_parquet(OUT/arm/'reconstruction.parquet'))
    ds=ds[ds.src.isin(['cache','cache_tail','follow']) & ds.at_risk & ds.observed & ds.known_negative & ds.event_mass.notna()].reset_index(drop=True)
    ds['age_bin']=ds.start//50
    tie=pd.util.hash_pandas_object(ds[['suite','task_id','init','start']],index=False).to_numpy()
    flags={}
    for c in ['event_mass','waypoint_x1.0','change_x1.0']:
        flag=np.zeros(len(ds),bool)
        for _,ix in ds.groupby(['task_id','age_bin']).indices.items():
            order=np.lexsort((tie[ix],ds[c].to_numpy()[ix]));k=int(np.ceil(.2*len(ix)))
            flag[np.asarray(ix)[order[-k:]]]=True
        flags[c]=flag
    for c,flag in flags.items():
        y=ds.bad.to_numpy(bool);ci=cluster_ci(ds,flag,flags['event_mass'])
        matched.append(dict(arm=arm,candidate=c,windows=len(ds),positive_windows=int(y.sum()),flagged=int(flag.sum()),tp=int((y&flag).sum()),delta=(int((y&flag).sum())-int((y&flags['event_mass']).sum()))/y.sum(),ci_lo=ci[0],ci_hi=ci[1]))
pd.DataFrame(matched).to_csv(OUT/'age_matched_risk.csv',index=False)
pd.DataFrame(coverage).to_csv(OUT/'decision_coverage.csv',index=False)
show('AGE MATCHED RISK',pd.DataFrame(matched))
show('DECISION COVERAGE',pd.DataFrame(coverage))
rec=pd.concat(rec,ignore_index=True)
show('A WAYPOINT RECONSTRUCTION',rec[rec.candidate.eq('waypoint_x1.0')].groupby(['resolution','suite']).agg(episodes=('episode_key','size'),median_knots=('knots','median'),p95_position_max=('translation_max',lambda x:x.quantile(.95)),p95_rotation_max=('SO3_rotation_max',lambda x:x.quantile(.95))).reset_index())
show('AUG_DONE at report time',[a for a in sorted(e.arm.unique()) if (ROOT/'state'/f'{a}.AUG_DONE').exists()])
cost=pd.DataFrame(coverage)
cost['variant']=cost.arm.str.rsplit('_',n=1).str[-1]
cost=cost.groupby('variant')[['owner_cost','applied']].sum()
cost['IR']=5*cost.owner_cost/cost.applied
show('PACKAGE COST',cost.reset_index())
paired=e[e.variant!='P10'].pivot(index=['suite','task_id','init','model','library_size'],columns='variant',values='success').astype(float)
paired_ci={}
for a,z in [('FL','A'),('CT','CU')]:
    d=(paired[a]-paired[z]).groupby(['suite','task_id','init']).sum()
    rng=np.random.default_rng(801);num=np.zeros(10000)
    for _,g in d.groupby(['suite','task_id']):
        v=g.to_numpy();num+=v[rng.integers(0,30,(10000,30))].sum(axis=1)
    paired_ci[a+'-'+z]=dict(delta=float(d.sum()/2400),ci=np.quantile(num/2400,[.025,.975]).tolist(),net_successes=int(d.sum()))
show('PACKAGE PAIRED SR',paired_ci)
(OUT/'package_summary.json').write_text(json.dumps(dict(cost=cost.reset_index().to_dict('records'),paired_SR=paired_ci),indent=2)+'\n')
(OUT/'report_numbers.txt').write_text('\n'.join(lines)+'\n')
print('WROTE',OUT/'report_numbers.txt')
