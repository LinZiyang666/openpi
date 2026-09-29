"""Summaries plus explicitly post-pilot precision/exploratory extensions.

No changes to frozen estimation, splits, tests or controller selection.
"""
from pathlib import Path
from collections import defaultdict, Counter
import csv
import hashlib
import json
import math
import re
import numpy as np
from scipy import stats
from exp.offline_search.rounds.r06.ideation_Q2 import pilot_q2 as p
from exp.offline_search.rounds.r06.ideation_Q2.frontier import build_frontier as f
from exp.offline_search.rounds.r06.p3_profiling.pilot_stats import icc

HERE = Path(__file__).resolve().parent
Q2 = HERE.parent
OUT = HERE / 'derived'
OUT.mkdir(exist_ok=True)

def write(name, rows):
    p.write_csv(OUT / (name + '.csv'), rows)

def dump(name, value):
    p.dump(OUT / (name + '.json'), value)

def read(name):
    return list(csv.DictReader((HERE / 'preregistered' / (name + '.csv')).open()))

result = json.loads((HERE/'preregistered/estimates.json').read_text())
audit = json.loads((HERE/'preregistered/audit_q2.json').read_text())
lock = json.loads((Q2/'PREREG_LOCK.json').read_text())
checks = {n: hashlib.sha256((Q2/n).read_bytes()).hexdigest() == lock['sha256'][n]
          for n in ['PREREG.md', 'pilot_q2.py', 'library_quality.json']}
assert all(checks.values())
assert audit['selected_episodes'] == 4320 and audit['selected_arms'] == 216
assert len(result['completeness']) == 8 and all(x['complete'] for x in result['completeness'])
agreement = {}
for cell in sorted(x['cell'] for x in result['completeness']):
    old = json.loads((Q2/f'pilot_{cell}/estimates.json').read_text())
    agreement[cell] = all([x for x in result[k] if x['cell'] == cell] == v for k, v in old.items())
assert all(agreement.values())
dump('verification', dict(frozen_hash_matches=checks, per_cell_exact_agreement=agreement,
                         wrapper_change='CPU mask only: 18-21,62-65 -> 2-5,46-49'))

es = [r for r in result['estimates'] if r['task'] == 'ALL']
cs = [r for r in result['contrasts'] if r['task'] == 'ALL']
write('pilot_cell_curves', [r for r in es if r['split'] == 'all'])
write('pilot_cell_contrasts', cs)
write('pilot_marginal_returns', [r for r in result['marginal_slopes'] if r['task']=='ALL' and r['split']=='all'])
write('pilot_diminishing_returns', [r for r in result['diminishing_returns'] if r['task']=='ALL' and r['split']=='all'])
riskpairs = [r | {'point_dominates': r['SR_delta'] >= 0 and r['IR_delta'] <= 0}
             for r in cs if r['reference'].startswith('uniform')]
write('risk_vs_uniform', riskpairs)

quality, _ = p.quality_rows()
episodes = read('episode_costs')
precision = []
for cell in sorted({r['cell'] for r in episodes}):
    maps = defaultdict(dict)
    for r in episodes:
        if r['cell'] == cell:
            maps[r['cohort']][int(r['task']),int(r['init']),int(r['block'])] = np.array([float(r[k]) for k in ['Y','V','M','N','C5']])
    for rho in p.RHOS:
        for uniform in (False, True):
            rule = p.allocation(quality, cell, rho, uniform)
            name = ('uniform' if uniform else 'risk') + f'_rho{rho:.2f}'
            maps[name] = {u: sum(rule['weights'][u[0]][c]*maps[c][u] for c in p.DOSES) for u in maps['A']}
    candidates = [c for c in maps if c in p.CORE or c.startswith(('risk_', 'uniform_'))]
    for candidate in candidates:
        references = ['P10','B'] + ([candidate.replace('risk_', 'uniform_')] if candidate.startswith('risk_') else [])
        for reference in references:
            if candidate == reference: continue
            units = sorted(maps[candidate])
            diff = {u: float(maps[candidate][u][0]-maps[reference][u][0]) for u in units}
            groups = [np.array([diff[t,i,b] for b in range(3)]) for t in range(10) for i in range(2)]
            measured = icc(groups, [t for t in range(10) for _ in range(2)])
            calibration = icc(groups[::2], list(range(10)))
            assert calibration['icc'] is None
            msb, msw = measured['ms_between'], measured['ms_within']
            sigma2 = (msb+2*msw)/3
            estimate = measured['icc']
            row = dict(cell=cell,candidate=candidate,reference=reference,delta_SR=float(np.mean(list(diff.values()))),
                       paired_variance=sigma2,ICC=estimate,MS_between=msb,MS_within=msw,
                       between_df=10,within_df=40,calibration_only_ICC=None,
                       calibration_only_within_variance=float(np.mean([g.var(ddof=1) for g in groups[::2]])),
                       scope='post-pilot model-based precision sensitivity; ALL inits, not selection or retuning')
            if estimate is not None and sigma2 > 0:
                corr=max(0.,estimate)
                row['ICC_used']=corr
                for name, ks in [('pilot_all',range(2)), ('full_all',range(25)),
                                 ('full_validation',[i for i in range(25) if i%5]),
                                 ('continuation_only',[i for i in range(2,25)])]:
                    repeats = np.array([int(i<25)+int(i<15)+int(i<10) for i in ks]*10)
                    neff = len(repeats)**2 / np.sum((1+(repeats-1)*corr)/repeats)
                    var = sigma2/neff
                    se=math.sqrt(var)
                    df=len(repeats)-10
                    crit=stats.t.ppf(1-p.ALPHA,df)
                    row.update({name+'_episodes':int(sum(repeats)),name+'_init_clusters':len(repeats),
                                name+'_neff':neff,name+'_halfwidth95_pp':100*1.959964*se,
                                name+'_episodeweighted_neff':float(sum(repeats)/(1+corr*np.sum(repeats*(repeats-1))/sum(repeats))),
                                name+'_MDE80_pp':100*(1.959964+.841621)*se,
                                name+'_NI_power_at_zero_family':float(stats.norm.cdf(.02/se-crit)),
                                name+'_NI_power_at_observed_family':float(stats.norm.cdf((.02+row['delta_SR'])/se-crit))})
                row['required_effective_N_NI_zero_nominal']=math.ceil((stats.norm.ppf(.95)+stats.norm.ppf(.8))**2*sigma2/.02**2)
                row['required_effective_N_NI_zero_family']=math.ceil((stats.norm.ppf(1-p.ALPHA)+stats.norm.ppf(.8))**2*sigma2/.02**2)
                for corr_s in [0,1]:
                    reps=np.array([3]*8+[2]*4+[1]*8)*1.
                    neff=(len(reps)*10)**2/(np.sum((1+(reps-1)*corr_s)/reps)*10)
                    row[f'full_validation_halfwidth95_ICC{corr_s}_pp']=100*1.959964*math.sqrt(sigma2/neff)
            else:
                row['precision_unresolved']='zero empirical paired variation does not certify population equality'
            precision.append(row)
write('measured_precision', precision)
task_ranges=[]
for cell in sorted({r['cell'] for r in result['estimates']}):
    task_est={(r['task'],r['controller']):r for r in result['estimates'] if r['cell']==cell and r['task']!='ALL' and r['split']=='all'}
    gaps=[task_est[t,'P10']['SR']-task_est[t,'A']['SR'] for t in range(10)]
    calls=[task_est[t,'B']['calls'] for t in range(10)]
    task_ranges.append(dict(cell=cell,A_to_P10_gap_min=min(gaps),A_to_P10_gap_max=max(gaps),
        B_mean_calls_min=min(calls),B_mean_calls_max=max(calls),nonpositive_gap_tasks=sum(x<=0 for x in gaps),
        zero_B_calls_tasks=sum(x==0 for x in calls),episodes_per_task_per_controller=6,init_clusters_per_task=2))
write('per_task_ranges',task_ranges)

front = json.loads((HERE/'frontier/frontier_data.json').read_text())
outcomes = json.loads((HERE/'frontier/outcomes.json').read_text())
source = json.loads((HERE/'frontier/source_evidence.json').read_text())
points = {x['id']: x for x in front['points']}
ab = {(x['cell'], x['label']): x for x in front['pooled_AB']}
native_rows = []
ablation_rows = []
ab_contrasts = []
for cell,label in ab:
    if label!='B':continue
    aa,bb=ab[cell,'A'],ab[cell,'B']
    delta,lo,hi=f.paired_boot(bb['runs'],aa['runs'],outcomes)
    ab_contrasts.append(dict(cell=cell,A_SR=aa['SR'],B_SR=bb['SR'],A_IR=aa['owner_IR'],B_IR=bb['owner_IR'],
                             delta_SR=delta,lo=lo,hi=hi,init_clusters=500,n_per_controller=1500))
for x in front['points']:
    if x['run'] not in ['r06_frontier','r06_abl'] or not x['eligible']:continue
    refs={L:f.reference_ids(x['model'],x['suite'],L) for L in [10,5]}
    row = {k:x[k] for k in ['id','cell','SR','owner_IR','v','m','method_family','n']}
    row['kwargs']=source[x['id']]['kwargs']
    for L,ids in refs.items():
        row[f'L{L}_delta']=x['SR']-np.mean([points[r]['SR'] for r in ids])
        row[f'L{L}_NI_lower']=f.aggregate_lower([x['id']],ids,outcomes,.05)
        row[f'L{L}_NI_pass']=row[f'L{L}_NI_lower'] > -.02
        row[f'L{L}_pairs']=[dict(ref=r,wins=(wl:=f.paired_counts(x['id'],r,outcomes))[0],losses=wl[1],
                                 p=float(stats.binomtest(wl[0],sum(wl[:2]),.5).pvalue) if sum(wl[:2]) else 1.) for r in ids]
    if x['run']=='r06_frontier':native_rows.append(row)
    else:
        for label in ['A','B']:
            ids=ab[x['cell'],label]['runs']
            row[label+'_delta'],row[label+'_delta_lo'],row[label+'_delta_hi']=f.paired_boot([x['id']],ids,outcomes)
            row[label+'_pairs']=[dict(ref=r,wins=(wl:=f.paired_counts(x['id'],r,outcomes))[0],losses=wl[1],
                                     p=float(stats.binomtest(wl[0],sum(wl[:2]),.5).pvalue) if sum(wl[:2]) else 1.) for r in ids]
        ablation_rows.append(row)
write('new_frontier_native_tests',native_rows)
write('ablations_paired',ablation_rows)
write('AB_clustered_contrasts',ab_contrasts)
dump('new_frontier_native_tests',native_rows)
dump('ablations_paired',ablation_rows)

minima=[]
for gap in front['gaps']:
    row=dict(gap)
    ident=gap['minimum_NI']
    if ident in points:
        point=points[ident]
        ids=f.reference_ids(point['model'],point['suite'],gap['reference_L'])
        row['selected_SR']=point['SR']
        row['selected_delta_SR']=point['SR']-gap['reference_SR']
        row['plain_tests']=[dict(reference=r,wins=(wl:=f.paired_counts(ident,r,outcomes))[0],losses=wl[1],
            p=float(stats.binomtest(wl[0],sum(wl[:2]),.5).pvalue) if sum(wl[:2]) else 1.) for r in ids]
    else:
        row['selected_SR']=gap['reference_SR']
        row['selected_delta_SR']=0.
        row['plain_tests']='reference identity, no test'
    minima.append(row)
write('native_minimum_IR',minima)
dump('native_minimum_IR',minima)

summary=dict(audit=audit,recommendations=result['recommendations'],completeness=result['completeness'],
    risk_validation_point_dominance=[(r['cell'],r['candidate']) for r in riskpairs if r['split']=='validation' and r['point_dominates']],
    risk_validation_total=sum(r['split']=='validation' for r in riskpairs),
    dose_mix_zero_support_all_tasks=sum(not r['supported'] for r in result['dose_mix'] if r['split']=='all' and r['task']!='ALL'),
    dose_mix_total_task_doses=sum(r['split']=='all' and r['task']!='ALL' for r in result['dose_mix']),
    diminishing_cell_calls=Counter(r['conclusion'] for r in result['diminishing_returns'] if r['split']=='all' and r['task']=='ALL'),
    positive_gap_cells=sorted({r['cell'] for r in result['gap_closure'] if r['split']=='all' and r['task']=='ALL' and r['gap_resolved']}),
    frontier_points=len(front['points']),new_frontier_arms=len(native_rows),ablation_arms=len(ablation_rows),
    snapshot=front['snapshot_utc'])
dump('summary',summary)
# The old renderer contains historical prose counts and old-CPU commands.
# Refresh only that prose in this new snapshot; never edit the historical report.
report_path=HERE/'frontier/frontier.md'
report=report_path.read_text()
sources=Counter(r['cost_source'] for r in front['points'])
dump('frontier_cost_sources',sources)
report=re.sub(r'This frozen census contains .*?Four DUAL references are explicitly added from outside os_closed_loop because PAPER_AB uses them for L=5\.',
    'This snapshot contains '+str(sources.get('summary.cost_ledger',0))+' summary-ledger rows, '+
    str(sum(v for k,v in sources.items() if k.startswith('prior accepted')))+' verified legacy raw-audit rows, '+
    'four historical native-policy references, and two conflicting-cost rows. All exclusions and reasons are in `skipped.csv`; counts are not inherited from the previous census.',report)
report=report.replace('See `completion_plan.md` for placement-only pilot use.', 'See `../../FINAL.md` for the separate preregistered pilot result.')
report=report.replace('`../arms.json`','`../../arms.json`')
report=report.split('## Reproduce / provenance')[0]+'''## Reproduce / provenance

This is the new post-hoc snapshot generated by `../audit_frontier.py`, then summarized by `../derive_results.py`. Exact invocation, snapshot time and limitations are in `../../FINAL.md`. All Python commands used CPUs 2-5,46-49 with CUDA hidden and BLAS/OMP threads=1. No external figure was rewritten. `summary_paths.json`, `source_evidence.json` and `outcomes.json` preserve the inventory, hashes, configurations and accepted outcomes used here. The prior census and its external figure are historical artifacts.
'''
report_path.write_text(report)
print(json.dumps(p.clean(summary),indent=2))
