"""R7 E4 exploratory, CPU-only analysis. Does not fit any deployment artifact.

Accepted U-arm traces -> causal phase excursions and cross-fitted, small
whole-trajectory importance reweighting. Phase labels are library-progress
tertiles of the PRE-COIN top retrieval row, not eventual episode progress.
Run from repo root with the exact taskset/env prefix in PROPOSAL.md.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = Path('/home/weiland/trace_runs/os_closed_loop/r06_c_validation/runs')
SCRATCH = Path('/tmp/r7_E4_allocation')


def load_arm(root):
    cell = root.name.removeprefix('r6c_').rsplit('_', 1)[0]
    model, suite, scale = cell.split('_')
    summary = json.loads((root / 'summary.json').read_text())
    frontier = csv.DictReader((HERE.parents[2] / 'r06/frontier_final/frontier_points.csv').open())
    lib = Path(next(r['library_path'] for r in frontier if r['run']=='r06_c_validation' and r['arm']==root.name))
    progress = np.load(lib / 'progress.npy', mmap_mode='r')
    task_id = np.load(lib / 'task_id.npy', mmap_mode='r')
    # Outcome-independent diagnostic choice K=3; not a deployed segmenter.
    cuts = {int(t): np.quantile(progress[task_id == t], [1/3, 2/3]) for t in np.unique(task_id)}
    records = {}
    for line in (root / 'client/journal.jsonl').open():
        j = json.loads(line)
        if j.get('accepted') and j.get('status') in ('done', 'failed') and not j.get('error'):
            uid = j['task_uid']
            t, init = map(int, uid.rsplit(':', 2)[1:])
            records[uid] = dict(uid=uid, task=t, init=init, Y=int(j['success']),
                                attempt=j['attempt'], N=0, V=0, M=0, anchors=[])
    assert len(records) == 500
    seen = set()
    files = sorted(root.glob('server_*/decisions_*.jsonl'))
    source_hashes = {}
    for f in files:
        hasher = hashlib.sha256()
        with f.open('rb') as stream:
            for line in stream:
                hasher.update(line)
                d = json.loads(line)
                if d.get('ev') != 'dec':
                    continue
                e = records.get(d['uid'])
                if e is None or d['attempt'] != e['attempt']:
                    continue
                key = d['uid'], d['step']
                assert key not in seen, key
                seen.add(key)
                e['N'] += 1
                e['V'] += int(d['vision'])
                e['M'] += int(not d['hit'])
                if not d['vision']:
                    continue
                x = d['extras']
                p = float(x['os_q2_p_call'])
                z = int(not d['hit'])
                assert z == int(x['os_q2_coin'] < p)
                row = int(d['top1'])
                assert int(task_id[row]) == e['task']
                stage = int(np.searchsorted(cuts[e['task']], progress[row], side='right'))
                e['anchors'].append((int(d['step']), stage, p, z))
        source_hashes[str(f)] = hasher.hexdigest()
    rows = sorted(records.values(), key=lambda d: (d['task'], d['init']))
    c1 = .152 if model == 'pi05' else .148
    for e in rows:
        e['anchors'].sort()
        assert e['anchors'][0][0] == 0
        e['C'] = c1 * e['V'] + (1-c1) * e['M']
        assert len({a[2] for a in e['anchors']}) == 1
    totals = {k: sum(e[k] for e in rows) for k in ('N', 'V', 'M', 'Y')}
    ledger = summary['cost_ledger']
    assert totals['N'] == ledger['decisions']
    assert totals['V'] == ledger['vision_decisions']
    assert totals['M'] == ledger['misses']
    assert totals['Y'] == summary['success']
    info = dict(cell=cell, c1=c1, totals=totals, library=str(lib), cuts={t:c.tolist() for t,c in cuts.items()},
                sources=source_hashes, dose_episodes=dict(Counter(str(e['anchors'][0][2]) for e in rows)),
                SR=totals['Y']/500, IR=sum(e['C'] for e in rows)/totals['N'])
    return rows, info


def cross_baseline(rows, mask):
    return {t: np.mean([e['Y'] for i,e in enumerate(rows) if mask[i] and e['task']==t]) for t in range(10)}


def transfer(gradient, occupancy, bound):
    """Exact fractional knapsack for +/-bound shifts at fixed call count."""
    shift=np.full(3,-bound)
    remaining=bound*occupancy.sum()
    for s in np.argsort(-gradient/occupancy,kind='stable'):
        increase=min(2*bound,remaining/occupancy[s])
        shift[s]+=increase
        remaining-=increase*occupancy[s]
    assert abs(float(occupancy@shift))<1e-8
    return shift


def phase_excursions(rows):
    """First eligible encounter per phase: one intervention, original continuation.

    Reach/phase are fixed before its coin. An earlier coin may affect reach of
    later phases; this is not a phase-completion SR nor an all-phase treatment.
    """
    result = []
    for s in range(3):
        data = []
        for e in rows:
            first = next((a for a in e['anchors'] if a[1]==s and 0<a[2]<1), None)
            if first:
                data.append((e, first))
        v = []
        for e,a in data:
            other = [f['Y'] for f,_ in data if f['task']==e['task'] and f['init']%2 != e['init']%2]
            b = float(np.mean(other)) if other else 0.
            p,z = a[2:]
            v.append((z-p)/(p*(1-p))*(e['Y']-b))
        v=np.array(v)
        se=float(np.std(v,ddof=1)/math.sqrt(len(v)))
        result.append(dict(phase=s, n=len(v), calls=sum(a[3] for _,a in data),
                           effect=float(v.mean()), ci=[float(v.mean()-1.96*se),float(v.mean()+1.96*se)]))
    return result


def reallocation(rows, max_shift):
    """2-fold policy learning, exact path likelihood ratio for held-out policy.

    A three-variable fractional knapsack transfers <=max_shift call probability
    between phases, with fixed-occupancy expected additional calls exactly zero.
    Learning uses episode score gradients, not regression on realized call count.
    The modest max_shift is an exploratory sensitivity, not a deployment knob.
    """
    n=len(rows)
    weights=np.ones(n); residual=np.zeros(n); selected=[]
    for parity in (0,1):
        train=np.array([e['init']%2 != parity for e in rows]); test=~train
        base=cross_baseline(rows,train)
        count=np.zeros((n,3)); score=np.zeros((n,3))
        for i,e in enumerate(rows):
            for _,s,p,z in e['anchors']:
                if 0<p<1:
                    count[i,s]+=1
                    score[i,s]+=(z-p)/(p*(1-p))
        centered=np.array([e['Y']-base[e['task']] for e in rows])
        g=np.mean(score[train]*centered[train,None],axis=0)
        occupancy=np.mean(count[train],axis=0)
        bound=min(max_shift,min(min(a[2],1-a[2]) for i,e in enumerate(rows) if train[i] for a in e['anchors'] if 0<a[2]<1)/2)
        shift=transfer(g,occupancy,bound)
        assert abs(float(occupancy@shift))<1e-8
        for i,e in enumerate(rows):
            if not test[i]:continue
            logw=0.
            for _,s,p,z in e['anchors']:
                if p in (0.,1.):continue  # no support: preserve controller exactly
                q=p+shift[s]
                logw+=math.log(q/p) if z else math.log((1-q)/(1-p))
            weights[i]=math.exp(logw)
            residual[i]=(weights[i]-1)*(e['Y']-base[e['task']])
        selected.append(dict(test_parity=parity, gradient=g.tolist(), occupancy=occupancy.tolist(), shifts=shift.tolist(),
                             predicted_delta_SR=float(g@shift)))
    y=np.array([e['Y'] for e in rows]); c=np.array([e['C'] for e in rows]); d=np.array([e['N'] for e in rows])
    se=float(np.std(residual,ddof=1)/math.sqrt(n))
    # Paired cost difference ratio IF; valid for fixed learned folds only.
    ir=float(weights@c/(weights@d)); old_ir=float(c.sum()/d.sum())
    influence=weights*(c-ir*d)/np.mean(weights*d)-(c-old_ir*d)/np.mean(d)
    irse=float(np.std(influence,ddof=1)/math.sqrt(n))
    return dict(max_probability_shift=max_shift, folds=selected,
                delta_SR=float(residual.mean()), ci_conditional=[float(residual.mean()-1.96*se),float(residual.mean()+1.96*se)],
                SR_HT=float(weights@y/n), SR_self_normalized=float(weights@y/weights.sum()),
                IR=ir, delta_IR=ir-old_ir, delta_IR_ci_conditional=[ir-old_ir-1.96*irse,ir-old_ir+1.96*irse],
                mean_weight=float(weights.mean()), ESS=float(weights.sum()**2/(weights@weights)), max_weight=float(weights.max()))


def bootstrap_refit(rows, draws=2000):
    """Refit both selection folds inside task x parity bootstrap resamples.

    Screening uncertainty only: model selection is nonsmooth, phases are coarse,
    and these same historic benchmark outcomes may not calibrate deployment.
    """
    n=len(rows); counts=np.zeros((n,3)); calls=counts.copy()
    p=np.array([e['anchors'][0][2] for e in rows])
    for i,e in enumerate(rows):
        for _,s,prob,z in e['anchors']:
            if 0<prob<1:
                counts[i,s]+=1;calls[i,s]+=z
    p=np.where((p>0)&(p<1),p,.5)
    score=(calls-p[:,None]*counts)/(p*(1-p))[:,None]
    task=np.array([e['task'] for e in rows]); parity=np.array([e['init']%2 for e in rows])
    y=np.array([e['Y'] for e in rows]); c=np.array([e['C'] for e in rows]); d=np.array([e['N'] for e in rows])
    groups=[np.where((task==t)&(parity==f))[0] for t in range(10) for f in (0,1)]
    bound=min(.05,float(np.minimum(p,1-p).min()/2))
    rng=np.random.default_rng(20260930)
    values=[]
    for _ in range(draws):
        sample=np.concatenate([rng.choice(group,len(group),replace=True) for group in groups])
        weights=np.ones(n); residual=np.zeros(n)
        for f in (0,1):
            train=sample[parity[sample]!=f]
            test_pos=np.where(parity[sample]==f)[0]; test=sample[test_pos]
            b=np.array([y[train[task[train]==t]].mean() for t in range(10)])
            g=np.mean(score[train]*(y[train]-b[task[train]])[:,None],axis=0)
            shift=transfer(g,counts[train].mean(axis=0),bound)
            q=p[test,None]+shift[None,:]
            logw=(calls[test]*np.log(q/p[test,None])+(counts[test]-calls[test])*np.log((1-q)/(1-p[test,None]))).sum(axis=1)
            weights[test_pos]=np.exp(logw)
            residual[test_pos]=(weights[test_pos]-1)*(y[test]-b[task[test]])
        values.append((residual.mean(),weights@c[sample]/(weights@d[sample])-c[sample].sum()/d[sample].sum()))
    values=np.array(values)
    return dict(draws=draws, seed=20260930, max_probability_shift=.05,
                delta_SR_percentile95=np.quantile(values[:,0],[.025,.975]).tolist(),
                delta_IR_percentile95=np.quantile(values[:,1],[.025,.975]).tolist(),
                label='Exploratory task-by-fold bootstrap refits both allocation policies; not simultaneous or a noninferiority certificate.')


def main():
    SCRATCH.mkdir(exist_ok=True)
    out={}
    for root in sorted(RUNS.glob('r6c_*_U*')):
        cache=SCRATCH/(root.name+'.json')
        if cache.exists() and '--refresh' not in sys.argv:
            rows,info=json.loads(cache.read_text())
        else:
            rows,info=load_arm(root)
            cache.write_text(json.dumps([rows,info]))
        info['excursions']=phase_excursions(rows)
        info['reallocation']=[reallocation(rows,delta) for delta in (.025,.05,.10)]
        info['bootstrap_refit']=bootstrap_refit(rows)
        out[info['cell']]=info
        main_result=info['reallocation'][1]
        print(info['cell'], 'SR',round(info['SR'],4),'IR',round(info['IR'],4),'dSR',round(main_result['delta_SR'],4),
              'CI',np.round(main_result['ci_conditional'],4).tolist(),'dIR',round(main_result['delta_IR'],5),
              'ESS',round(main_result['ESS']), 'first effects',[(r['n'],round(r['effect'],3)) for r in info['excursions']],flush=True)
    (HERE/'allocation_results.json').write_text(json.dumps(out,indent=2)+'\n')
    audit=dict(episodes=500*len(out), decisions=sum(v['totals']['N'] for v in out.values()),
               vision_anchors=sum(v['totals']['V'] for v in out.values()),
               phase_intervals_containing_zero=sum(r['ci'][0]<=0<=r['ci'][1] for v in out.values() for r in v['excursions']),
               refit_intervals_containing_zero=sum(v['bootstrap_refit']['delta_SR_percentile95'][0]<=0<=v['bootstrap_refit']['delta_SR_percentile95'][1] for v in out.values()),
               same_best_destination=sum(len(set(int(np.argmax(np.array(f['gradient'])/np.array(f['occupancy']))) for f in v['reallocation'][1]['folds']))==1 for v in out.values()))
    (HERE/'audit_summary.json').write_text(json.dumps(audit,indent=2)+'\n')
    sample={str(n):1-.05**(1/n) for n in (2,3,10,50,149,299)}
    (HERE/'calibration_limits.json').write_text(json.dumps(dict(one_sided_95_zero_adverse_event_upper_bound=sample,
        n_for_2pp_zero_adverse_events=math.ceil(math.log(.05)/math.log(.98)),
        independent_n_per_arm_90pct_SR_2pp_halfwidth_two_sided=math.ceil(1.96**2*.18/.02**2)),indent=2)+'\n')


if __name__=='__main__':main()
