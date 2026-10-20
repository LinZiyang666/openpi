"""Fresh long-segment lanes only; no withdrawn SR/levels enter selection."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch
from functools import lru_cache

from .audit import HERE,EVAL
from .io_budget import dump
from exp.offline_search.rounds.r17.filter_failed_knob.common import read,cells as library_cells,continued_lanes
from exp.offline_search.rounds.r17.filter_failed_knob import formal_a_analysis as a
from exp.offline_search.rounds.r17.knob import common as c,descent

LABELS={'dp':'DP','groot':'GR00T','fastwam':'Fast-WAM','cosmos':'Cosmos'}

@lru_cache(maxsize=None)
def prices(label,suite,gpu):
    from exp.offline_search.rounds.r17.knob import ir_formal as ir
    model={v:k for k,v in LABELS.items()}[label]
    benchmark='mt50' if suite=='mt50' else 'libero'
    if model=='cosmos':
        spec=read(HERE.parent/'filter_failed_knob/startup_checks/fa_co_r_u_l.json')['actual_spec']
    else:
        job=next(j for j in read(HERE/'CAMPAIGNS.json')['jobs'] if
                 (j['model'],j['benchmark'],j['gpu'],j['stage'])==(model,benchmark,gpu,'long'))
        proof=read(HERE/'cpu'/f'{job["name"]}.json')
        spec=read(Path(proof['native_strict_loader_proof']))['actual_spec']
    keys=('model_id','checkpoint_identity','code_digest','norm_identity','layout','feature_spec','python_island')
    suffix='_mw_mt50.json' if suite=='mt50' else '_libero_'+suite+'.json'
    paths=[Path('results_mt50' if suite=='mt50' else 'results')/(model+suffix)]
    if model=='fastwam' and suite!='mt50':paths.append(Path('results_cached_text')/(model+suffix))
    for root in (HERE/'profiles'/('h100' if gpu=='H100' else '4090'),ir.PROFILE):
        if all((root/p).exists() and read(root/p).get('valid') is True and
               all(read(root/p).get('model_spec',{}).get(k)==spec.get(k) for k in keys) and
               ('H100' in read(root/p).get('gpu_name','') if gpu=='H100' else '4090' in read(root/p).get('gpu_name','')) for p in paths):
            with patch.object(ir,'PROFILE',root):return ir.coefficients(label,suite)
    return None,None

def decision_summary(data,model):
    st=c.stats(data,model)
    heads=st['policy']+st['library']
    value=st['policy']/heads if st['n'] and st['cost_n']==st['n'] and heads else None
    st.update(level_ir=value,ir_proxy=value,decision_heads=heads,
              level_ir_definition='model-call heads / segment heads; tails excluded')
    return st


def stats(data,model,mode,suite,gpu):
    st=decision_summary(data,LABELS[model]);system,pure=prices(LABELS[model],suite,gpu)
    num=den=0
    for e in data.values():
        if 'cost' not in e:break
        cost=e['cost'];den+=cost.get('controls',0)
        if system is not None:
            price=pure if e.get('pure') else system
            num+=price['vision']*(cost.get('policy',0)+cost.get('library',0))+price['reduced_miss' if mode=='reduced' else 'full_miss']*cost.get('policy',0)
    else:
        st['ir']=num*c.C0[LABELS[model]]/den if system is not None and den else None
        st['latency_ir']=st['ir']
        return st
    st['ir']=st['latency_ir']=None;return st

_LANES={}

def lane_data(root):
    if str(root) in _LANES:return _LANES[str(root)]
    data,_=c.load_knob(continued_lanes([Path(root)]),'A');return data

def prefetch(roots):
    """Parse each run's lanes in its own process (single-process JSON parsing took >6 min for LIBERO)."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    roots=sorted({str(r) for r in roots if (Path(r)/'RECEIPT.json').exists()}-set(_LANES))
    if roots:
        with ProcessPoolExecutor(max_workers=max(1,min(len(roots),(os.cpu_count() or 2)//2))) as pool:
            _LANES.update(zip(roots,pool.map(lane_data,roots)))

def reference(model,benchmark,mode,gpu,*,long=False):
    registry=read(HERE/'REFERENCE_SELECTION.json')['references']
    entry=registry[f'{model}:{benchmark}:{mode}:{gpu}'+(':long' if long else '')]
    root=Path(entry['run'])
    if not (root/'RECEIPT.json').exists():return {}
    if read(root/'campaign.json').get('cache_study'):
        result={}
        for (label,suite,size),arms in lane_data(root).items():
            result.setdefault(suite,{}).update(arms.get('pure',{}))
        return result
    from exp.offline_search.rounds.r17.knob.test import load_controls
    result=load_controls(EVAL,{LABELS[model]},mode,[LABELS[model]+'='+root.name],suites=('mt50',) if benchmark=='mt50' else c.cache.SUITES)
    return {s:v for (m,s),v in result.items()}

def paired(left,right):
    keys=sorted(left.keys()&right.keys())
    diffs={k:int(right[k]['success'])-int(left[k]['success']) for k in keys}
    return dict(n=len(keys),difference=sum(diffs.values())/len(keys) if keys else None,
        by_task_init=[dict(task=k[1],init=k[2],difference=d) for k,d in diffs.items()],
        by_init={str(i):sum(d for k,d in diffs.items() if k[2]==i)/sum(k[2]==i for k in diffs)
                 for i in range(50) if any(k[2]==i for k in diffs)})

def whole_segment_result(result,data,target,long_pure,model,mode,suite,gpu):
    """A pure fallback in the complete system must also execute C10.

    Preserve the existing target/selection rule, and expose its actual
    tolerance result. Never substitute native C5 outcomes into a C10 system.
    """
    chosen=dict(data);tasks={k:dict(v) for k,v in result.get('tasks',{}).items()}
    for task,decision in tasks.items():
        if decision.get('selected')=='pure':
            chosen={k:v for k,v in chosen.items() if k[1]!=task}
            chosen.update({k:v for k,v in long_pure.items() if k[1]==task})
            decision['executed_fallback']='pure_long'
        ref={k:v for k,v in target.items() if k[1]==task}
        actual={k:v for k,v in chosen.items() if k[1]==task}
        decision['executed_tolerance_met']=(sum(v['success'] for v in ref.values())-
            sum(v['success'] for v in actual.values())<=2) if len(ref)==50 and actual.keys()==ref.keys() else None
    st=stats(chosen,model,mode,suite,gpu)
    out=dict(st,tasks=tasks,no_knob=result.get('no_knob'),final_complete=len(chosen)==(2500 if suite=='mt50' else 500),
        native_length_fallback_diagnostic=result,
        fallback='pure_long; no native-length episodes substituted',
        tolerance_all_tasks=(all(v['executed_tolerance_met'] for v in tasks.values())
            if tasks and all(v['executed_tolerance_met'] is not None for v in tasks.values()) else None))
    return out,chosen

def unaffected_libero():
    """Reuse the accepted formal-A analysis without rewriting its old output.

    Legacy VLA keeps its original per-decision cost ledger. Cosmos uses only
    a valid matching H100 timing receipt, or reports formal IR as missing.
    """
    from exp.offline_search.rounds.r17.filter_failed_knob import formal_a as f
    from exp.offline_search.rounds.r17.filter_failed_knob import formal_a_analysis as old
    jobs=[j for j in old.active_jobs() if j['model'] in ('cosmos','pi05','groot')]
    output=HERE/'unaffected_libero';output.mkdir(exist_ok=True)
    overrides=f.ROOT/'PURE_A_OVERRIDES.json'
    if overrides.exists():dump(output/'PURE_A_OVERRIDES.json',read(overrides))
    native_stats=old.stats
    def measured(data,model,mode,suite):
        return stats(data,model,mode,suite,'H100') if model=='cosmos' else native_stats(data,model,mode,suite)
    with patch.object(f,'ROOT',output),patch.object(f,'MODELS',('cosmos','pi05','groot')),patch.object(old,'active_jobs',return_value=jobs),patch.object(old,'stats',measured):
        old.analyze()
    rows=read(output/'RESULTS.json')['rows']
    for r in rows:
        r.update(benchmark='libero',mode=r['step_mode'],gpu='H100' if r['model']=='cosmos' or r['model']=='pi05' and r['step_mode']=='full' else 'RTX 4090',
            clock_source='Unchanged native Cosmos/R15/R16 pipeline; no long-segment defect',
            ir_source='Matching H100 profile or missing' if r['model']=='cosmos' else 'Unchanged native per-decision cost ledger')
    return rows

def _main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--benchmark',choices=('mt50','libero','all'),default='all')
    p.add_argument('--include-unaffected',action='store_true',help='Also reuse Cosmos and legacy VLA formal-A outcomes after those queues finish')
    args=p.parse_args();jobs=read(HERE/'CAMPAIGNS.json')['jobs']
    if (HERE/'FORMAL_QUEUE.json').exists():
        for u in read(HERE/'FORMAL_QUEUE.json')['units']:
            for j in u['jobs']:
                if j['name'] not in {v['name'] for v in jobs}:jobs.append(j)
    all_data=defaultdict(dict);gpu_for={};rows=[];libs=library_cells();active_keys=set()
    recalibration=read(HERE/'RECALIBRATION.json') if (HERE/'RECALIBRATION.json').exists() else {}
    calibration_levels=recalibration.get('level_inventory',[])
    selected=[j for j in jobs if j['stage'] in ('descent','off','levels') and args.benchmark in ('all',j['benchmark'])]
    references=read(HERE/'REFERENCE_SELECTION.json')['references']
    prefetch([j['run'] for j in selected]+[e['run'] for k,e in references.items()
              if (args.benchmark=='all' or k.split(':')[1]==args.benchmark) and (Path(e['run'])/'campaign.json').exists()
              and read(Path(e['run'])/'campaign.json').get('cache_study')])
    for j in selected:
        key=(j['model'],j['benchmark'],j['mode'],j['arm']);gpu_for[key[:3]]=j['gpu']
        for g in read(Path(j['run'])/'assets/CACHE_STUDY.json')['groups']:
            all_data[key].setdefault((g['suite'],g['library_size']),{})
        if not (Path(j['run'])/'RECEIPT.json').exists():continue
        active_keys.add(key[:3])
        for (_,suite,size),arms in lane_data(j['run']).items():
            for level,data in arms.items():c.merge(all_data[key].setdefault((suite,size),{}).setdefault(level,{}),data)
    for model,benchmark,mode in sorted(gpu_for):
        gpu=gpu_for[model,benchmark,mode]
        refs=reference(model,benchmark,mode,gpu) if (model,benchmark,mode) in active_keys else {}
        long_refs=reference(model,benchmark,mode,gpu,long=True) if (model,benchmark,mode) in active_keys else {}
        selections={}
        for arm in ('original','filtered'):
            data=all_data[model,benchmark,mode,arm]
            if benchmark=='mt50':
                native=descent.analyze_mt50({(LABELS[model],s,n):v for (s,n),v in data.items()},
                    {(LABELS[model],s):v for s,v in refs.items()},HERE/'results'/f'{model}_{mode}_{arm}',mode,
                    control_overrides=[LABELS[model]+'='+Path(read(HERE/'REFERENCE_SELECTION.json')['references'][f'{model}:{benchmark}:{mode}:{gpu}']['run']).name],
                    prices=lambda label,suite:prices(label,suite,gpu),summary=decision_summary)
                for r in native['cells']:
                    chosen={}
                    for task,d in r['tasks'].items():
                        level=d['selected'];pool=refs.get(r['suite'],{}) if level=='pure' else data[r['suite'],r['size']].get(level,{})
                        chosen.update({k:v for k,v in pool.items() if k[1]==task})
                    selections[arm,r['suite'],r['size']]=(r,chosen)
            else:
                for (suite,size),arms in data.items():
                    d=a.choose(arms,refs.get(suite,{}),('k50','k40','k30','k20','k10','koff'))
                    selections[arm,suite,size]=(dict(stats(d['data'],model,mode,suite,gpu),
                        tasks=d['tasks'],no_knob=d['no_knob'],complete=d['complete']),d['data'])
        for suite,size in sorted({(s,n) for arm,s,n in selections}):
            converted={arm:whole_segment_result(*selections.get((arm,suite,size),({},{})),
                refs.get(suite,{}),long_refs.get(suite,{}),model,mode,suite,gpu) for arm in ('original','filtered')}
            x=converted['original'];y=converted['filtered']
            for arm,converted_arm in (('original',x),('filtered',y)):
                measured=all_data[model,benchmark,mode,arm].get((suite,size),{})
                predicted={r['level']:{k:r.get(k) for k in ('old_gap','new_gap','nominal_ir','predicted_ir','absolute_error','passed')}
                    for r in calibration_levels if (r['model'],r['benchmark'],r['mode'],r['arm'],r['suite'],r['size'])==
                    (model,benchmark,mode,arm,suite,size)}
                converted_arm[0]['measured_levels']={level:dict(stats(measured.get(level,{}),model,mode,suite,gpu),
                    calibration=predicted.get(level),unused_off_sufficient=bool(converted_arm[0].get('no_knob')) and level!='koff')
                    for level in ('k50','k40','k30','k20','k10','koff')}
                for task,selection in converted_arm[0]['tasks'].items():
                    for level,entry in selection.get('levels',{}).items():
                        pool=refs.get(suite,{}) if level=='pure' else measured.get(level,{})
                        metric=stats({k:v for k,v in pool.items() if k[1]==task},model,mode,suite,gpu)
                        entry.update(level_ir=metric['level_ir'],latency_ir=metric['latency_ir'],decision_heads=metric['decision_heads'])
            info=libs[model,suite,size]
            rows.append(dict(model=model,benchmark=benchmark,mode=mode,suite=suite,size=size,gpu=gpu,
                original=x[0],filtered=y[0],paired_filtered_minus_original=paired(x[1],y[1]),
                filtered_episodes_kept=info['kept'],collected=info['collected'],
                empty_tasks=info['empty_tasks'],expected_outcomes_per_arm=2500 if suite=='mt50' else 500,
                native_pure_target=stats(refs.get(suite,{}),model,mode,suite,gpu),
                pure_long=stats(long_refs.get(suite,{}),model,mode,suite,gpu),
                fallback_note='Selection uses the same-step native-length pure target. Any selected pure fallback is executed from fresh pure+long outcomes. Missing control outcomes stay missing; actual tolerance is reported explicitly.',
                ir_note='Only accept formal IR when its latency receipt GPU equals this row GPU.'))
    if args.include_unaffected and args.benchmark in ('all','libero'):rows+=unaffected_libero()
    dump(HERE/'RESULTS.json',dict(rows=rows,names=read(HERE/'NAMES.json'),
         dev_outcomes_used=False,withdrawn_outcomes_used=False,clock_unit='segment_decision',
         level_ir_definition='M/N segment heads; tails count in neither',
         latency_ir_definition='Separate measured stage-cost metric, unchanged native-C0/control normalization'))

def main():
    from .io_budget import filesystem_budget
    with filesystem_budget():_main()

if __name__=='__main__':main()
