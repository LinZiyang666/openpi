"""The existing descent/test rules, with untouched source pure references."""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from .common import HERE, EVAL, LIBERO, MT, read, dump, continued_lanes
from exp.offline_search.rounds.r17.knob import common as c, descent, test

LABELS={'pi05':'π0.5','groot':'GR00T','dp':'DP','fastwam':'Fast-WAM','cosmos':'Cosmos'}
SHORT={'dp':'dp','fastwam':'fw','cosmos':'cos'}
TEST_SOURCE={'dp':('knobtestdp_red','knobtestdp'),'fastwam':('knobtestfw_red','knobtestfw3'),
             'cosmos':('knobtestcos_red','knobtestcos2')}


def mt_controls(model,mode):
    source=HERE.parent/'knob/mt50'/f'{model}_{mode}'/'descent.json'
    defaults=c.MW_REDUCED_CONTROL_NAMES if mode=='reduced' else c.MW_CONTROL_NAMES
    recorded=read(source).get('controls',defaults) if source.exists() else defaults
    return [f'{LABELS[model]}={recorded[LABELS[model]][-1]}']


def pure_b(model,mode):
    prefix='knobdev'+SHORT[model]+('red' if mode=='reduced' else '')
    paths=[p for p in sorted(EVAL.glob(prefix+'*')) if (p/'EPISODE_PLAN.json').exists() and
           ('red' in p.name)==(mode=='reduced')]
    cells,_=c.load_knob(paths,'B'); refs={}
    for (label,suite,size),arms in cells.items():
        if size==500 and 'pure' in arms:c.merge(refs.setdefault((label,suite),{}),arms['pure'])
    return refs


def controls(model,mode):
    return ['--control','Fast-WAM=hs7fwk1_4090c'] if model=='fastwam' and mode=='reduced' else []


def off_rules(model,mode):
    path=EVAL/f'ffk1015_{model}_libero_{mode}_off'
    if mode!='reduced' or not path.exists():return {}
    cells,_=c.load_knob(continued_lanes([path]),'A')
    ref=test.load_controls(c.CONTROL_ROOT,{LABELS[model]},mode,
        ['Fast-WAM=hs7fwk1_4090c'] if model=='fastwam' else [])
    result={}
    for cell,arms in cells.items():
        label,suite,size=cell;off=arms['koff'];pure=ref[label,suite]
        if len(off)!=500 or set(off)!=set(pure):raise ValueError('Complete knob-off A before applying the original no-knob rule')
        result[cell]=sum(r['success'] for r in off.values())>=sum(r['success'] for r in pure.values())
    return result


def dev(model,benchmark,mode):
    out=HERE/'results'/f'{model}_{benchmark}_{mode}'/'dev'
    run=EVAL/(f'ffk1015_{model}_{benchmark}_{mode}_'+('descent' if benchmark=='mt50' else 'dev'))
    runs=([Path(j['run']) for j in read(HERE/'CAMPAIGNS.json')['jobs'] if j['model']==model and j['benchmark']==benchmark and j['mode']==mode and j['stage']=='descent'] if benchmark=='mt50' else [run])
    runs=continued_lanes(runs)
    args=[v for lane in runs for v in ('--lane',str(lane))]+['--out',str(out),'--miss-steps',mode]
    if benchmark=='mt50':args += [v for control in mt_controls(model,mode) for v in ('--control',control)]
    if benchmark=='libero' and mode=='reduced':
        # Supply only the source's pure-B arm to the standard analyzer, never
        # merge any unfiltered cache outcome into filtered descent.
        with patch.object(c,'pure_references',side_effect=lambda paths:pure_b(model,mode)):
            result=descent.main(args+['--reference-lane',str(EVAL/LIBERO[model][0])])
    else:result=descent.main(args)
    if benchmark=='libero':
        levels=read(out/'levels.json');rules=off_rules(model,mode)
        for (label,suite,size),no_knob in rules.items():
            if no_knob:levels[label][suite][str(size)]={t:'koff' for t in levels[label][suite][str(size)]}
        dump(out/'levels.json',levels)
        dump(out/'NO_KNOB.json',dict(cells=[dict(model=k[0],suite=k[1],size=k[2],no_knob=v) for k,v in rules.items()]))
    return result


def test_a(model,mode,generation,override=()):
    prefix=f'ffk1015_{model}_libero_{mode}_'
    lanes=[p for p in sorted(EVAL.glob(prefix+'test*')) if (p/'EPISODE_PLAN.json').exists()]
    off=EVAL/(prefix+'off')
    if off.exists():lanes.append(off)
    lanes=continued_lanes(lanes)
    out=HERE/'results'/f'{model}_libero_{mode}'/f'test{generation:02d}'
    args=[v for lane in lanes for v in ('--lane',str(lane))]
    extra=controls(model,mode)+[v for item in override for v in ('--control',item)]
    result=test.main(args+['--out',str(out),'--controls',mode]+extra)
    rules=off_rules(model,mode)
    raises=read(out/'raise.json')
    for (label,suite,size),no_knob in rules.items():
        if no_knob:raises.get(label,{}).get(suite,{}).pop(str(size),None)
    for row in result['cells']:
        row['no_knob']=rules.get((row['model'],row['suite'],row['size']),False)
        row['all_tasks_within_tolerance']=all(t['within_tolerance'] for t in row['tasks'].values())
    dump(out/'raise.json',raises)
    dump(out/'test.json',result)
    return result


def next_campaign(model,mode,generation):
    from .build import build
    key=f'{model}_libero_{mode}'
    table=HERE/'results'/key/('dev/levels.json' if generation==0 else f'test{generation-1:02d}/raise.json')
    values=read(table).get(LABELS[model],{})
    if not any(tasks for sizes in values.values() for tasks in sizes.values()):
        print('NO_TASKS_TO_RAISE');return
    # Cells satisfying the whole-cell off rule keep their already measured A.
    values={s:dict(v) for s,v in values.items()}
    for (label,suite,size),no_knob in off_rules(model,mode).items():
        if no_knob:values.get(suite,{}).pop(str(size),None)
    flat={LABELS[model]:values}
    if not any(tasks for sizes in values.values() for tasks in sizes.values()):
        print('ALL_CELLS_USE_MEASURED_KNOB_OFF');return
    # Select the source's corresponding official-A runtime/topology. Later
    # source raises reuse their final serving release after the last old round.
    base=TEST_SOURCE[model][mode=='full']
    source=EVAL/base
    suffix='_red' if mode=='reduced' else ''
    candidate=EVAL/f'knobtest{SHORT[model]}{suffix}_r{generation}'
    if generation and candidate.exists():source=candidate
    row=build(source,f'ffk1015_{key}_test{generation:02d}',table=flat,
              ladder_source=EVAL/LIBERO[model][mode=='full'])
    print(row)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['descent','test','build-test'])
    p.add_argument('--model',choices=LABELS,required=True);p.add_argument('--benchmark',choices=['mt50','libero'],default='libero')
    p.add_argument('--mode',choices=['full','reduced'],required=True);p.add_argument('--generation',type=int,default=0)
    p.add_argument('--control',action='append',default=[])
    a=p.parse_args()
    if a.action=='descent':dev(a.model,a.benchmark,a.mode)
    elif a.action=='test':test_a(a.model,a.mode,a.generation,a.control)
    else:next_campaign(a.model,a.mode,a.generation)


if __name__=='__main__':main()
