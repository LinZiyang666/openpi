"""One JSON row per cell, with actual IR and paired task/init differences."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .common import HERE, EVAL, MT, OFF, cells, read, dump, continued_lanes
from .workflow import LABELS, SHORT, mt_controls, controls
from exp.offline_search.rounds.r17.filter_failed.analyze import means
from exp.offline_search.rounds.r17.knob import common as c
from exp.offline_search.rounds.r17.knob.ir_formal import coefficients, ir
from exp.offline_search.rounds.r17.knob.test import load_controls
from exp.r17_integration.ops.cache_resume import segment_resources

KNOB=HERE.parent/'knob'
FINAL={('dp','full'):'dp_r4',('dp','reduced'):'dp_red_r7',('fastwam','full'):'fw_r2',
       ('fastwam','reduced'):'fw_red_r2',('cosmos','full'):'cos_r4',('cosmos','reduced'):'cos_red_r5'}


def comparison(filtered,baseline,mt50):
    keys=sorted(filtered.keys() & baseline.keys())
    values=[(k[1],int(filtered[k]['success'])-int(baseline[k]['success'])) for k in keys]
    by_init={}
    for i in range(50):
        selected=[v for k,v in zip(keys,values) if k[2]==i]
        if selected:by_init[str(i)]=means(selected,mt50)
    return dict(paired_n=len(keys),difference=means(values,mt50),by_init=by_init,
                gains=sum(filtered[k]['success'] and not baseline[k]['success'] for k in keys),
                losses=sum(baseline[k]['success'] and not filtered[k]['success'] for k in keys),
                pairs=[dict(task=k[1],init=k[2],filtered=filtered[k]['success'],unfiltered=baseline[k]['success'],
                            difference=int(filtered[k]['success'])-int(baseline[k]['success'])) for k in keys])


def stats(data,model,suite,mode):
    result=c.stats(data,LABELS[model])
    result['ir']=None
    if result['n'] and result['cost_n']==result['n'] and result['controls']:
        coef,_=coefficients(LABELS[model],suite)
        result['ir']=ir(result,coef,'reduced_miss' if mode=='reduced' else 'full_miss',c.C0[LABELS[model]])
    return result


def select(arms,planned,levels,reference):
    data={};chosen={}
    names=set(planned) | set(levels)
    for task in names:
        level=levels.get(task)
        if level is None:
            choices=planned.get(task,set());level=min(choices,key=c.LEVELS.index) if choices else None
        chosen[task]=level
        candidates=reference if level=='pure' and level not in arms else arms.get(level,{})
        data.update({k:v for k,v in candidates.items() if k[1]==task})
    return data,chosen


def r17(model,benchmark,mode,inventory_only):
    label=LABELS[model]; prefix=f'ffk1015_{model}_{benchmark}_{mode}_'
    if benchmark=='mt50':
        old_lanes=[EVAL/MT[model][mode=='full']]
        new_lanes=[Path(j['run']) for j in read(HERE/'CAMPAIGNS.json')['jobs'] if j['model']==model and j['benchmark']==benchmark and j['mode']==mode and j['stage']=='descent']
        old_result=KNOB/'mt50'/f'{model}_{mode}'/'test.json'
        new_result=HERE/'results'/f'{model}_{benchmark}_{mode}'/'dev/test.json'
    else:
        stem='knobtest'+SHORT[model]+('_red' if mode=='reduced' else '')
        old_lanes=[p for p in sorted(EVAL.glob(stem+'*')) if (p/'RECEIPT.json').exists() and ('_red' in p.name)==(mode=='reduced')]
        new_lanes=[p for p in sorted(EVAL.glob(prefix+'test*')) if (p/'EPISODE_PLAN.json').exists()]
        off=EVAL/(prefix+'off')
        if off.exists():new_lanes.append(off)
        old_result=KNOB/f'test_results_{FINAL[model,mode]}/test.json'
        outputs=sorted((HERE/'results'/f'{model}_{benchmark}_{mode}').glob('test*/test.json'))
        new_result=outputs[-1] if outputs else HERE/'NOT_RUN.json'
    new_lanes=continued_lanes(new_lanes)
    def result_rows(path):return {(r['suite'],r['size']):r for r in read(path)['cells']} if path.exists() else {}
    old_rows,new_rows=result_rows(old_result),result_rows(new_result)
    old_cells=old_planned=new_cells=new_planned={}
    references={}
    baseline_off={};baseline_off_complete=True
    if not inventory_only:
        old_cells,old_planned=c.load_knob(old_lanes,'A')
        completed=[p for p in new_lanes if (p/'RECEIPT.json').exists()]
        if completed:new_cells,new_planned=c.load_knob(completed,'A')
        extra=controls(model,mode) if benchmark=='libero' else []
        override=mt_controls(model,mode) if benchmark=='mt50' else extra[1::2]
        references=load_controls(c.CONTROL_ROOT,{label},mode,override,
                                 suites=('mt50',) if benchmark=='mt50' else c.cache.SUITES)
        if benchmark=='libero' and mode=='reduced':
            off=EVAL/(f'ffk1015_cosmos_libero_reduced_off_u' if model=='cosmos' else OFF[model])
            baseline_off_complete=(off/'RECEIPT.json').exists()
            if baseline_off_complete:
                off_cells,_=c.load_knob([off],'A')
                for key,arms in off_cells.items():
                    ref=references[key[0],key[1]];data=arms['koff']
                    if len(data)!=500:baseline_off_complete=False
                    elif sum(r['success'] for r in data.values())>=sum(r['success'] for r in ref.values()):baseline_off[key]=data
    result=[]
    resources=[segment for lane in new_lanes for segment in segment_resources(read(lane/'campaign.json'),lane)]
    for (actual,suite,size),cell in cells().items():
        if actual!=model or (suite=='mt50')!=(benchmark=='mt50') or cell['source']!='R17':continue
        old,new=old_rows.get((suite,size),{}),new_rows.get((suite,size),{})
        key=label,suite,size
        levels=lambda row:{t:v.get('selected',v.get('level')) for t,v in row.get('tasks',{}).items()}
        fd,fl=select(new_cells.get(key,{}),new_planned.get(key,{}),levels(new),references.get((label,suite),{}))
        bd,bl=select(old_cells.get(key,{}),old_planned.get(key,{}),levels(old),references.get((label,suite),{}))
        if key in baseline_off:
            bd=baseline_off[key];bl={k[1]:'koff' for k in bd}
        fs=stats(fd,model,suite,mode) if fd else None
        bs=stats(bd,model,suite,mode) if bd else ({k:old.get(k) for k in ('sr','group_mean','task_mean','n','ir')} if old else None)
        complete=len(fd)==len(bd)==(2500 if benchmark=='mt50' else 500) and bool(new) and not new.get('incomplete',True) and baseline_off_complete
        result.append(dict(model=model,benchmark=benchmark,suite=suite,step_mode=mode,size=size,
            collected=cell['collected'],filtered_episodes_kept=cell['kept'],removed_failures=cell['failed'],
            empty_tasks=cell['empty_tasks'],empty_tasks_are_simulated=True,per_task_levels=fl or levels(new),
            unfiltered_per_task_levels=bl or levels(old),filtered=fs,unfiltered=bs,
            unfiltered_no_knob=key in baseline_off,
            paired=comparison(fd,bd,benchmark=='mt50') if fd and bd else None,
            lane_execution_segments=resources,
            status='COMPLETE' if complete else 'PENDING',
            unfiltered_result=str(old_result),filtered_result=str(new_result)))
    return result


def vla_rows(mode):
    full=HERE/'vla_full_r15/RESULTS.json'
    if mode=='full' and full.exists():return read(full)['rows']
    source=Path('/home/weiland/projects/openpi/exp/offline_search/rounds/r16/k1ref/results/retest.json')
    old=read(source);final=read(HERE/'vla/FINAL.json') if (HERE/'vla/FINAL.json').exists() else None
    rows=[]
    for cell in read(HERE/'vla/CELLS.json')['cells']:
        name=cell['cell'];s=cell['suite'];fd={};bd={}
        for target,packed in ((bd,old['merged_episodes'][name]),(fd,final['episodes'][name] if final and mode=='reduced' else None)):
            if packed is None:continue
            for t in range(10):
                for i,y in enumerate(packed['y'][t]):target[s,str(t),i]=dict(success=bool(y))
        summary=next(r for r in old['summary'] if r['model']==cell['model'] and r['suite']==s and r['size']==cell['size'])
        result=dict(model=cell['model'],benchmark='libero',suite=s,step_mode=mode,size=cell['size'],
            collected=cell['collected'] if mode=='reduced' else None,
            filtered_episodes_kept=cell['kept'] if mode=='reduced' else None,
            removed_failures=cell['failed'] if mode=='reduced' else None,
            empty_tasks=cell['empty_tasks'] if mode=='reduced' else None,
            per_task_levels=final['selection']['levels'][name] if final and mode=='reduced' else {},
            unfiltered_per_task_levels=old['selection']['levels'][name] if mode=='reduced' else {},
            filtered=None,unfiltered=dict(sr=summary['sr'],ir=summary['ir']) if mode=='reduced' else None,
            paired=comparison(fd,bd,False) if fd and mode=='reduced' else None,
            unfiltered_result=str(source) if mode=='reduced' else None,
            filtered_result=str(HERE/'vla/FINAL.json') if mode=='reduced' else None,
            status='COMPLETE' if fd and mode=='reduced' else 'PENDING' if mode=='reduced' else 'AWAITING_FULL_STEP_DECISION')
        if fd:
            p=final['episodes'][name]
            controls=sum(map(sum,p['n']));result['filtered']=dict(sr=sum(r['success'] for r in fd.values())/500,
                ir=sum(map(sum,p['cost']))/controls if controls else None)
        rows.append(result)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inventory-only',action='store_true')
    p.add_argument('--output',type=Path,default=HERE/'analysis/cells.json');args=p.parse_args()
    rows=[]
    for mode in ('reduced','full'):
        for model in MT:rows+=r17(model,'mt50',mode,args.inventory_only)
        for model in SHORT:rows+=r17(model,'libero',mode,args.inventory_only)
        rows+=vla_rows(mode)
    dump(args.output,dict(rows=rows,refit=False,measured_ir_not_recalibrated=True,
                         paired_unit='original suite/task/init; MT50 primary equal difficulty-group means'))
    print(f'{len(rows)} cells -> {args.output}')


if __name__=='__main__':main()
