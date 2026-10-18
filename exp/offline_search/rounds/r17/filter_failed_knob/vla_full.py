"""R15 full-step definitions and analysis; only candidate rows are filtered.

R15's emitter, serving classes, planner, accepted-attempt loaders and numerical
analysis are reused. Size 50 adds catalog entries from the existing R13 fits.
Only the coordinator invokes prepare/launch; build/check/freeze/analyze are CPU.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import copy
import gc
import io
import json
import os
from pathlib import Path
import pickle
import shlex
import sys
from unittest.mock import patch

import numpy as np

from .common import HERE, LEGACY, read, dump, sha, summarize
from .vla import candidates
from exp.offline_search.closed_loop import devset
from exp.offline_search.rounds.r15.descent import prepare as p, analyze as a, analyze_test_a as aa
from exp.offline_search.rounds.r14.descent import prepare as old50
from exp.offline_search.rounds.r17.filter_failed.row_filter import R10_CANDIDATES

ROOT=HERE/'vla_full_r15'
SOURCE=p.HERE
R14_SOURCE=HERE.parents[1]/'r14/descent_astra'
TRACE=p.TRACE
CELLS=tuple(f'{m}_{s}_{n}' for m in ('pi05','groot') for s in ('l10','spatial') for n in (50,100,200,500))
ARMS=('filtered',)
CPU='env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python'
MODULE='exp.offline_search.rounds.r17.filter_failed_knob.vla_full'


def arms(arm):
    if arm not in ARMS:raise ValueError('Only filtered knob arms are authorized')
    return CELLS


def original(cell,level,test=False):
    return copy.deepcopy(old50.frozen(cell,level) if cell.endswith('_50') else p.frozen(cell,level,test=test))


def unchanged(old,new):
    """The reused slicer must share every field outside candidate arrays."""
    for left,right,omit in ((old['method'],new['method'],{'inner','_full_tasks'}),
            (old['method'].inner,new['method'].inner,{'base'}),
            (old['method'].inner.base,new['method'].inner.base,{'tasks'})):
        assert type(left) is type(right)
        for key,value in vars(left).items():
            if key not in omit:assert getattr(right,key) is value,(type(left).__name__,key)
    for task,table in new['method'].inner.base.tasks.items():
        for key,value in vars(old['method'].inner.base.tasks[task]).items():
            if key not in R10_CANDIDATES:assert getattr(table,key) is value,key


def topology(cell,level,test=False):
    """Use the actual R15 arm's recorded ports/counts, including early R3 dev."""
    source_cell=cell.rsplit('_',1)[0]+'_100' if cell.endswith('_50') else cell
    root=TRACE/(('r15_descent_test_' if test else 'r15_descent_')+source_cell)
    rows=read(root/'arms.json')
    row=next(r for r in rows if r['r14_descent_level']==('task_descent' if test else level))
    record=read(root/'runs'/row['arm']/'h100_launch.json')
    ports=[s['port'] for s in record['servers']]
    workers=record['driver']['workers'];assert len(set(workers))==1
    host=record.get('server_host','h100');node=record['worker_host']
    assert node!='timan1'
    env=dict(SERVER_HOST=host,WORKER_HOST=node,PORTS=','.join(map(str,ports)),WPS=str(workers[0]),
             LOCAL_SERVER_COUNT=str(len(ports)),OMP='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    env.update(record['servers'][0]['env'])
    env.update(NEED_MB=str(record['servers'][0]['need_mb']),SYNC_PORT='23195' if node=='timan107' else '23196',
               MAX_ATTEMPTS='3',POLL_SECONDS='60')
    return dict(source_root=str(root),source_arm=row['arm'],source_receipt_sha256=sha(root/'runs'/row['arm']/'h100_launch.json'),
                gpu='RTX 4090' if host=='local' else 'H100',workers={node:sum(workers)},replicas=len(ports),env=env)


@contextmanager
def catalog(arm):
    # Configuration lookup only; no serving/fit/selection method is replaced.
    with patch.object(p,'HERE',ROOT/arm),patch.object(p,'CELLS',arms(arm)):
        yield


def root_for(cell,arm,phase):
    model,suite,size=cell.split('_')
    return ROOT/'runs'/f'ffk1015_{model}_libero_full_r15_{suite}_{size}_{arm}_{phase}'


def build():
    ROOT.mkdir(exist_ok=False)
    selection=copy.deepcopy(read(SOURCE/'selection.json'))
    for key,entry in selection['model_suites'].items():
        model,suite=key.split('_');entry['served_subsets']['50']=devset.r10_subset(model,suite,50)
    dump(ROOT/'selection.json',selection)
    cells=[];jobs=[];proof=[]
    for arm in ARMS:
        fits={};caps={}
        for cell in arms(arm):
            model,suite,size=cell.split('_');size=int(size)
            off=original(cell,'off');parent=LEGACY/f'{model}_{suite}'/off['kwargs']['library']
            ids=sorted(e for es in off['kwargs']['episode_subset'].values() for e in es)
            summary=summarize(read(parent/'episodes.json'),ids,id_key='episode_id',task_key='task_id')
            if arm=='filtered':cells.append(dict(cell=cell,model=model,suite='libero_10' if suite=='l10' else 'libero_spatial',size=size,
                parent=str(parent),labels=str(parent/'episodes.json'),**summary))
            fits[cell]={};cap={};dev_sha={};test_sha={}
            for level in p.LEVELS:
                records={}
                for test in (False,True):
                    rec=original(cell,level,test);src=Path(rec['path'])
                    target=ROOT/arm/'artifacts'/f'{cell}_{level}_{"A" if test else "B"}.pkl'
                    target.parent.mkdir(parents=True,exist_ok=True)
                    data=src.read_bytes();blob=pickle.loads(data)
                    assert set(np.unique(blob['method'].inner.base.lib_ep))==set(ids)
                    if arm=='filtered' and level!='c10':
                        filtered=candidates(blob,summary['kept_episode_ids']);unchanged(blob,filtered)
                        data=pickle.dumps(filtered,protocol=4)
                    target.write_bytes(data)
                    records[test]=dict(rec,path=str(target),sha256=sha(target))
                    if not test:cap[level]=0. if level in ('off','c10') else float(blob['method'].knob_settings['setting'])
                    proof.append(dict(arm=arm,cell=cell,level=level,pool='A' if test else 'B',source=str(src),source_sha256=sha(src),
                        path=str(target),sha256=sha(target),no_refit=True,fitted_fields_unchanged=True,
                        no_filter_byte_identical=sha(target)==sha(src) if level=='c10' else None))
                    del blob;gc.collect()
                rec=records[False]
                if rec.get('serving_wrapper'):
                    rec.update(source_path=records[True]['path'],source_sha256=records[True]['sha256'])
                fits[cell][level]=rec;dev_sha[level]=rec['sha256'];test_sha[level]=records[True]['sha256']
            caps[cell]=dict(periodic_cap_by_level=cap,fit_sha256_by_level=dev_sha,test_fit_sha256_by_level=test_sha)
        dump(ROOT/arm/'fits.json',fits);dump(ROOT/arm/'cap_tables.json',dict(cells=caps))
        for cell in arms(arm):
            root=root_for(cell,arm,'dev');model,suite,size=cell.split('_')
            manifest=root/'dev_selection.json';template=copy.deepcopy(read(TRACE/f'r15_descent_{model}_{suite}_100/dev_selection.json'))
            template['selection_sha256']=sha(ROOT/'selection.json');dump(manifest,template)
            with catalog(arm):
                specs=[p.arm_spec(cell,p.arm_name(cell,level),p.frozen(cell,level),manifest) for level in p.DEV_ORDER]
                p.emit(root,cell,specs,p.DEV_ORDER,pool='B',final_root=root)
            job=dict(name=root.name,root=str(root),cell=cell,arm=arm,phase='dev',episodes=1500,
                     levels={level:topology(cell,level) for level in p.DEV_ORDER})
            dump(root/'FFK_R15.json',job);jobs.append(job)
            print('READY',root.name,'1500',flush=True)
    dump(ROOT/'CELLS.json',dict(cells=cells));dump(ROOT/'ARTIFACTS.json',dict(rows=proof))
    dump(ROOT/'CAMPAIGNS.json',dict(jobs=jobs));write_commands(jobs)


def write_commands(jobs):
    from .commands import script
    for model in ('pi05','groot'):
        for wave,level in enumerate(p.DEV_ORDER,1):
            selected=[j for j in jobs if j['phase']=='dev' and j['cell'].startswith(model+'_')]
            for action in ('prepare','launch'):
                lines=[]
                for job in selected:
                    env=job['levels'][level]['env'];prefix='env '+('CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu ' if action=='prepare' else '-u CUDA_VISIBLE_DEVICES -u JAX_PLATFORMS ')
                    prefix+=' '.join(shlex.quote(k+'='+v) for k,v in env.items())+' PYTHONPATH=.:src .venv/bin/python'
                    mode='sync --concurrent' if action=='prepare' else 'chain'
                    lines.append(prefix+' -m exp.offline_search.rounds.r17.filter_failed.legacy '+mode+' '+shlex.quote(job['root'])+' '+p.arm_name(job['cell'],level))
                script(ROOT/'commands'/f'{action}_{model}_dev{wave:02d}.sh',lines)


def references():
    """R15's pi05 A pure receipts say local; provide matching H100 references."""
    proof=copy.deepcopy(read(SOURCE/'c10_proof.json'));jobs=read(ROOT/'CAMPAIGNS.json')['jobs']
    audit=[]
    for key,ref in list(proof['test_references'].items()):
        old=read(Path(ref['root'])/'runs'/ref['arm']/'h100_launch.json')
        actual=old.get('server_host','h100');expected='h100' if key.startswith('pi05_') else 'local'
        audit.append(dict(reference=key,source=ref['root'],actual_server=actual,required_server=expected,reused=actual==expected))
        if actual==expected:continue
        assert key.startswith('pi05_')
        model,suite=key.split('_');cell=key+'_50'
        root=ROOT/'runs'/f'ffk1015_{model}_libero_full_r15_{suite}_pure_A'
        manifest=root/'eval500.json';manifest.parent.mkdir(parents=True,exist_ok=False)
        manifest.write_bytes((TRACE/'r11_knob_1/eval500.json').read_bytes())
        fit=old50.frozen(cell,'c10');name=p.arm_name(cell,'c10',test=True)
        p.emit(root,cell,[p.arm_spec(cell,name,fit,manifest)],['c10'],pool='A',final_root=root)
        proof['test_references'][key]=dict(ref,root=str(root),arm=name,arms_sha256=sha(root/'arms.json'),
            manifest=str(manifest),fit_path=fit['path'],fit_sha256=fit['sha256'],host='h100',reusable=False,
            reason='R15 pi05 A reference actually used local 4090; matching H100 pure reference required')
        job=dict(name=root.name,root=str(root),cell=cell,arm='pure_reference',phase='pure-test',episodes=500,
                 topology=topology(cell,'task_descent',True))
        dump(root/'FFK_R15.json',job);jobs.append(job)
    dump(ROOT/'REFERENCES.json',proof);dump(ROOT/'REFERENCE_HOSTS.json',dict(rows=audit))
    for arm in ARMS:dump(ROOT/arm/'c10_proof.json',proof)
    dump(ROOT/'CAMPAIGNS.json',dict(jobs=jobs))
    from .commands import script
    for action in ('prepare','launch'):
        lines=[]
        for job in jobs:
            if job['phase']!='pure-test':continue
            env=job['topology']['env'];prefix='env '+('CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu ' if action=='prepare' else '-u CUDA_VISIBLE_DEVICES -u JAX_PLATFORMS ')
            prefix+=' '.join(shlex.quote(k+'='+value) for k,value in env.items())+' PYTHONPATH=.:src .venv/bin/python'
            name=read(Path(job['root'])/'arms.json')[0]['arm']
            lines.append(prefix+' -m exp.offline_search.rounds.r17.filter_failed.legacy '+('sync --concurrent' if action=='prepare' else 'chain')+' '+job['root']+' '+name)
        script(ROOT/'commands'/f'{action}_pi05_pure_A.sh',lines)


def protocol():
    doc=copy.deepcopy(read(SOURCE/'protocol.json'))
    doc.update(cells=list(CELLS),root_prefix='ffk1015_',source_protocol_sha256=sha(SOURCE/'protocol.json'))
    doc['input_sha256'][str((ROOT/'selection.json').relative_to(p.REPO))]=sha(ROOT/'selection.json')
    if (ROOT/'REFERENCES.json').exists():doc['test_references']=read(ROOT/'REFERENCES.json')['test_references']
    return doc


def freeze(model):
    reader=a.Reader();doc=protocol();selection=read(ROOT/'selection.json');results={};outputs={}
    for arm in ARMS:
        caps=read(ROOT/arm/'cap_tables.json');levels={};uniform={}
        for cell in arms(arm):
            if not cell.startswith(model+'_'):continue
            # Validate the unchanged shared pure source against its own original
            # protocol, then validate the new five-arm root against its seal.
            refcell=cell.rsplit('_',1)[0]+'_100'
            with patch.object(a,'ARMS',('c10',)):
                pure=a.load_cell(TRACE/('r15_descent_c10_'+cell.rsplit('_',1)[0]),refcell,
                    read(SOURCE/'protocol.json'),read(SOURCE/'selection.json'),read(SOURCE/'cap_tables.json'),reader)
            with patch.object(a,'ARMS',p.DEV_ORDER),patch.object(a,'HERE',ROOT):
                observed=a.load_cell(root_for(cell,arm,'dev'),cell,doc,selection,caps,reader)
            row=a.analyze_cell(cell,dict(pure,**observed),doc,a.bootstrap_indices(doc))
            choice=row['tolerances']['5'];levels[cell]=choice['levels'];uniform[cell]=choice['uniform'];results[arm,cell]=row
        outputs[arm]=(levels,uniform)
    reader.stable()
    for arm,(levels,uniform) in outputs.items():
        out=ROOT/arm/'freeze'/model
        out.mkdir(parents=True,exist_ok=False)
        dump(out/'levels.json',levels);dump(out/'uniform.json',uniform)
        dump(out/'FREEZE.json',dict(inputs=reader.inputs,levels_sha256=sha(out/'levels.json'),uniform_sha256=sha(out/'uniform.json'),
                                   frozen_at_utc=datetime.now(timezone.utc).isoformat()))
    dump(ROOT/f'DESCENT_{model}.json',dict(cells={f'{arm}/{cell}':v for (arm,cell),v in results.items()}))


def emit_a(model):
    jobs=[]
    for arm in ARMS:
        levels,uniform,_=frozen_tables(arm,model)
        stage=ROOT/arm/f'test_{model}'
        with catalog(arm):records=p.emit_test_a(levels,uniform,stage)
        for rec in records:
            source=Path(rec['root']);target=root_for(rec['cell'],arm,'test')
            assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);source.rename(target)
            for file in target.rglob('*.json'):
                if file.name=='ROOT.json':continue
                dump(file,p.replace_owned(read(file),source,target))
            # Official identities retain their original bytes.
            (target/'eval500.json').write_bytes((TRACE/'r11_knob_1/eval500.json').read_bytes())
            p.seal(target,final_root=target,pool='A',cell=rec['cell'])
            top=topology(rec['cell'],'task_descent',True)
            job=dict(name=target.name,root=str(target),cell=rec['cell'],arm=arm,phase='test',episodes=1000,topology=top)
            dump(target/'FFK_R15.json',job);jobs.append(job)
    dump(ROOT/f'TEST_CAMPAIGNS_{model}.json',dict(jobs=jobs))
    from .commands import script
    for action in ('prepare','launch'):
        lines=[]
        for job in jobs:
            env=job['topology']['env'];prefix='env '+('CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu ' if action=='prepare' else '-u CUDA_VISIBLE_DEVICES -u JAX_PLATFORMS ')
            prefix+=' '.join(shlex.quote(k+'='+v) for k,v in env.items())+' PYTHONPATH=.:src .venv/bin/python'
            names=' '.join(r['arm'] for r in read(Path(job['root'])/'arms.json'))
            lines.append(prefix+' -m exp.offline_search.rounds.r17.filter_failed.legacy '+('sync --concurrent' if action=='prepare' else 'chain')+' '+job['root']+' '+names)
        script(ROOT/'commands'/f'{action}_{model}_test.sh',lines)


def frozen_tables(arm,model):
    root=ROOT/arm/'freeze'/model;freeze=read(root/'FREEZE.json')
    a.require(sha(root/'levels.json')==freeze['levels_sha256'] and sha(root/'uniform.json')==freeze['uniform_sha256'],
              'Changed frozen B tables')
    return read(root/'levels.json'),read(root/'uniform.json'),freeze


class Layout:
    def __init__(self,arm):self.arm=arm
    def __truediv__(self,name):
        return root_for(name.removeprefix('r15_descent_test_'),self.arm,'test')


def analyze():
    baselines={name:read(path/'test_a.json') for name,path in (('R14',R14_SOURCE),('R15',SOURCE))}
    doc=protocol();template=(TRACE/'r11_knob_1/eval500.json').read_bytes();rows=[]
    for cellinfo in read(ROOT/'CELLS.json')['cells']:
        cell=cellinfo['cell'];model=cellinfo['model'];observed={}
        for arm in ARMS:
            root=root_for(cell,arm,'test')
            if not (root/'descent_table.json').exists():continue
            if not all((root/'runs'/r['arm']/'summary.json').exists() for r in read(root/'arms.json')):continue
            levels,uniform,freeze=frozen_tables(arm,model)
            with patch.object(aa,'TRACE',Layout(arm)):
                data=aa.load_a(cell,levels[cell],uniform[cell],doc,read(ROOT/arm/'cap_tables.json'),a.Reader(),template)
            aa.check_freeze_order({cell:data},{model:dict(timestamp=datetime.fromisoformat(freeze['frozen_at_utc']).timestamp())})
            ep=data['task_descent'];observed[arm]=dict(y=ep.y.tolist(),cost=ep.cost.tolist(),n=ep.n.tolist())
        source_pipeline='R14' if cellinfo['size']==50 else 'R15'
        source=(R14_SOURCE if source_pipeline=='R14' else SOURCE)/'test_a.json'
        baseline=baselines[source_pipeline]
        old=baseline['episode_data'][cell]['task_descent']
        new=observed.get('filtered')
        def stats(data):
            if not data:return None
            return dict(sr=float(np.mean(data['y'])),ir=float(np.sum(data['cost'])/np.sum(data['n'])),n=500)
        pair=None
        if old and new:
            delta=np.asarray(new['y'],int)-np.asarray(old['y'],int)
            pair=dict(paired_n=500,difference=float(delta.mean()),by_init={str(i):float(delta[:,i].mean()) for i in range(50)},by_task_init=delta.tolist())
        freeze_path=ROOT/'filtered/freeze'/model/'levels.json'
        rows.append(dict(model=model,benchmark='libero',suite=cellinfo['suite'],step_mode='full',size=cellinfo['size'],
            collected=cellinfo['collected'],filtered_episodes_kept=cellinfo['kept'],removed_failures=cellinfo['failed'],empty_tasks=cellinfo['empty_tasks'],
            per_task_levels=read(freeze_path).get(cell,{}) if freeze_path.exists() else {},
            unfiltered_per_task_levels=baseline['levels'][cell],
            filtered=stats(new),unfiltered=stats(old),paired=pair,status='COMPLETE' if old and new else 'PENDING',
            unfiltered_source=str(source),unfiltered_source_sha256=sha(source),
            filtered_pipeline='R15',unfiltered_pipeline=source_pipeline,
            cross_pipeline_comparison=source_pipeline!='R15',
            comparison_pipeline=source_pipeline+'→R15',raises_upper_bound=0))
    dump(ROOT/'RESULTS.json',dict(rows=rows,refit=False,pipeline='R15',unfiltered_reruns=0,raises_upper_bound=0))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['build','references','freeze','emit-test','analyze','check'])
    parser.add_argument('--model',choices=['pi05','groot']);args=parser.parse_args()
    if args.action=='build':build();references();analyze()
    elif args.action=='references':references()
    elif args.action=='freeze':freeze(args.model)
    elif args.action=='emit-test':emit_a(args.model)
    elif args.action=='check':
        from .vla_full_check import check
        check()
    else:analyze()


if __name__=='__main__':main()
