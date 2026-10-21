"""Off-first formal-A queue definitions. Only dispatch() prepares or launches."""
from __future__ import annotations

import argparse
import copy
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
import re
from pathlib import Path
import shlex
import subprocess
from unittest.mock import patch

from . import formal_a as f
from .common import read,dump,sha
from .commands import script,PREFIX,LAUNCH_PREFIX
from . import formal_a_names as names

INDEX=f.ROOT/'QUEUE.json'
DRAFTS=f.ROOT/'QUEUE_DRAFTS.json'
from exp.offline_search.rounds.r17.scheduler.defaults import SHARED, physical_maximum
# R17 units: the shared pool (formal-A is LIBERO). Staging applies each batch's eligible nodes.
WORKERS = {n: physical_maximum(n, 'libero') for n in SHARED}


def queue_workers(u):
    """R17: the shared pool. R15/R16: each listed node at its physical maximum (timan108 -> timan107)."""
    if u['model'] in f.TEST_SOURCE:
        return dict(WORKERS)
    return {n: physical_maximum(n, 'libero') for n in sorted({'timan107' if n == 'timan108' else n for n in u['workers']})}


def name(model,mode,arm,stage):
    return names.unit(model,mode,arm,stage) if names.MAP.exists() else f'ffk1015_{model}_libero_{mode}_formal_A_{arm}_{stage}'
def templates():return read(DRAFTS)['jobs']
def cells(model):return [(s,n) for s in ('libero_10','libero_spatial') for n in ((50,100,200,300,400,500) if model in f.TEST_SOURCE else (50,100,200,500))]
def off_level(model):return 'koff' if model in f.TEST_SOURCE else 'off'
def levels(model):return (f.R17_LEVELS if model in f.TEST_SOURCE else f.VLA_LEVELS)[:-1]


def fresh(root):
    if any((root/n).exists() for n in ('LAUNCH.json','launch.json','PREPARED.json','RECEIPT.json','DONE.json')):
        raise ValueError(f'Never change a prepared/launched run: {root}')


def put(unit):
    index=read(INDEX) if INDEX.exists() else dict(units=[])
    index['units']=[u for u in index['units'] if u['name']!=unit['name']]+[unit]
    dump(INDEX,index)
    dump(f.ROOT/'queue'/unit['name']/'UNIT.json',unit)


def unit(model,mode,arm,stage):
    source=next(j for j in templates() if (j['model'],j['mode'],j['arm'])==(model,mode,arm))
    deps=[]
    if stage=='levels':
        if mode=='full':
            deps=[name(model,mode,arm,'off')]
            if model=='pi05':
                overrides=read(f.ROOT/'PURE_A_OVERRIDES.json')
                deps += [Path(overrides['pi05:full:'+s]['root']).name for s in ('libero_10','libero_spatial')]
        else:
            p=Path(source['off_reuse']);deps=[p.parent.name if p.name=='group.json' else p.name]
            if model in ('pi05','groot') and arm=='original':deps.append(name(model,mode,arm,'off_complete'))
    return dict(name=name(model,mode,arm,stage),model=model,mode=mode,arm=arm,stage=stage,
        gpu=source['gpu'],replicas=source['replicas'],batch=source['batch'],
        workers=queue_workers(dict(source,model=model)),jobs=[],status='WAITING_OFF',episodes=None,
        maximum_episodes=len(cells(model))*len(levels(model))*500,dependencies=deps)


def references(model,mode):
    from . import formal_a_analysis as a
    if model not in f.TEST_SOURCE:return a.vla_references(model,mode)
    from exp.offline_search.rounds.r17.knob.test import load_controls
    return {s:d for (m,s),d in load_controls(f.EVAL,{f.LABELS[model]},mode,[f.LABELS[model]+'='+f.PURE[model][mode]]).items()}


def classify(off,pure):
    complete=len(pure)==500 and set(off)==set(pure)
    return dict(n=len(off),pure_n=len(pure),successes=sum(e['success'] for e in off.values()),
        pure_successes=sum(e['success'] for e in pure.values()),
        status=('skip' if sum(e['success'] for e in off.values())>=sum(e['success'] for e in pure.values()) else 'tune') if complete else 'pending')


def evaluate(model,mode,arm):
    from . import formal_a_analysis as a
    data={};jobs=[]
    if mode=='reduced':
        jobs=[dict(j,reuse_only=True) for j in templates() if (j['model'],j['mode'],j['arm'])==(model,mode,arm)]
    if INDEX.exists():
        jobs += [j for u in read(INDEX)['units'] if (u['model'],u['mode'],u['arm'])==(model,mode,arm) and u['stage'] in ('off','off_complete') for j in u['jobs']]
    for j in jobs:
        loaded=a.load_r17(j) if j['pipeline']=='R17' else a.load_vla(j)
        for cell,arms in loaded.items():data.setdefault(cell,{}).update(arms.get(off_level(model),{}))
    pure=references(model,mode);rows=[]
    for suite,size in cells(model):
        rows.append(dict(suite=suite,size=size,**classify(data.get((suite,size),{}),pure.get(suite,{}))))
    result=dict(model=model,mode=mode,arm=arm,rows=rows,reference='same-step same-GPU pure A',dev_outcomes_used=False)
    dump(f.ROOT/'off_decisions'/f'{model}_{mode}_{arm}.json',result)
    return result


def r17_job(u,keep,selected_levels):
    from . import build as b
    from exp.offline_search.rounds.r17.filter_failed.resume import topology
    from exp.r17_integration.ops.formal_eval import open_tunnels
    old=next(j for j in templates() if (j['model'],j['mode'],j['arm'])==(u['model'],u['mode'],u['arm']))
    target=f.EVAL/u['name']
    if target.exists():
        fresh(target)
        j=read(target/'FFK_DEFINITION.json')
    else:
        with redirect_stdout(io.StringIO()):
            j=b.build(old['source'],u['name'],unfiltered=u['arm']=='original',ladder_source=old['run'],
                      test_levels=selected_levels,test_cells=set(keep))
    root=Path(j['run']);campaign=read(root/'campaign.json');campaign['runtime']=old['runtime']
    if campaign['server_node']=='local' and any(n!='local' for n in u['workers']) and not 23100<=campaign['public_port']<=23199:
        before=campaign['public_port'];delta=23160-before
        campaign['public_port']=23160
        campaign['backend_ports']=[p+delta for p in campaign['backend_ports']]
        campaign['ports']=[p+delta if p==before or p in [q-delta for q in campaign['backend_ports']] else p for p in campaign['ports']]
        campaign['formal_A_endpoint_override']=dict(source_public_port=before,public_port=23160,
            reason='Existing direct-port range required when local-only source adds remote execution workers')
    dump(root/'campaign.json',campaign)
    with redirect_stdout(io.StringIO()):
        topology(root,','.join(f'{k}={v}' for k,v in u['workers'].items()))
    campaign=read(root/'campaign.json');open_tunnels(campaign,root);dump(root/'campaign.json',campaign)
    _,model,tasks,_=b.cpu_plan(campaign);plan=read(root/'EPISODE_PLAN.json')
    plan.update(model=model,tasks=[asdict(t) for t in tasks],expected_episodes=len(tasks),endpoint=campaign['endpoint']);dump(root/'EPISODE_PLAN.json',plan)
    j.update(model=u['model'],mode=u['mode'],arm=u['arm'],stage=u['stage'],pipeline='R17',runtime=old['runtime'],
        workers=u['workers'],batch=u['batch'],mps=old.get('mps'),levels=list(selected_levels),off_reuse=None,episodes=len(tasks))
    dump(root/'FFK_DEFINITION.json',j)
    return j


def r15_jobs(u,keep,selected_levels):
    p=f.vf.p;jobs=[]
    for old in templates():
        if (old['model'],old['mode'],old['arm'])!=(u['model'],u['mode'],u['arm']):continue
        src=Path(old['run']);rows=read(src/'arms.json');cell=read(src/'ROOT.json')['cell']
        if (rows[0]['suite'],rows[0]['r10_size']) not in keep:continue
        short=u['name'].startswith('fa_')
        root=f.EVAL/names.cell_name(u['name'],cell) if short else f.ROOT/'queue'/u['name']/(u['name']+'_'+cell)
        manifest=root/'eval500.json'
        if not root.exists():
            from exp.r17_integration.ops.immutable_release import nvme_dir
            nvme_dir(root);manifest.write_bytes((src/'eval500.json').read_bytes());specs=[]  # real dir on /nvme
            catalog={r['arm']:r for r in read(src/'arms.json')};inputs=read(src/'arms_in.json')
            for level in selected_levels:
                original=next(s for s in inputs if catalog[s['name']]['r14_descent_level']==level)
                namespace=(u.get('renamed_from',u['name'])+'_'+cell) if short else root.name
                spec=copy.deepcopy(original);spec.update(name=namespace+'_'+level,manifest=str(manifest));specs.append(spec)
            p.emit(root,cell,specs,selected_levels,pool='A',final_root=root)
        j=dict(old,name=root.name,run=str(root),stage=u['stage'],episodes=500*len(selected_levels),levels=list(selected_levels),
            artifacts=str(src.parent/'ARTIFACTS.json'))
        if u.get('worker_host_failover'):j['worker_host_failover']=True
        jobs.append(j)
    return jobs


def r16_jobs(u,keep,selected_levels):
    from exp.offline_search.rounds.r16 import conductor as c,analyze as a
    old=next(j for j in templates() if (j['model'],j['mode'],j['arm'])==(u['model'],u['mode'],u['arm']))
    source=read(Path(old['run'])/'group.json');root=f.EVAL/u['name'] if u['name'].startswith('fa_') else f.ROOT/'queue'/u['name']/u['name']
    entries=[copy.deepcopy(e) for e in source['entries'] if e['formal_grid_level'] in selected_levels and
             (e['row']['suite'],e['row']['r10_size']) in keep]
    if not entries:return []
    from exp.r17_integration.ops.immutable_release import nvme_dir
    nvme_dir(root,exist_ok=True)  # real dir on /nvme, /data path a symlink
    group=dict(source,root=str(root),entries=entries);group.pop('group_sha256',None)
    group['group_sha256']=a.sha(c.canonical(group));dump(root/'group.json',group)
    dump(root/'ROOT.json',dict(group_sha256=group['group_sha256'],definitions_sha256=group['definitions_sha256']))
    return [dict(old,name=u['name'],run=str(root),stage=u['stage'],episodes=sum(len(e['pairs']) for e in entries),levels=list(selected_levels))]


def materialize(u,keep,selected_levels):
    u = dict(u, workers=queue_workers(u))
    if not keep:jobs=[]
    elif u['model'] in f.TEST_SOURCE:jobs=[r17_job(u,keep,selected_levels)]
    elif u['mode']=='full':jobs=r15_jobs(u,keep,selected_levels)
    else:jobs=r16_jobs(u,keep,selected_levels)
    u.update(jobs=jobs,status='BUILT' if jobs else 'SKIP_OFF_SUFFICIENT',episodes=sum(j['episodes'] for j in jobs),
             selected_cells=[list(c) for c in keep],levels=list(selected_levels))
    put(u);print('QUEUE_DEFINED',u['name'],u['episodes'],u['status'],flush=True)
    return u


def pure_jobs():
    p=f.vf.p;overrides=read(f.ROOT/'PURE_A_OVERRIDES.json') if (f.ROOT/'PURE_A_OVERRIDES.json').exists() else {}
    for short in ('l10','spatial'):
        n=f'ffk1015_pi05_libero_full_formal_A_pure_{short}';root=f.ROOT/'queue'/n/n;cell=f'pi05_{short}_50'
        if not root.exists():
            root.mkdir(parents=True);manifest=root/'eval500.json';manifest.write_bytes((f.vf.TRACE/'r11_knob_1/eval500.json').read_bytes())
            fit=f.vf.original(cell,'c10',True);p.emit(root,cell,[p.arm_spec(cell,n+'_c10',fit,manifest)],['c10'],pool='A',final_root=root)
        top=f.vf.topology(cell,'task_descent',True)
        j=dict(name=n,run=str(root),source=top['source_root'],model='pi05',mode='full',arm='pure_reference',pipeline='R15',stage='pure',
            episodes=500,gpu='H100',replicas=top['replicas'],batch=1,workers=top['workers'],topology=top,levels=['c10'])
        put(dict(j,jobs=[j],status='BUILT',maximum_episodes=500))
        suite='libero_10' if short=='l10' else 'libero_spatial'
        overrides[f'pi05:full:{suite}']=dict(root=str(root),arm=read(root/'arms.json')[0]['arm'],authorized=True,shared_by_both_arms=True)
    dump(f.ROOT/'PURE_A_OVERRIDES.json',overrides)


def build_levels(model,mode,arm,*,require_complete=True,decision=None):
    n=name(model,mode,arm,'levels');u=next(u for u in read(INDEX)['units'] if u['name']==n)
    if u['status']!='WAITING_OFF':return u
    decision=decision or evaluate(model,mode,arm)
    dump(f.ROOT/'off_decisions'/f'{model}_{mode}_{arm}.json',decision)
    pending=[r for r in decision['rows'] if r['status']=='pending']
    if pending:
        u.update(off_decision=str(f.ROOT/'off_decisions'/f'{model}_{mode}_{arm}.json'),pending_cells=pending);put(u)
        if require_complete:raise ValueError(f'{n}: wait for complete off and pure A cohorts; pending {[(r["suite"],r["size"],r["n"]) for r in pending]}')
        return u
    keep=[(r['suite'],r['size']) for r in decision['rows'] if r['status']=='tune']
    u['off_decision_sha256']=sha(f.ROOT/'off_decisions'/f'{model}_{mode}_{arm}.json')
    return materialize(u,keep,levels(model))


def initialize():
    if not DRAFTS.exists():dump(DRAFTS,read(f.ROOT/'CAMPAIGNS.json'))
    for j in templates():fresh(Path(j['run']))
    if INDEX.exists():
        index=read(INDEX)
        if index.get('initialized'):raise ValueError('Queue already defined; use build-levels for pending stages')
        for u in index['units']:
            for j in u['jobs']:fresh(Path(j['run']))
    pure_jobs()
    for model in f.MODELS:
        for mode in ('reduced','full'):
            for arm in f.ARMS:
                existing={u['name']:u for u in read(INDEX)['units']}
                if mode=='full' and name(model,mode,arm,'off') not in existing:
                    materialize(unit(model,mode,arm,'off'),cells(model),(off_level(model),))
                if mode=='reduced' and model in ('pi05','groot') and arm=='original' and name(model,mode,arm,'off_complete') not in existing:
                    materialize(unit(model,mode,arm,'off_complete'),cells(model),('off',))
                if name(model,mode,arm,'levels') not in existing:put(unit(model,mode,arm,'levels'))
                if mode=='reduced':
                    # These were just measured by the standard loaders for this
                    # review. Pending data never authorizes non-off dispatch.
                    audit=f.ROOT/'review_off_audit'/f'{model}_{arm}.json'
                    decision=None
                    if audit.exists():
                        rows=read(audit)['rows']
                        decision=dict(model=model,mode=mode,arm=arm,source_audit=str(audit),source_audit_sha256=sha(audit),
                            rows=[dict(r,status=('skip' if r['successes']>=r['pure_successes'] else 'tune') if r['complete'] else 'pending') for r in rows])
                    build_levels(model,mode,arm,require_complete=False,decision=decision)
    publish()
    index=read(INDEX);index['initialized']=True;dump(INDEX,index)


def analysis_jobs(include_reuse=True):
    jobs=[names.resolve_job(j) for u in read(INDEX)['units'] if u['stage']!='pure' or not include_reuse for j in u['jobs']]
    if include_reuse:jobs += [dict(j,reuse_only=True) for j in templates() if j['mode']=='reduced']
    return jobs


def publish():
    units=read(INDEX)['units'];children=[j for u in units for j in u['jobs']]
    fleet=Path('/home/weiland/projects/openpi_ext/scratch/r17_ops/ff')
    queue_text=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    queue_text+='\n'.join(p.read_text() for p in fleet.glob('run_*.sh'))
    protected=set(re.findall(r'(?:ffk1015|fa)_[A-Za-z0-9_-]+',queue_text))
    def ensure(path,lines):
        body='#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(f.REPO))+'\n'+'\n'.join(lines)+'\n'
        if path.exists():
            if path.read_text()!=body:
                affected=[u for u in units if u['name'] in path.name]
                if any(u['name'] in protected for u in affected):raise ValueError(f'Never replace a queued command: {path}')
                for u in affected:
                    for j in u['jobs']:fresh(Path(j['run']))
                script(path,lines)
        else:script(path,lines)
    def native(path,lines):
        if '_profile_' in path.name:return
        target=path.with_name(path.stem+'.native.sh');ensure(target,lines)
        n=path.stem.removeprefix('prepare_').removeprefix('launch_')
        if n not in {u['name'] for u in units}:
            for base in (f.ROOT/'commands',f.HERE/'commands'):ensure(base/path.name,[f'bash {target}'])
    with patch.object(f,'script',native):f.commands(children)
    for u in units:
        for action in ('prepare','launch'):
            prefix=PREFIX if action=='prepare' else LAUNCH_PREFIX
            line=f'{prefix} -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a_queue dispatch {action} --name {u["name"]}'
            for base in (f.ROOT/'commands',f.HERE/'commands'):ensure(base/f'{action}_{u["name"]}.sh',[line])
    for gpu,file in (('H100','H100_ORDER.txt'),('RTX 4090','4090_ORDER.txt')):
        order=sorted((u for u in units if u['gpu']==gpu),key=lambda u:({'pure':0,'off_complete':1,'off':1,'levels':2}[u['stage']],u['mode']!='reduced',f.MODELS.index(u['model']),u['arm']!='original',u['name']))
        (f.ROOT/'commands'/file).write_text(''.join(u['name']+'\n' for u in order))
    totals=[]
    for model in f.MODELS:
        for mode in ('reduced','full'):
            for arm in f.ARMS:
                subset=[u for u in units if (u['model'],u['mode'],u['arm'])==(model,mode,arm)]
                totals.append(dict(model=model,mode=mode,arm=arm,ready_episodes=sum(u['episodes'] or 0 for u in subset),
                    deferred_upper_bound=sum(u['maximum_episodes'] for u in subset if u['episodes'] is None),
                    units=[u['name'] for u in subset]))
    dump(f.ROOT/'QUEUE_TOTALS.json',dict(rows=totals,pure_reference_episodes=1000,ready_episodes=sum(r['ready_episodes'] for r in totals)+1000,
        deferred_upper_bound=sum(r['deferred_upper_bound'] for r in totals)))
    dump(f.ROOT/'CAMPAIGNS.json',dict(jobs=children,queue_units=units,old_unlaunched_drafts=str(DRAFTS)))
    dump(f.ROOT/'TOTALS.json',read(f.ROOT/'QUEUE_TOTALS.json'))
    dump(f.ROOT/'LAUNCH_ORDER.json',dict(units=units,per_gpu_lists=['commands/H100_ORDER.txt','commands/4090_ORDER.txt'],
        pi05_full_hold=False,pure_references_authorized=True,rule='Pure references first, reduced first, off before levels; serialize shared worker nodes'))


def dispatch(action,n):
    u=next(u for u in read(INDEX)['units'] if u['name']==n)
    if action=='prepare' and u['status']=='WAITING_OFF':
        u=build_levels(u['model'],u['mode'],u['arm']);publish()
    if u['status']=='WAITING_OFF':raise ValueError('Prepare this unit after its off dependency finishes')
    if not u['jobs']:print('SKIP_OFF_SUFFICIENT',n,0,flush=True)
    for j in u['jobs']:
        subprocess.run(['bash',str(f.ROOT/'commands'/f'{action}_{j["name"]}.native.sh')],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['init','build-levels','dispatch','publish'])
    p.add_argument('dispatch_action',nargs='?',choices=['prepare','launch']);p.add_argument('--name')
    p.add_argument('--model',choices=f.MODELS);p.add_argument('--mode',choices=['full','reduced'],default='full');p.add_argument('--arm',choices=f.ARMS)
    args=p.parse_args()
    if args.action=='init':initialize()
    elif args.action=='publish':publish()
    elif args.action=='build-levels':build_levels(args.model,args.mode,args.arm);publish()
    else:dispatch(args.dispatch_action,args.name)


if __name__=='__main__':main()
