"""CPU-only R15 startup/planner proof, with all launch/transport calls blocked."""
from __future__ import annotations

from contextlib import ExitStack
import gc
import json
import os
from pathlib import Path
import pickle
import socket
import subprocess
from unittest.mock import patch
import numpy as np

from . import vla_full as v
from .common import dump,read,sha
from exp.offline_search.closed_loop import plugin
from exp.offline_search.rounds.r15.descent import verify
from exp.offline_search.rounds.r15.descent.mechanics import Harness
from exp.offline_search.rounds.r15.descent.proof import lifecycle
from exp.offline_search.closed_loop.ops.h100 import control,node,local_server


def blocked(*a,**kw):raise AssertionError('No transport, process launch or server in CPU check')


def policy_identity(job,row,top):
    """Compare the stock R15 policy CLI with its actual source startup receipt."""
    root=Path(job['root']);port=int(top['env']['PORTS'].split(',')[0])
    with patch.dict(os.environ,top['env']):spec=control.server_spec(root,row,port)
    spec['out']=str(v.ROOT/'cpu/commands'/job['name']/row['arm'])
    if top['env']['SERVER_HOST']=='local':argv,_=local_server.command(spec)
    else:argv,_=node.command(spec['model'],spec['suite'],port,spec['yaml'],spec['out'],spec['tag'],spec['plugin'],dict(spec['env']))
    wanted=('--checkpoint','--denoising-steps') if row['model']=='groot' else ('--policy.config','--policy.dir')
    actual={k:argv[argv.index(k)+1] for k in wanted}
    source=Path(top['source_root'])/'runs'/top['source_arm'];expected=None
    for path in sorted(source.glob('server_*/decisions_*.jsonl')):
        for line in path.open():
            record=json.loads(line)
            if record.get('ev')=='startup':
                old=record['argv'];expected={k:old[old.index(k)+1] for k in wanted};break
        if expected is not None:break
    assert expected==actual,(job['name'],expected,actual)
    return dict(policy_cli=actual,source_startup_equal=True,interpreter=next(x for x in argv if x.endswith('/bin/python')))


def native_plans(records):
    # Fixture basenames intentionally match R15's layout. Keep their plans
    # separate from the initial campaign plans.
    groups={}
    for record in records:
        path=Path(record['root'])
        key=path.parent.name if 'cpu_future' in path.parts else 'initial'
        groups.setdefault(key,[]).append(record)
    audited=[]
    for key,group in groups.items():
        out=v.ROOT/'plans'/key
        verify.plans(group,out);audited+=read(out/'audit.json')['records']
    result=dict(PASS=True,records=audited,remote_operations=0,servers=0)
    dump(v.ROOT/'plans/audit.json',result)
    return result


def no_filter_equivalence():
    result=[]
    with patch.object(subprocess,'Popen',blocked),patch.object(socket,'create_connection',blocked):
        for cell in (c for c in v.CELLS if c.endswith('_50')):
            with v.catalog('filtered'),patch.object(plugin,'_git_head',return_value=None):
                harness=Harness(cell,v.ROOT/'cpu/equivalence_keep_all'/cell)
                try:
                    for level in v.p.LEVELS:
                        # Replay original source fits directly; no unfiltered
                        # campaign or copied control artifact is created.
                        blob=pickle.loads(Path(v.original(cell,level)['path']).read_bytes())
                        kept=np.unique(blob['method'].inner.base.lib_ep).tolist()
                        view=v.candidates(blob,kept);v.unchanged(blob,view)
                        for task,table in blob['method'].inner.base.tasks.items():
                            assert view['method'].inner.base.tasks[task] is table
                        source=pickle.loads(Path(v.old50.frozen(cell,level)['path']).read_bytes())['method']
                        x=harness.run(source,steps=8,force_guards=True)
                        y=harness.run(view['method'],steps=8,force_guards=True)
                        assert x[0]==y[0] and lifecycle(x[2])==lifecycle(y[2])
                        result.append(dict(cell=cell,level=level,keep_all_candidate_path=True,
                            serving_tables_identical=True,actions_byte_identical=True,lifecycle_identical=True,decisions=len(x[2])))
                finally:harness.rt.flush_all()
    return result


def check():
    records=[];startup=[];jobs=read(v.ROOT/'CAMPAIGNS.json')['jobs']
    for name in ('pi05','groot'):
        path=v.ROOT/f'TEST_CAMPAIGNS_{name}.json'
        if path.exists():jobs+=read(path)['jobs']
    # Exercise every future A cell with arbitrary mixed task levels, without
    # selecting outcomes or placing a definition in the actual test queue.
    for arm in v.ARMS:
        path=v.ROOT/'cpu_future'/arm
        if not path.exists():
            levels={cell:{str(t):v.p.LEVELS[t%len(v.p.LEVELS)] for t in range(10)} for cell in v.arms(arm)}
            uniform={cell:'off' for cell in levels}
            with v.catalog(arm):v.p.emit_test_a(levels,uniform,path)
        for rec in read(path/'test_a_roots.json'):
            jobs.append(dict(name=Path(rec['root']).name,root=rec['root'],cell=rec['cell'],phase='cpu-fixture',
                topology=v.topology(rec['cell'],'task_descent',True)))
    with ExitStack() as stack:
        stack.enter_context(patch.object(subprocess,'Popen',blocked))
        stack.enter_context(patch.object(socket,'create_connection',blocked))
        stack.enter_context(patch.object(plugin,'_git_head',return_value=None))
        for job in jobs:
            root=Path(job['root']);seal=read(root/'ROOT.json');records.append(seal)
            for row in read(root/'arms.json'):
                args=['--os-method',row['method'],'--os-cell',row['cell'],*row['plugin_args'],
                      '--os-kwargs',json.dumps(row['kwargs']),'--os-log-dir',str(v.ROOT/'cpu'/root.name/row['arm'])]
                opts,rest=plugin.parse_cli(args);assert not rest
                runtime=plugin.PluginRuntime(opts,row['model'])
                fit=Path(opts.os_fit_artifact)
                assert runtime.r14_fit_sha256==sha(fit)
                assert runtime.opts.kwargs==row['kwargs'] and runtime.method is not None
                top=job['topology'] if job['phase']!='dev' else job['levels'][row['r14_descent_level']]
                assert row['full_model'] and not row.get('recording') and top['env']['STAGE1_ONLY']=='0'
                identity=policy_identity(job,row,top)
                startup.append(dict(campaign=job['name'],arm=row['arm'],status='PASS',fit_sha256=sha(fit),
                    native_plugin_identity_checked=True,full_policy=True,recording=False,
                    source_launch=top['source_root']+'/runs/'+top['source_arm']+'/h100_launch.json',
                    source_launch_sha256=top['source_receipt_sha256'],topology=top,model_loaded=False,**identity))
                runtime.flush_all();del runtime;gc.collect()
            print('STARTUP_PASS',root.name,flush=True)
    # Native R15 sync planner, with all its launch/remote operations fenced.
    native_plans(records)
    artifacts=read(v.ROOT/'ARTIFACTS.json')['rows']
    for item in artifacts:
        assert sha(item['source'])==item['source_sha256'] and sha(item['path'])==item['sha256']
    equivalence=no_filter_equivalence()
    import torch
    assert not torch.cuda.is_initialized()
    dump(v.ROOT/'CPU_CHECKS.json',dict(status='PASS',startup=startup,planner=read(v.ROOT/'plans/audit.json'),
        keep_all_source_equivalence=equivalence,artifacts=len(artifacts),cuda_initialized=False,
        original_r15_runtime_manifest=sha(v.SOURCE/'RUNTIME_SOURCES.sha256'),servers_started=False,workers_started=False))
    print('R15_CPU_PASS',len(startup),'startup rows;',len(records),'native plans;',len(equivalence),'size-50 replay arms')
