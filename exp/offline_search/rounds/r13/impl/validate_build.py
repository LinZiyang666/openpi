"""CPU exact solver, complete asset plans and frozen input verification."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time

from exp.offline_search.closed_loop import devset
from exp.offline_search.closed_loop.ops.h100 import assets, control
from exp.offline_search.rounds.r11.knob.calibration import tables
from .prepare import HERE, ASTRA, R13, write_json


def solver():
    rows=[]
    for cell in dict.fromkeys(r['cell'] for r in json.loads((ASTRA/'periodic_fits.json').read_text())):
        model,suite,size=cell.split('_')
        _,_,_,schedule=tables(model,suite,int(size))
        for c in json.loads((ASTRA/'periodic_fits.json').read_text()):
            if c['cell']!=cell:continue
            setting,pred,status=schedule.solve('periodic',c['target'])
            assert setting==c['unrounded_setting'] and round(setting,4)==c['setting']
            assert schedule.ir('periodic',round(setting,4))==c['pred_ir_library']
            assert status==c['status']
            rows.append(dict(cell=cell,target=c['target'],setting=c['setting'],solver_exact=True))
        print('SOLVER',cell,flush=True)
    write_json(HERE/'periodic_solver.json',dict(PASS=True,count=len(rows),records=rows))


def plans():
    # One immutable asset's real SHA is read once while its stat tuple remains
    # unchanged. This only avoids repeated hashing, never skips verification.
    cache={};original=devset.sha
    def digest(path):
        p=Path(path);st=p.stat();k=(str(p.resolve()),st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
        if k not in cache:cache[k]=original(p)
        return cache[k]
    devset.sha=assets.sha=digest
    rows=[]
    records=json.loads((HERE/'pilot_roots.json').read_text())+json.loads((HERE/'test_a_roots.json').read_text())
    for r in records:
        root=Path(r['root']);local=r['intended_host']=='local_4090'
        os.environ['SERVER_HOST']='local' if local else 'h100'
        os.environ['WORKER_HOST']='timan107' if local else 'timan108'
        for env in ('OSCL_MANIFEST','OSCL_EPISODES','OSCL_TASKS','OSCL_INIT_POOL',
                    'OSCL_EXECUTION_AUDIT','OSCL_RECORD_LOOKS','OSCL_ROLLOUT_SEED'):
            os.environ.pop(env,None)
        work=HERE/'plans'/root.name
        plan=assets.build_plan(root,r['arms'],work,local=local)
        source_rows=json.loads((root/'arms.json').read_text())
        for source,ready in zip(source_rows,plan['arms'],strict=True):
            assert ready['arm']==source['arm']
            assert control.execution_env(ready)['OSCL_ROLLOUT_SEED']==str(source['rollout_replicate_seed'])
            assert '--os-rollout-seed' in ready['plugin_args']
            artifact=assets.flag(source['plugin_args'],'--os-fit-artifact')
            fit=next(f for f in plan['files'] if f['original']==artifact)
            assert fit['source_sha256']==digest(artifact)
            if r['init_pool']=='B':
                contract=next(f for f in plan['files'] if source['arm']+': B recording contract' in f['reasons'])
                assert contract['sha256']==assets.flag(ready['plugin_args'],'--os-look-contract-sha256')
                assert '--os-record-look-raw-distance' in ready['plugin_args']
            else:
                assert not any(x.startswith('--os-record') for x in ready['plugin_args'])
            # The real serving spec and argv builder must preserve the new
            # options across local/remote relocation. No process is launched.
            spec=control.server_spec(root,ready,23120 if local else 23220)
            from exp.offline_search.closed_loop.ops.h100 import local_server, node
            if local:
                spec.update(run_root=str(work/'argv'),out=str(work/'argv'/source['arm']))
                argv,env=local_server.command(spec)
                remaining=argv[5:]
            else:
                argv,env=node.command(spec['model'],spec['suite'],spec['port'],spec['yaml'],
                                      spec['out'],spec['tag'],spec['plugin'],dict(spec['env']))
                remaining=argv[2:]
            opts,rest=__import__('exp.offline_search.closed_loop.plugin',fromlist=['parse_cli']).parse_cli(remaining)
            assert opts.os_rollout_seed==source['rollout_replicate_seed'] and env['STAGE1_ONLY']=='0'
        rows.append(dict(root=str(root),fleet='local/timan107' if local else 'h100/timan108',
                         arms=len(source_rows),files=len(plan['files']),bytes=plan['bytes'],
                         plan_sha256=digest(work/'plan.json'),recording_contracts=len(plan.get('dev_contracts',{})),
                         relocation=True,no_transport=True,no_launch=True))
        print('PLAN',root.name,len(source_rows),flush=True)
    write_json(HERE/'plans.json',dict(PASS=True,roots=rows,arms=sum(r['arms'] for r in rows),
                                    unique_hashed_assets=len(cache),remote_operations=0))


def runtime():
    # Run separately, in an otherwise clean interpreter, before importing any
    # calibration tool. Only unpickle; no serving/fitting request is made.
    catalog=json.loads((HERE/'fits.json').read_text());files=set()
    for rec in catalog:
        pickle.loads(Path(rec['path']).read_bytes())
    forbidden=[n for n in sys.modules if n.startswith(('exp.offline_search.rounds.r13.astra',
                                                     'exp.offline_search.rounds.r11.astra',
                                                     'exp.offline_search.rounds.r11.opus',
                                                     'exp.offline_search.rounds.r12.opus'))]
    # validate_build imports R11.calibration for its solver entry point only;
    # the runtime audit is provided by runtime_audit.py instead.
    assert not forbidden


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('solver','plans'));a=ap.parse_args()
    solver() if a.action=='solver' else plans()
