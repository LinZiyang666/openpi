"""Fixed-count adapters to native formal stage / launch_prepared, never a chain editor.

No action at import. `check` is local CPU only. `prepare` stages CPU payloads;
only an explicit coordinator `launch` may start servers, renderers or drivers.
"""
import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import inspect
import json
import os
import runpy
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from pilot_campaigns import W, REPO, read, write, sha, scientific, planner
from artifacts import digest
sys.path.insert(0,str(REPO))

def harness_pins():
    return {n:sha(W/n) for n in ('pilot_formal.py','pilot_campaigns.py','pilot_trace.py','pilot_readmap.py','client_trace.py')}

def fixed(c):
    p=c['xgpu_pilot']
    if c.get('autoscale') or p['autoscale'] is not False:raise ValueError('autoscale must be OFF')
    for key in ('worker_nodes','execution_worker_nodes'):
        if c[key]!=p['fixed_worker_nodes']:raise ValueError('fixed renderer placement changed')
    if set(c['execution_worker_nodes'])-{'timan1','timan107'}:raise ValueError('forbidden worker node')
    if c['execution_workers']!=p['native_worker_count'] or c['workers']!=p['native_worker_count']:
        raise ValueError('native worker total changed')
    if c['worker_egl_devices_by_node']['timan1']!=[2]:raise ValueError('GPU2 only')

def check(root, plan=True):
    root=Path(root).resolve();c=read(root/'campaign.json');fixed(c)
    if root.parent!=W or len(root.name)!=2 or not (root/'XGPU_DRAFT.json').exists():
        raise ValueError('only study-owned short campaign directories')
    if sha(root/'PLAN.json')!=c['xgpu_pilot']['canonical_plan_sha256']:
        raise ValueError('paired PLAN changed')
    if plan:
        tasks=planner(root/'campaign.json',root/'PLANNED_TASKS.json')
        original=read(root/'PLAN.json')['tasks']
        # Current native OFF planner preserves longseg task_namespace. Older
        # preregistration used recalibrated recipe IDs. Only the two namespace
        # fields may differ; all observations/inits/seeds/protocols stay exact.
        def population(rows):
            return [{k:v for k,v in t.items() if k not in ('task_uid','yaml_id')}
                    for t in scientific(rows)]
        if len(tasks)!=500 or population(tasks)!=population(original):
            raise ValueError('formal planner differs from paired preregistration')
        changed=scientific(tasks)!=scientific(original)
        if changed and not any(g.get('task_namespace') for g in read(root/'assets/CACHE_STUDY.json')['groups']):
            raise ValueError('unexpected task namespace change')
        write(root/'TASK_NAMESPACE_MAP.json',dict(changed=changed,reason='native OFF planner honours sealed longseg task_namespace',
            rows=[dict(preregistered=a['task_uid'],formal=b['task_uid']) for a,b in zip(original,tasks)]))
        ep=read(root/'EPISODE_PLAN.json')
        if ep['tasks']!=tasks and (root/'PREPARED.json').exists():raise ValueError('prepared EPISODE_PLAN changed')
        ep['tasks']=tasks;ep['endpoint']=c['endpoint'];write(root/'EPISODE_PLAN.json',ep)
        paired=read(root/'PLAN.json');parent=paired.pop('identity')
        paired['parent_plan_identity']=parent
        paired['tasks']=[dict(t,server_host=a['server_host'],server_port=a['server_port']) for t,a in zip(tasks,original)]
        paired['identity']=digest(paired)
        if (root/'PREPARED.json').exists() and read(root/'PAIRED_PLAN.json')!=paired:
            raise ValueError('prepared paired analysis plan changed')
        write(root/'PAIRED_PLAN.json',paired)
    paths=runpy.run_path(c['runtime']+'/exp/r17_integration/models/fastwam/ops/mps.py')['paths']
    _,pipe,_=paths(Path(c['server_run']).parent,'GPU-'+'0'*36)
    if len(os.fsencode(pipe))>85:raise ValueError('frozen MPS pipe exceeds 85 bytes')
    return dict(status='PASS',run=c['run_name'],episodes=500,workers=c['execution_workers'],
                nodes=c['execution_worker_nodes'],autoscale=False,mps_pipe_bytes=len(os.fsencode(pipe)))

def sealed(root):
    proof=read(root/'PREPARED.json')
    if proof.get('status')!='PASS' or proof.get('gpu_bound') or proof.get('dispatch_started'):
        raise ValueError('invalid native prepare seal')
    c=read(root/'campaign.json');deployment=read(root/'WORKER_DEPLOYMENT.json')
    if set(deployment)!=set(c['execution_worker_nodes']):raise ValueError('prepared worker nodes differ')
    for node,item in deployment.items():
        if item['env'].get('XGPU_TRACE_ROOT')!=item['worker_run']+'/xgpu_trace':raise ValueError('missing worker trace')

def worker_rewrite(value,node,trace_source):
    """First preserve native GPU2/share wrapping; instrument its final child factory."""
    from exp.offline_search.rounds.r17.filter_failed.launch import worker_command, WORKER
    from pilot_trace import fleet_prefix
    if isinstance(value,list) and WORKER in value:
        index=value.index(WORKER)
        if index and value[index-1]=='-m':
            result=worker_command(value,node)
            pos=result.index('-c')
            # GPU2's outer -S code execs the sealed island program in a new
            # interpreter. Inject into that program, after namespace isolation.
            if node=='timan1':
                from exp.offline_search.rounds.r17.filter_failed import timan1_gpu2 as t
                source=Path(t.__file__).read_text()
                marker='        f.main()'
                if source.count(marker)!=1:raise ValueError('GPU2 bootstrap changed')
                source=source.replace(marker,'        exec('+repr(fleet_prefix(trace_source))+', globals())\n'+marker)
                result[pos+1]='s='+repr(source)+';g={"SOURCE":s};exec(s,g);g["namespace"]('+repr(value[0])+','+repr([WORKER,*value[index+1:]])+',"fleet")'
            else:result[pos+1]=fleet_prefix(trace_source)+result[pos+1]
            return result
    if isinstance(value,list):return [worker_rewrite(x,node,trace_source) for x in value]
    if isinstance(value,dict):
        out={k:worker_rewrite(v,node,trace_source) for k,v in value.items()}
        if 'argv' in out and 'env' in out and out.get('log','').endswith('/workers.log'):
            out['env']['XGPU_TRACE_ROOT']=out['env'].get('XGPU_TRACE_ROOT') or out.get('cwd','')+'/xgpu_trace'
        return out
    if isinstance(value,str) and value.startswith(('{','[')) and WORKER in value:
        return json.dumps(worker_rewrite(json.loads(value),node,trace_source))
    return value

def server_code(c,root):
    p=c['xgpu_pilot'];source=(root/'assets/pilot_readmap.py').read_text()
    code='import runpy\n'
    if p['read_aliases']:
        code+='scope={};exec('+repr(source)+',scope);_xgpu_aliases=scope["install"]('+repr(p['read_aliases'])+')\n'
    bootstrap=root/'assets/ffk_bootstrap.py'
    if bootstrap.exists():code+='exec(compile('+repr(bootstrap.read_text())+',"sealed_ffk_bootstrap","exec"),{"__name__":"__main__"})\n'
    else:code+='runpy.run_module("exp.r17_integration.ops.formal_server",run_name="__main__",alter_sys=True)\n'
    return code

def stage_server(launch,c,root):
    if c['server_node']!='h100':return
    from exp.r17_integration.ops.sync_back import snapshot_run
    archive=root/'startup_assets.tar';snapshot_run(str(root/'assets'),str(archive))
    stage='/tmp/r17_formal/'+c['run_name']+'/server'
    launch.remote('h100',['mkdir','-p',stage],replay_safe=True)
    launch.transfer(['rsync','push',str(archive),'h100:'+stage+'/startup_assets.tar'])
    code=('from exp.r17_integration.ops.sync_back import install_snapshot;'
          'install_snapshot('+repr(stage+'/startup_assets.tar')+','+repr(c['assets'])+',expected_sha256='+repr(sha(archive))+')')
    launch.remote('h100',['env','PYTHONPATH='+launch.pythonpath(c['runtime']),launch.host_python(c),'-c',code],seconds=600,replay_safe=True)
    # Exact existing private cache inputs staged through the authorised 23177
    # read-only module; never install over canonical/active formal directories.
    if c['xgpu_pilot']['read_aliases']:
        seal=read(Path(c['xgpu_pilot']['bundle'])/'CACHE_FILES.json')
        pins={c['xgpu_pilot']['read_aliases'][k]:v['sha256'] for k,v in seal['files'].items()}
        code=('from pathlib import Path;'
              'assert all(Path(p).is_file() for p in '+repr(list(pins))+'),"run stage_h100.sh first: missing private cache inputs"')
        launch.remote('h100',['python3','-c',code],seconds=600,replay_safe=True)
    c['server_campaign']=stage+'/campaign.json';write(root/'campaign.json',c)
    for name in ('campaign.json','launch_campaign.json'):
        launch.transfer(['rsync','push',str(root/'campaign.json'),'h100:'+stage+'/'+name])

@contextmanager
def context(root):
    """OFF is process-local; never write the owner's global ENABLED switch."""
    from exp.r17_integration.ops import formal_eval as f,driver_release as dr,direct_transfer
    from exp.r17_integration.ops import immutable_release
    from exp.offline_search.rounds.r17.autoscale_1017 import defaults,switch
    from exp.offline_search.rounds.r17.filter_failed import stage,worker_resilience as wr,timan1_gpu2
    c=read(root/'campaign.json');fixed(c)
    native_prepare=f.prepare_workers;native_driver=dr.prepare;native_install=dr.install
    def reuse_runtime(archive,path,archive_sha,identity):
        # All five model releases already exist here. Do not create a lock or
        # install anything in the shared frozen release tree.
        p=Path(path);manifest=read(p/'RUNTIME_RELEASE.json')
        return dict(present=True,installed=str(p),reused=True,**manifest)
    def prepare_workers(c,d,**kw):
        fixed(c);result=native_prepare(c,d,**kw)
        for node,item in result.items():
            item['env'].update(XGPU_TRACE_ROOT=item['worker_run']+'/xgpu_trace',XGPU_RECORD_LIMIT='0')
        write(Path(d)/'WORKER_DEPLOYMENT.json',result);fixed(c);return result
    def driver_prepare(c,d,checkout):
        aliases={}
        def install(archive,path,*args):
            target=W/'pilot_driver_releases'/Path(path).name
            aliases[str(path)]=str(target);return native_install(archive,target,*args)
        with patch.object(dr,'install',install):native_driver(c,d,checkout)
        c['driver_runtime']=aliases[c['driver_runtime']]
        c['orchestration']['driver_runtime']=c['driver_runtime'];write(Path(d)/'ORCHESTRATION.json',c['orchestration'])
    with ExitStack() as stack:
        stack.enter_context(patch.object(switch,'enabled',lambda:False))
        stack.enter_context(patch.object(defaults,'enabled',lambda:False))
        stack.enter_context(patch.object(f,'worker_resilience',None,create=True))
        stack.enter_context(patch.object(wr,'prepare',lambda c,d,native:native(c,d)))
        stack.enter_context(patch.object(f,'prepare_workers',prepare_workers))
        stack.enter_context(patch.object(dr,'prepare',driver_prepare))
        stack.enter_context(patch.object(immutable_release,'install',reuse_runtime))
        stack.enter_context(patch.object(stage,'stage_server_inputs',stage_server))
        stack.enter_context(patch.object(direct_transfer,'BWLIMIT_KIBPS',16384))
        stack.enter_context(timan1_gpu2.coordinator())
        yield

def renderer_admission(root):
    """Read-only slot check before claiming a launch or starting any server."""
    from exp.r17_integration.ops import formal_eval as f
    from exp.offline_search.rounds.r17.filter_failed import worker_resilience
    c=read(root/'campaign.json');result={}
    for node,item in read(root/'WORKER_DEPLOYMENT.json').items():
        worker=read(root/(node+'.json'))
        if worker['node_workers']!=c['execution_worker_nodes'][node]:raise ValueError('renderer count changed')
        if worker.get('cache_study') and node!='timan1':
            worker['worker_memory_mib']=max(4096,worker.get('worker_memory_mib',4096))
        prefix=worker_resilience.worker_patch() if node=='timan107' else ''
        code=prefix+'\nfrom exp.r17_integration.ops import worker_gpu as g,worker_memory as m\nimport json\nc='+repr(worker)+'\nr='+repr(item['worker_run'])+'\nprint(json.dumps(dict(gpu=g.admission(c,r),ram=m.admission(c,r))))'
        env=['env','PYTHONPATH='+f.pythonpath(item['payload'])]
        report=json.loads(f.remote(node,[*env,item['python'],'-c',code],seconds=120,replay_safe=True))
        result[node]=report
        write(root/'XGPU_RENDERER_ADMISSION.json',result)
        if not all(report[k]['admitted'] for k in ('gpu','ram')):
            raise RuntimeError(node+': fixed renderer slots unavailable; no launch claimed')
    return result

def prepare(root):
    check(root)
    from exp.offline_search.rounds.r17.filter_failed import stage
    from exp.r17_integration.ops.formal_prestage import preparation_lock
    with context(root),preparation_lock(root):
        # Same stage body as queue scripts; omit adaptive placement decorators.
        inspect.unwrap(stage._stage)(root)
    receipt=check(root);receipt['harness_files']=harness_pins()
    proof=read(root/'PREPARED.json')
    proof['files'].update({n:sha(root/n) for n in ('PLAN.json','PAIRED_PLAN.json','TASK_NAMESPACE_MAP.json')})
    write(root/'PREPARED.json',proof);write(root/'XGPU_PREPARED_CHECK.json',receipt)

def launch(root):
    sealed(root)
    if read(root/'XGPU_PREPARED_CHECK.json')['harness_files']!=harness_pins():
        raise ValueError('pilot launcher changed since prepare; prepare again before launch')
    check(root)
    from exp.r17_integration.ops import formal_eval as f,cache_control
    from exp.r17_integration.ops.formal_prestage import launch_prepared
    from exp.offline_search.rounds.r17.filter_failed_knob.launch import rewrite
    c=read(root/'campaign.json');boot=server_code(c,root)
    evidence_file=root/'assets/ffk_evidence.py'
    body=evidence_file.read_text() if evidence_file.exists() else ''
    evidence=('ffk_cpu_evidence={};exec(compile('+repr(body)+',"ffk_cpu_evidence","exec"),ffk_cpu_evidence);ffk_cpu_evidence["install"]()') if body else ''
    trace=(root/'assets/client_trace.py').read_text()
    popen,remote=f.subprocess.Popen,f.remote
    def new_popen(argv,*a,**k):return popen(rewrite(argv,boot,evidence),*a,**k)
    def new_remote(node,argv,*a,**k):
        if node not in ('h100','timan1','timan107'):raise ValueError('node outside pilot allowlist')
        value=worker_rewrite(rewrite(argv,boot,evidence),node,trace)
        if node=='timan1':
            from exp.offline_search.rounds.r17.filter_failed.timan1_gpu2 import auxiliary
            value=auxiliary(value)
        return remote(node,value,*a,**k)
    # Serialize this study's shared renderer fleets across the two model chains.
    # Does not reserve, stop or resize any other formal fleet.
    with (W/'pilot_renderer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with context(root):
          renderer_admission(root)
          with patch.object(f.subprocess,'Popen',new_popen),patch.object(f,'remote',new_remote),patch.object(cache_control,'refresh_prepared_workers',lambda c,d,x:(c,x)):
            if body:
                scope={};exec(compile(body,'ffk_cpu_evidence','exec'),scope);scope['install']()
            return launch_prepared(root)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['check','prepare','launch']);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();root=a.run.resolve()
    if a.mode=='check':print(json.dumps(check(root)))
    elif a.mode=='prepare':prepare(root)
    else:raise SystemExit(launch(root))
