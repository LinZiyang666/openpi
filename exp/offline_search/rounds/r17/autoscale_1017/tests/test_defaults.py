"""CPU-only post-apply preparation/transport tests. No remote or GPU calls."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from exp.offline_search.rounds.r17.autoscale_1017 import defaults, legacy_release
from exp.offline_search.rounds.r17.autoscale_1017.node import atomic, read


@pytest.mark.parametrize('domain,expected', [('libero',(83,104,83)),('metaworld',(332,440,332))])
def test_physical_defaults_and_explicit_overrides(domain,expected):
    assert tuple(defaults.physical_maximum(n,domain) for n in ('timan1','timan107','local')) == expected
    c = dict(domain=domain, worker_nodes={'timan1':150,'timan107':100,'timan108':40}, workers=290)
    defaults.apply(c)
    assert c['autoscale']['mode'] == 'dynamic'
    promoted={'timan1':max(150,expected[0]),'timan107':max(100,expected[1])}
    assert c['autoscale_node_maxima'] == promoted
    assert c['autoscale_maximum'] == sum(promoted.values())
    assert c['worker_nodes']['timan108'] == 40  # scientific baseline topology is not rewritten
    if domain == 'metaworld':assert c['worker_max_episodes'] == 32


def test_continuation_keeps_frozen_mode_and_sealed_preparation_is_not_changed(tmp_path):
    c = dict(cache_resume_selection=dict(source='old'), worker_nodes=dict(timan107=78))
    defaults.apply(c)
    assert c['autoscale'] is None and c['autoscale_disposition'] == 'kept_fixed'
    forced = dict(c, autoscale=dict(mode='dynamic'))
    assert defaults.apply(forced)['autoscale'] is None  # CLI defaults cannot override frozen-source pins
    atomic(tmp_path/'PREPARED.json', {})
    sealed = dict(worker_nodes=dict(timan107=64))
    assert defaults.apply(copy.deepcopy(sealed),tmp_path) == sealed


def test_scalar_count_is_an_explicit_desired_maximum():
    assert defaults.apply(dict(workers=64,worker_node='timan107'))['autoscale_node_maxima']==dict(timan107=104)
    assert defaults.apply(dict(workers=200,worker_node='timan1'))['autoscale_node_maxima']==dict(timan1=200)
    assert defaults.apply(dict(worker_nodes=dict(timan108=24)))['autoscale_node_maxima']==dict(timan107=104)


def test_r17_dead_node_refused_before_transport(monkeypatch):
    from exp.r17_integration.ops import formal_eval as f
    monkeypatch.setattr(f,'execute',lambda *a,**kw:pytest.fail('dead node reached transport'))
    with pytest.raises(ValueError,match='disabled'):f.remote('timan108',['true'])
    with pytest.raises(ValueError,match='disabled'):f.transfer(['rsync','push','source','timan108:/tmp/target'])


def dry_stage(root, c, monkeypatch, tasks=None, invoke=None):
    """Exercise native ff stage up to sealing with only external effects stubbed."""
    from exp.offline_search.rounds.r17.filter_failed import stage, worker_resilience
    from exp.r17_integration.ops import formal_eval as f, formal_runtime, cache_control, driver_release, formal_prestage, immutable_release, formal_ports
    from exp.r17_integration.ops.build_client_bundle import build_client_bundle
    source = root.parent/'source';source.mkdir(exist_ok=True)
    (source/'native.py').write_text('SCIENCE = 7\n')
    build_client_bundle(source,['native.py'],source/'workers.tar')
    # A source read after native failover may legitimately include local workers.
    # This synthetic source must carry the same receipt as a real frozen source.
    atomic(source/'LOCAL_WORKER_ORCHESTRATION.json',dict(
        schema='r17.command_worker_orchestration.v1',
        archive_sha256=immutable_release.sha(source/'workers.tar'),
        native_worker_bytes_unchanged=True,owner_authorized_any_workload=True))
    (source/'runtime.tar').write_bytes(b'frozen server model')
    atomic(source/'WORKER_DEPLOYMENT.json',{})
    c=copy.deepcopy(c)
    c.update(filter_failed_source=str(source),runtime=str(root/'native'),runtime_identity='native',replicas=1,ports=[23240],local_assets=str(root/'assets'),server_node='local')
    if c.get('cache_resume_selection'):c['cache_resume_selection']={'source':str(source)}
    if tasks is None:tasks=[dict(task_uid='frozen:eval:3:49',orig_init_state_idx=49,extra=dict(seed=2026101401))]
    atomic(root/'campaign.json',c)
    atomic(root/'EPISODE_PLAN.json',dict(tasks=tasks,expected_episodes=len(tasks)))
    before=(root/'EPISODE_PLAN.json').read_bytes()
    atomic(root/'assets/model_spec.json',dict(checkpoint_identity='model'))
    atomic(root/'assets/CACHE_STUDY.json',dict(benchmark_protocol='native'))
    (root/'runtime.tar').write_bytes(b'frozen server model')
    monkeypatch.setattr(stage,'HERE',root/'no_overrides')
    for mod,name in ((formal_ports,'prepare_admission'),(immutable_release,'install'),(formal_runtime,'attach'),(cache_control,'attach'),(driver_release,'prepare')):
        monkeypatch.setattr(mod,name,lambda *a,**kw:None)
    monkeypatch.setattr(driver_release,'planned_tasks',lambda *a:tasks)
    monkeypatch.setattr(stage,'stage_server_inputs',lambda *a:None)
    monkeypatch.setattr(stage,'stage_worker_inputs',lambda *a:None)
    monkeypatch.setattr(worker_resilience,'prepare',lambda c,r,fn:fn(c,r))
    monkeypatch.setattr(f,'remote',lambda *a,**k:pytest.fail('remote is forbidden'))
    def workers(c,r,*,payload_override):
        import shutil
        from exp.offline_search.rounds.r17.autoscale_1017.release import refresh_worker
        shutil.copyfile(payload_override,r/'workers.tar')
        if c.get('autoscale'):refresh_worker(c,r,legacy_release.REPO,{})
        for n in c.get('execution_worker_nodes',c['worker_nodes']):
            atomic(r/(n+'_MEMORY_ADMISSION.json'),dict(admitted=True))
        return {n:{} for n in c.get('execution_worker_nodes',c['worker_nodes'])}
    monkeypatch.setattr(f,'prepare_workers',workers)
    monkeypatch.setattr(formal_prestage,'seal_prepared',lambda c,r,d:atomic(r/'PREPARED.json',dict(cpu_only=True,nodes=list(d))))
    # The long-segment runtime fix (archive rewrite + release installs) is an
    # external effect here; only its invocation is checked.
    from exp.offline_search.rounds.r17.longseg_1017 import runtime_fix
    fixes=[]
    monkeypatch.setattr(runtime_fix,'stage',lambda launch,c,r:fixes.append(r) or immutable_release.sha(r/'runtime.tar'))
    (invoke or inspect.unwrap(stage._stage))(root)
    assert bool(fixes)==bool(c.get('longseg_1017'))
    assert (root/'EPISODE_PLAN.json').read_bytes()==before
    assert (root/'runtime.tar').read_bytes()==b'frozen server model'
    return read(root/'campaign.json')


QUEUED = {
 'fa_co_f_f_oc1':'R17-fixed', 'fa_pi_f_u_o':'R15','fa_pi_f_f_o':'R15',
 'fa_co_r_f_l':'R17','fa_co_f_u_l':'R17','fa_co_f_f_l':'R17',
 'fa_pi_f_u_l':'R15','fa_pi_f_f_l':'R15','fa_pi_r_f_l':'R16',
 'fa_gr_r_u_l':'R16','fa_gr_r_f_l':'R16','fa_gr_f_u_l':'R15','fa_gr_f_f_l':'R15'}


@pytest.mark.parametrize('name,pipeline',QUEUED.items())
def test_each_queued_prepare_dispatch_cpu_only(tmp_path,monkeypatch,name,pipeline):
    from exp.offline_search.rounds.r17.filter_failed_knob import formal_a_queue as q, vla_workers as v
    from exp.offline_search.closed_loop.ops.h100 import control
    from exp.r17_integration.ops import formal_eval as f
    from exp.offline_search.rounds.r17.autoscale_1017 import dryrun
    # Live queue metadata is read-only; all output paths and transitions are in tmp_path.
    unit=dryrun.unit(name)
    if unit['jobs']:
        assert all(j['pipeline']==pipeline.split('-')[0] for j in unit['jobs'])
    monkeypatch.setattr(q,'INDEX',tmp_path/'QUEUE.json')
    monkeypatch.setattr(q.f,'ROOT',tmp_path)
    monkeypatch.setattr(q,'publish',lambda:None)
    monkeypatch.setattr(q,'levels',lambda m:['k1'])
    monkeypatch.setattr(q,'evaluate',lambda *a:dict(rows=[dict(suite='libero_10',size=50,status='tune')]))
    c = dict(run_name=name,domain='libero',worker_nodes=dict(timan1=35,timan107=25),workers=60)
    if pipeline=='R17-fixed':c['cache_resume_selection']=dict(source='frozen')
    prepared=[]
    def materialized(u,*a):
        return dict(name=name,run=str(tmp_path/name),pipeline=pipeline.split('-')[0],workers=u['workers'],episodes=1,
                    topology=dict(env=dict(PORTS='23230,23231,23232',WPS='13',SERVER_HOST='h100')))
    monkeypatch.setattr(q,'r17_job',materialized)
    monkeypatch.setattr(q,'r15_jobs',lambda u,*a:[materialized(u)])
    monkeypatch.setattr(q,'r16_jobs',lambda u,*a:[materialized(u)])
    job=materialized(unit)
    unit['jobs']=[job] if unit['status']!='WAITING_OFF' else []
    atomic(q.INDEX,dict(units=[unit]))
    def remote(*a,**kw):prepared.append(('staged',a))
    monkeypatch.setattr(f,'transfer',remote)
    monkeypatch.setattr(control,'remote',remote)
    monkeypatch.setattr(v,'probe',lambda h,p:dict(usable=h=='timan107',host=h,island='/frozen/island',installed=True,admission=dict(capacity=1)))
    monkeypatch.setattr(v,'stage_r16',lambda *a:None)
    def execute(argv,**kw):
        assert argv[0]=='bash' and 'prepare_' in argv[1]
        root=tmp_path/name;root.mkdir(exist_ok=True)
        current=read(q.INDEX)['units'][0]
        job=current['jobs'][0]
        if pipeline.startswith('R17'):
            if pipeline=='R17':c['execution_worker_nodes']=job['workers']
            prepared.append(dry_stage(root,c,monkeypatch))
        elif pipeline=='R16':
            v.prepare_r16_job(job)
            prepared.append(read(root/'LEGACY_AUTOSCALE.json'))
        else:
            # The real R15 prepare entry produces the native sync invocation;
            # freeze its lower-layer release here instead of transferring assets.
            atomic(root/'arms.json',[dict(arm='same_arm')])
            cmd,env=v.r15_command(job,'prepare','timan107')
            assert cmd[-2:]==[str(root),'same_arm'] and 'sync' in cmd
            prepared.append(legacy_release.stage(root,'timan107',pipeline='R15',maximum=int(env['OPENPI_DESIRED_MAXIMUM']),base='/frozen/island'))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(q.subprocess,'run',execute)
    q.dispatch('prepare',name)
    result=prepared[-1]
    if pipeline=='R17-fixed':assert result['autoscale_disposition']=='kept_fixed'
    elif pipeline=='R17':assert result['autoscale']['mode']=='dynamic'
    else:
        assert result['mode']=='dynamic' and result['maximum']==104
        assert all('autoscale_' in result[k] for k in ('local','remote'))
    assert 'timan108' not in result.get('autoscale_node_maxima',{})


def test_supervisor_zero_workers_never_dispatches_an_episode(monkeypatch,tmp_path):
    from openpi.conductor import driver, protocol
    from exp.offline_search.rounds.r17.autoscale_1017.legacy import LegacyDriver
    obj=object.__new__(driver.ConductorDriver)
    obj._journal=SimpleNamespace(path=tmp_path/'journal.jsonl')
    obj._scheduler=SimpleNamespace(all_done=lambda:False)
    obj._autoscaler=LegacyDriver(obj)
    obj._record_pull=lambda *a:pytest.fail('supervisor is not a worker')
    obj.handle_pull=lambda *a:pytest.fail('supervisor must not receive tasks')
    frames=iter([(protocol.MSG_PULL,dict(_autoscale_supervisor=True,_autoscale=dict(node=dict(node='timan107',campaign='legacy',maximum=104,time=__import__('time').time(),active=0))))])
    def receive(*a):
        try:return next(frames)
        except StopIteration:raise ConnectionError()
    monkeypatch.setattr(protocol,'recv_message',receive)
    replies=[]
    monkeypatch.setattr(protocol,'send_message',lambda conn,kind,payload:replies.append(payload))
    obj._handle_conn(SimpleNamespace(getpeername=lambda:('127.0.0.1',1),close=lambda:None),lambda:False)
    assert replies[0]['task'] is None and replies[0]['_autoscale_control']['targets']==dict(timan107=104)


def test_legacy_supervisor_recovers_after_bad_frame():
    from exp.offline_search.rounds.r17.autoscale_1017.legacy import AgentGovernor
    from openpi.conductor.protocol import ProtocolError
    supervisor=object.__new__(AgentGovernor)
    waits=iter([False,True]); calls=[]
    supervisor.agent=SimpleNamespace(_stop=SimpleNamespace(is_set=lambda:False,wait=lambda seconds:next(waits)))
    supervisor.slots=SimpleNamespace(close=lambda:calls.append('closed'))
    def exchange():
        calls.append('exchange')
        if len(calls)==1:raise ProtocolError('truncated frame')
    supervisor.exchange=exchange
    supervisor.supervise()
    assert calls==['exchange','exchange','closed']


def test_denied_legacy_slots_do_not_resample_the_whole_node_per_slot():
    from exp.offline_search.rounds.r17.autoscale_1017.legacy import AgentGovernor
    now=[0.];samples=[]
    governor=object.__new__(AgentGovernor)
    governor.last=0;governor.gpu_retry_after=0;governor.registered={}
    governor.agent=SimpleNamespace(_handles={})
    governor.local=SimpleNamespace(clock=lambda:now[0],config=SimpleNamespace(sample_s=2),
        snapshot={},entries={},target=500,retired=set())
    def sample():
        samples.append(now[0]);governor.local.snapshot={'admit':False}
    governor.local.tick=sample
    governor.local.permit=lambda:pytest.fail('blocked slots must wait for the next node sample')
    for instant in (0.,1.,2.):
        now[0]=instant
        for _ in range(500):assert governor.spawn(None,None,None,None) is None
    assert samples==[0.,2.]


def test_legacy_overlay_imports_new_core_and_preserves_frozen_fallback(tmp_path):
    import os
    import subprocess
    import sys
    record=legacy_release.build(tmp_path)
    overlay=record['local']
    code='''
from pathlib import Path
import openpi.conductor.driver as d
import openpi.conductor.agent as a
import openpi.conductor.worker as w
from exp.offline_search.rounds.r16 import conductor as r
from exp.offline_search.rounds.r17.autoscale_1017 import legacy, defaults
assert all('/autoscale_releases/' in x.__file__ for x in (d,a,w,r,legacy))
assert '/autoscale_releases/' not in str(r.REPO)
assert defaults.physical_maximum('timan107') == 104
print('sealed core; original scientific fallback')
'''
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONPATH=overlay+':'+overlay+'/src:'+str(legacy_release.REPO)+':'+str(legacy_release.REPO/'src'))
    out=subprocess.run([sys.executable,'-c',code],env=env,cwd=tmp_path,text=True,capture_output=True,check=True)
    assert 'original scientific fallback' in out.stdout
    frozen=Path('/home/weiland/projects/openpi_r16_k1ref_server')
    if frozen.is_dir():
        # Match the queued R16 launch: its scientific checkout is cwd and the
        # current ffk orchestration package is supplied as a namespace fallback.
        env['PYTHONPATH']=overlay+':'+overlay+'/src:'+str(frozen)+':'+str(frozen/'src')+':'+str(legacy_release.REPO)+':'+str(legacy_release.REPO/'src')
        check=code+'\nfrom exp.offline_search.rounds.r17.filter_failed_knob import vla_workers\nassert Path(r.REPO)==Path('+repr(str(frozen))+')\n'
        subprocess.run([sys.executable,'-c',check],env=env,cwd=frozen,text=True,capture_output=True,check=True)


def test_legacy_plans_invariant_under_desired_count_and_endpoint_reassignment(tmp_path):
    from dataclasses import asdict
    from exp.gate_threshold_pareto.run_gtp import SweepStrategy
    from exp.offline_search.rounds.r16.conductor import R16Strategy, SEED
    from openpi.conductor.task import ServerEndpoint
    manifest=tmp_path/'pairs.json';manifest.write_text('[[3,49],[1,0]]')
    group=dict(references=None,phase='test',entries=[dict(pairs=[[3,49],[1,0]],row=dict(
        arm='sealed-arm',r14_descent_level='c10',suite='libero_10',init_pool='A',manifest=str(manifest),r16_cell='cell'))])
    reference=None
    for maximum in (1,44,104,250):
        c=defaults.apply(dict(domain='libero',worker_nodes=dict(timan107=maximum),
            worker_hard_limits={'timan107':dict(maximum=maximum,reason='CPU invariant fixture resource limit')}))
        server=ServerEndpoint('server',23150)
        r15=SweepStrategy('libero_10',{'sealed-arm':'sealed.yaml'},50)._episodes('sealed-arm',server)
        graph=R16Strategy(group).plan(['sealed-arm'],{'sealed-arm':server})
        r16=[t for stage in graph.stages.values() for t in stage.episodes]
        identity=[(t.task_uid,t.task_id,t.episode_idx,t.orig_init_state_idx,SEED) for t in [*r15,*r16]]
        if reference is None:reference=identity
        assert identity==reference and c['autoscale_maximum']==maximum


def test_r15_rpc_uses_new_helper_and_preserves_scoped_lock(tmp_path,monkeypatch):
    from exp.offline_search.closed_loop.ops.h100 import node
    from exp.offline_search.rounds.r17.filter_failed_knob.vla_shared import configure_node
    script=legacy_release.REPO/'exp/offline_search/closed_loop/ops/h100/run_arm.sh'
    records=[]
    monkeypatch.setattr(node,'atomic',lambda p,v:records.append(v))
    monkeypatch.setattr(node,'wait_fleet',node.wait_fleet)
    monkeypatch.setattr(node,'island_workers',node.island_workers)
    configure_node(node,'/new/scoped.lock','new-campaign')
    node.atomic(tmp_path/'job',dict(role='driver',env=dict(OPENPI_AUTOSCALE_RUN_ARM=str(script)),cmd=['bash',str(node.ISLAND/'os_cl/run_arm.sh'),'suite','arm']))
    job=records[-1]
    assert job['cmd'][:2]==['bash','-c']
    assert 'OPENPI_WORKER_MEMORY_GUARD=1' in job['cmd'][2]
    assert 'OPENPI_AUTOSCALE_OVERLAY' in job['cmd'][2]
    assert 'WORKER_CAP=64' not in job['cmd'][2]
    assert job['env']['FFK_WORKER_LOCK']=='/new/scoped.lock'
    monkeypatch.setenv('FFK_WORKER_FLEET_ROOT',str(tmp_path))
    cmd=legacy_release.rpc_command('/scratch/island/os_cl/node.py','start','spec.json',dict(role='driver',env=dict(OPENPI_AUTOSCALE_OVERLAY='/new/overlay')))
    assert cmd[:4]==['/usr/bin/python3','-B','-S','-c'] and 'configure_node' in cmd[4]
    compile(cmd[4],'<new RPC helper>','exec')
