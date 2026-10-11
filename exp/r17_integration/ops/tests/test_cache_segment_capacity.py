"""Owner-authorized cache capacity, episode-boundary drain and provenance."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import pytest

from exp.r17_integration.ops import cache_drain,cache_resume,worker_topology,local_workers
from exp.r17_integration.ops.formal_process import alive,ticks,write_json


def campaign():
    return dict(model='dp',domain='libero',cache_study='/sealed/STUDY.json',cache_control='/sealed/control',
        workers=32,worker_nodes={'timan107':32},replicas=2,public_port=23400,driver_port=23419,
        backend_ports=[23401,23402],ports=[23400,23401,23402,23419],server_node='local',
        checkpoint='/sealed/model',dp_batching=dict(batch=8,wait_ms=5),mps=True,worker_memory_mib=4096,
        local_worker_memory_mib=4096)


def test_cache_resize_preserves_canonical_native_batch_and_yaml(tmp_path,monkeypatch):
    from exp.r17_integration.ops import cache_control
    monkeypatch.setattr(cache_control,'attach',lambda *a:None)
    c=campaign();old=copy.deepcopy(c)
    worker_topology.redistribute(c,{'local':60},tmp_path,replicas=4,local_vram_mib=576,local_server_growth_mib=8192)
    assert c['workers']==32 and c['worker_nodes']=={'timan107':32}
    assert c['execution_workers']==60 and c['execution_worker_nodes']=={'local':60}
    assert c['replicas']==4 and c['backend_ports']==[23401,23402,23403,23404]
    assert c['checkpoint']==old['checkpoint'] and c['dp_batching']==old['dp_batching'] and c['mps']==old['mps']
    assert c['cache_worker_orchestration']['miss_batch_composition_may_change']
    w=local_workers.route(dict(c,worker_node='local'),c)
    assert w['worker_memory_mib']==4096 and w['worker_vram_mib']==576 and w['worker_server_vram_headroom_mib']==8192
    assert w['policy_endpoint']=='127.0.0.1:23400'


@pytest.mark.parametrize('field,value',[('cache_study',None),('domain','metaworld')])
def test_capacity_exception_cannot_reinterpret_legacy(tmp_path,field,value):
    c=campaign();c[field]=value
    with pytest.raises(ValueError,match='restricted'):worker_topology.redistribute(c,{'local':60},tmp_path)


@pytest.mark.parametrize('kwargs',[dict(replicas=5),dict(replicas=4,local_vram_mib=512),
    dict(replicas=4,local_server_growth_mib=4096)])
def test_capacity_invalid_resource_override_refuses(tmp_path,kwargs):
    with pytest.raises(ValueError):worker_topology.redistribute(campaign(),{'local':60},tmp_path,**kwargs)


def test_per_segment_resources_include_old_and_new_topologies(tmp_path):
    old=tmp_path/'old';old.mkdir();c=campaign();write_json(old/'campaign.json',c)
    new=dict(c,cache_resume_selection=dict(source=str(old)),execution_worker_nodes={'local':36},execution_workers=36,
        replicas=4,local_worker_vram_mib=576)
    resources=cache_resume.segment_resources(new,tmp_path/'new')
    assert [(r['worker_nodes'],r['workers'],r['replicas']) for r in resources]==[({'timan107':32},32,2),({'local':36},36,4)]
    assert all(r['canonical_workers']==32 and r['gated_miss_batch_composition_may_change'] for r in resources)
    assert cache_resume.segment_lifetimes(new,tmp_path/'new',{str(old):{1,2},str(tmp_path/'new'):{3,4,5,6}})
    assert not cache_resume.segment_lifetimes(new,tmp_path/'new',{str(old):{1,2,3}})
    assert not cache_resume.segment_lifetimes(new,tmp_path/'new',{'unknown_segment':{1}})


def test_real_control_pin_accepts_resized_cache_and_rejects_sampler_change(tmp_path,monkeypatch):
    from exp.r17_integration.ops import cache_control,cache_study,cache_lane
    source=cache_lane.RUN_ROOT/cache_study.CONTROLS['dp_libero']
    c=json.loads((source/'campaign.json').read_text())
    c.update(cache_study='/sealed/STUDY.json',worker_memory_mib=4096)
    worker_topology.redistribute(c,{'local':60},tmp_path,replicas=4,local_vram_mib=576,local_server_growth_mib=8192)
    proof=cache_control.control_proof(source,source,c)
    assert proof['worker_resource_changes']['replicas']['campaign']==4
    assert c['dp_batching']==json.loads((source/'campaign.json').read_text())['dp_batching']
    changed=copy.deepcopy(c);changed['dp_batching']['max_batch']=7
    with pytest.raises(ValueError,match='dp_batching'):cache_control.control_proof(source,source,changed)


def test_resume_job_forwards_explicit_capacity_but_defaults_stay_identical(tmp_path,monkeypatch):
    from exp.r17_integration.ops import cache_lane
    calls=[]
    monkeypatch.setattr(cache_resume,'prepare',lambda *a,**kw:calls.append((a,kw)) or tmp_path/'new')
    job=dict(run=str(tmp_path/'old'),model='dp_libero')
    cache_lane.resume_job(job,{'local':60},4)
    assert calls[-1][1]==dict(refresh_workers=True,worker_counts={'local':60},replicas=4,
        local_vram_mib=576,local_server_growth_mib=8192)
    cache_lane.resume_job(job)
    assert calls[-1][1]==dict(refresh_workers=True,worker_counts=None)


@pytest.mark.parametrize('bad',[None,'batch','sampler','control','authorization'])
def test_real_native_dp_admission_bridge_only_waives_replica_count(tmp_path,monkeypatch,bad):
    from types import SimpleNamespace as NS
    from exp.r17_integration.models.dp import formal_batch_server as native
    from exp.r17_integration.ops.formal_server import dp_capacity_serve
    from exp.r17_integration.ops import cache_control
    monkeypatch.setattr(cache_control,'attach',lambda *a:None)
    c=campaign();worker_topology.redistribute(c,{'local':33},tmp_path,replicas=4,
        local_vram_mib=576,local_server_growth_mib=8192)
    c['cache_control_proof']=dict(worker_resource_changes=dict(replicas=dict(control=2,campaign=4)))
    cache=NS(enabled=True,adapter_binding={'num_steps':100},checkpoints=[],recipe=NS(enabled=True,miss_mode='stall_only'))
    sampler=NS(family='ddpm',num_steps=100)
    batch=NS(max_batch=8,batch_wait_ms=5)
    provider=NS(sampler_for_steps=lambda k:sampler,config_validator=lambda config:None)
    original=native.cache_study_admission
    calls=[]
    def serve(actual,*a,**kw):
        calls.append(actual['replicas']);assert actual is c
        return native.cache_study_admission(actual,cache,batch,provider)
    monkeypatch.setattr(native,'serve',serve)
    if bad=='batch':batch.max_batch=7
    if bad=='sampler':sampler.num_steps=10
    if bad=='control':c['cache_control_proof']['worker_resource_changes']['replicas']['control']=3
    if bad=='authorization':c.pop('cache_worker_orchestration')
    if bad:
        with pytest.raises(ValueError):dp_capacity_serve(native,c,23401,tmp_path,cuda_graph=True)
    else:
        assert dp_capacity_serve(native,c,23401,tmp_path,cuda_graph=True) is True
        assert calls==[4] and c['replicas']==4
        assert json.loads((tmp_path/'CACHE_CAPACITY_ADMISSION.json').read_text())['actual_replicas']==4
    assert native.cache_study_admission is original


def test_boundary_audit_refuses_unreturned_dispatch(tmp_path):
    (tmp_path/'driver').mkdir()
    write_json(tmp_path/'driver/dispatch.jsonl',dict(task_uid='a',attempt=1))
    (tmp_path/'driver/dispatch.jsonl').write_text(json.dumps(dict(task_uid='a',attempt=1))+'\n')
    (tmp_path/'driver/attempts.jsonl').write_text('')
    assert cache_drain.outstanding(tmp_path)==[('a',1)]
    (tmp_path/'driver/attempts.jsonl').write_text(json.dumps(dict(task_uid='a',attempt=1,accepted=True))+'\n')
    assert cache_drain.outstanding(tmp_path)==[]


def test_real_cpu_fleet_pause_finishes_children_without_restart_or_worker_signal(tmp_path):
    name='p6_cpu_drain_'+uuid.uuid4().hex[:12]
    stage=Path('/tmp/r17_formal')/name;stage.mkdir()
    config=stage/'launch_campaign.json';write_json(config,dict(cache_study='sealed',worker_max_episodes=4))
    mod=tmp_path/'exp/r17_integration/ops';mod.mkdir(parents=True)
    for p in (tmp_path/'exp',tmp_path/'exp/r17_integration',mod):(p/'__init__.py').write_text('')
    ready=tmp_path/'ready.json';finished=tmp_path/'finished'
    (mod/'formal_workers.py').write_text('''import os,sys,json,signal,subprocess,time
stop=False
def end(*a):
 global stop
 stop=True
signal.signal(signal.SIGTERM,end)
child=subprocess.Popen([sys.executable,'-c',"import time; time.sleep(.5)",'--worker-id',sys.argv[-1].split('/')[-2]+'_cpu_0','--max-episodes','4'])
open(os.environ['CPU_READY'],'w').write(json.dumps(dict(pid=child.pid)))
while not stop:time.sleep(.02)
child.wait()
open(os.environ['CPU_FINISHED'],'w').write(str(child.returncode))
''')
    argv=[sys.executable,'-m','exp.r17_integration.ops.formal_workers','--campaign',str(config)]
    child=subprocess.Popen(argv,cwd=tmp_path,env=dict(os.environ,PYTHONPATH=str(tmp_path),CUDA_VISIBLE_DEVICES='',CPU_READY=str(ready),CPU_FINISHED=str(finished)))
    proof=dict(campaign=name,entry=dict(pid=child.pid,start_ticks=ticks(child.pid),argv=argv))
    try:
        until=time.monotonic()+5
        while not ready.exists() and time.monotonic()<until:time.sleep(.01)
        assert ready.exists()
        frozen=cache_drain.fleet_action(proof,'freeze');assert len(frozen['active_children'])==1
        with pytest.raises(ValueError,match='active episodes'):cache_drain.fleet_action(proof,'retire')
        time.sleep(.7)
        status=cache_drain.fleet_action(proof,'status');assert status['episode_boundary']
        assert alive(proof['entry'])
        cache_drain.fleet_action(proof,'retire');assert child.wait(timeout=5)==0
        assert finished.read_text()=='0'
        with pytest.raises(ValueError,match='birth/UID'):cache_drain.fleet_action(proof,'freeze')
    finally:
        if child.poll() is None:
            os.kill(child.pid,signal.SIGTERM);os.kill(child.pid,signal.SIGCONT);child.wait(timeout=5)
        config.unlink();stage.rmdir()
