"""Resource growth must preserve scientific pins and refuse unverified T1."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
from exp.r17_integration.ops import cache_capacity as capacity,timan1_workers as t1,worker_gpu,worker_memory


def campaign(model='fastwam',nodes=None):
    nodes=nodes or {'timan107':45,'timan1':40}
    request='Owner: more workers; DP local first; H100 timan10745+timan140 GPU3 only.'
    c=dict(model=model,domain='libero',cache_study='/frozen/STUDY.json',cache_control='/closed/control',
        workers={'fastwam':48,'dp':32,'cosmos':8}[model],execution_workers=sum(nodes.values()),
        execution_worker_nodes=nodes,replicas=4 if model=='dp' else 2)
    r=dict(schema='r17.cache_capacity_authorization.v1',owner_authorized=True,study=c['cache_study'],
        model=model,control=c['cache_control'],canonical_workers=c['workers'],worker_nodes=nodes,
        execution_workers=c['execution_workers'],replicas=c['replicas'],request=request,
        request_sha256=hashlib.sha256(request.encode()).hexdigest())
    r['digest']=capacity.digest(r);c['cache_capacity_authorization']=r
    return c


def test_t1_40_requires_the_frozen_capacity_exception_and_retains_gpu3_guard():
    c=campaign();lease=t1.authorize(c)
    assert lease['workers']==40 and lease['worker_memory_mib']==6144
    assert lease['uuid']==t1.UUID and lease['device']==3 and lease['excluded_devices']==[0,1,2]
    bad=copy.deepcopy(c);bad.pop('cache_capacity_authorization')
    with pytest.raises(ValueError):t1.authorize(bad)
    with pytest.raises(ValueError):capacity.validate_authorization(campaign(nodes={'timan1':41}))


@pytest.mark.parametrize('key,value',[('execution_workers',84),('workers',85),('replicas',3),
    ('execution_worker_nodes',{'local':85}),('cache_study','different'),('cache_control','other'),('domain','metaworld')])
def test_changed_capacity_binding_refuses(key,value):
    c=campaign();c[key]=value
    with pytest.raises(ValueError):capacity.validate_authorization(c)


def test_h100_never_uses_local_even_with_rehashed_record():
    with pytest.raises(ValueError):capacity.validate_authorization(campaign(nodes={'local':16,'timan107':45}))
    with pytest.raises(ValueError):capacity.validate_authorization(campaign('dp',{'local':32,'timan107':1}))


def test_request_and_record_are_immutable():
    c=campaign();c['cache_capacity_authorization']['request']+='altered'
    with pytest.raises(ValueError):capacity.validate_authorization(c)


def test_admission_refuses_60_local_and_reserves_joint_large_fleets(tmp_path,monkeypatch):
    monkeypatch.setattr(worker_gpu,'reservations',lambda *a,**k:({0:0},[]))
    card=dict(device=0,uuid='local',total_mib=49140,used_mib=2292,usable=True)
    c=dict(domain='libero',worker_node='local',worker_egl_device=0,worker_egl_devices=[0],
        node_workers=32,worker_vram_mib=1024,worker_server_vram_headroom_mib=8192)
    assert worker_gpu.admission(c,tmp_path/'next',observed=[card])['admitted']
    assert not worker_gpu.admission(dict(c,node_workers=60),tmp_path/'next',observed=[card])['admitted']
    mib=1024**2
    memory=(257792*mib,237*1024*mib)
    c=dict(domain='libero',cache_study='frozen',node_workers=32,worker_memory_mib=4096,worker_memory_headroom_mib=32768)
    assert worker_memory.admission(c,tmp_path/'next',memory=memory)['admitted']
    assert not worker_memory.admission(dict(c,node_workers=60),tmp_path/'next',memory=memory)['admitted']
    t=dict(domain='libero',node_workers=40,worker_memory_mib=6144)
    assert worker_memory.admission(t,tmp_path/'t1',memory=(515*1024*mib,468*1024*mib))['admitted']


@pytest.mark.parametrize('ready,expected',[(False,'fallback'),(True,'full')])
def test_fleet_selection_falls_back_then_readds_t1_at_model_boundary(tmp_path,monkeypatch,ready,expected):
    monkeypatch.setattr(capacity,'t1_ready',lambda path:ready)
    plan=dict(capacity_fleets={'cosmos_libero':str(tmp_path/'full')},
        capacity_fallbacks={'cosmos_libero':str(tmp_path/'fallback')},t1_fix_evidence='fixture')
    job=dict(model='cosmos_libero',name='full',run=str(tmp_path/'full'),argv=[])
    selected=capacity.select_job(job,plan,tmp_path)
    assert selected['name']==expected
    proof=json.loads((tmp_path/'FLEET_SELECTION_cosmos_libero.json').read_text())
    assert proof['t1_fix_ready']==ready and proof['cpu_only']


def test_missing_t1_fix_evidence_refuses_before_any_remote_command(tmp_path,monkeypatch):
    from exp.r17_integration.ops import formal_eval
    monkeypatch.setattr(formal_eval,'remote',lambda *a,**k:pytest.fail('must refuse before transport'))
    assert not capacity.t1_ready(tmp_path/'missing.json')


def test_t1_transport_failure_uses_fallback_instead_of_dispatch(tmp_path,monkeypatch):
    import subprocess
    from exp.r17_integration.ops import formal_eval
    from exp.r17_integration.ops.formal_process import write_json
    from exp.r17_integration.ops.immutable_release import sha
    names=['formal_workers','timan1_workers','cache_capacity']
    p=tmp_path/'proof.json'
    write_json(p,dict(status='PASS',wrapped_recycles=4,worker_failures=0,restart_limit=0,
        files={f'exp/r17_integration/ops/{n}.py':sha(formal_eval.ROOT/f'exp/r17_integration/ops/{n}.py') for n in names}))
    def unavailable(*a,**k):raise subprocess.CalledProcessError(75,['tether'])
    monkeypatch.setattr(formal_eval,'remote',unavailable)
    assert not capacity.t1_ready(p)


def test_only_authorized_cpu_preparation_can_defer_capacity_admission():
    c=campaign();assert not capacity.can_defer_admission(c)
    c['prepare_only']=True;assert capacity.can_defer_admission(c)
    c['execution_workers']=1
    with pytest.raises(ValueError):capacity.can_defer_admission(c)
    assert not capacity.can_defer_admission(dict(prepare_only=True))


def test_busy_ram_renderer_or_disk_waits_without_invoking_a_gpu_job(tmp_path,monkeypatch):
    import time
    calls=[];waits=[]
    job=dict(name='future',run='prepared',model='dp_libero')
    def probe(selected):
        calls.append(selected)
        if len(calls)==1:raise RuntimeError('disk guard')
        return dict(admitted=len(calls)>2,cpu_only=True)
    monkeypatch.setattr(capacity,'admission',probe)
    monkeypatch.setattr(time,'sleep',lambda seconds:waits.append(seconds))
    assert capacity.wait_admission(job,{},tmp_path,interval=1)==job
    assert waits==[1,1] and len(calls)==3


def test_unverified_t1_is_refused_before_launch_claim(tmp_path,monkeypatch):
    from exp.r17_integration.ops import formal_eval,formal_prestage
    directory=tmp_path/'full';directory.mkdir()
    c=campaign()
    monkeypatch.setattr(formal_eval,'FORMAL_ROOT',tmp_path)
    monkeypatch.setattr(formal_prestage,'validate_prepared',lambda d:(c,{}))
    monkeypatch.setattr(capacity,'t1_ready',lambda path:False)
    with pytest.raises(RuntimeError,match='before launch claim'):
        formal_prestage.launch_prepared(directory)
    assert not (directory/'LAUNCH.json').exists()


def test_capacity_readmission_precedes_launch_claim_and_gpu_binding(tmp_path,monkeypatch):
    from exp.r17_integration.ops import formal_eval,formal_prestage
    directory=tmp_path/'full';directory.mkdir()
    c=campaign()
    c['run_name']='full'
    c['cache_capacity_authorization']['preclaim_admission']=True
    monkeypatch.setattr(formal_eval,'FORMAL_ROOT',tmp_path)
    monkeypatch.setattr(formal_prestage,'validate_prepared',lambda d:(c,{}))
    monkeypatch.setattr(capacity,'t1_ready',lambda path:True)
    def busy(job,plan,directory):
        assert job['run']==str(directory)
        assert not (directory/'LAUNCH.json').exists()
        raise RuntimeError('RAM still belongs to the predecessor')
    monkeypatch.setattr(capacity,'wait_admission',busy)
    with pytest.raises(RuntimeError,match='predecessor'):
        formal_prestage.launch_prepared(directory)
    assert not (directory/'LAUNCH.json').exists()


def test_closed_capacity_refresh_only_overwrites_staging_files(tmp_path):
    from types import SimpleNamespace
    from exp.r17_integration.ops import formal_prestage
    calls=[]
    launch=SimpleNamespace(ROOT=tmp_path,remote=lambda *a,**k:'{}',transfer=lambda argv:calls.append(argv))
    formal_prestage.client_release(launch,'timan107',tmp_path,'/owned/stage','/immutable/new_payload',
        dict(archive_sha256='pin'),closed_capacity_refresh=True)
    assert all('--force' in argv for argv in calls)
    assert all('/owned/stage/' in argv[3] for argv in calls)
    assert all('/immutable/' not in str(argv) for argv in calls)


def test_resize_does_not_change_canonical_fast_batch_treatment(tmp_path,monkeypatch):
    from exp.r17_integration.ops import cache_control
    from exp.r17_integration.models.fastwam.formal_batching import treatment,validate
    source=Path('/data/mw_finetune/eval/cszef')
    c=json.loads((source/'campaign.json').read_text());before=treatment(c)
    record=campaign()['cache_capacity_authorization']
    record.update(study=c['cache_study'],control=c['cache_control'])
    record['digest']=capacity.digest({k:v for k,v in record.items() if k!='digest'})
    capacity.resize(c,record,tmp_path)
    assert c['execution_workers']==85 and c['workers']==48 and validate(c)==before
    assert sum(c['execution_worker_nodes'].values())==85
    cache_control.control_proof(c['cache_control'],c['cache_control_runtime'],c)
    bad=copy.deepcopy(c);bad['checkpoint']='changed'
    with pytest.raises(ValueError,match='checkpoint'):
        cache_control.control_proof(c['cache_control'],c['cache_control_runtime'],bad)
