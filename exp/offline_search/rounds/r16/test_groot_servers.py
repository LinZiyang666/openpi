"""Operational CPU tests. Every process/port/remote operation is mocked."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from . import conductor_runner as r
from . import conductor as c
from .test_conductor import fixture_group, driver, pull, report


def group(root):
    entries=[]
    for suite in ('libero_10','libero_spatial'):
        entries.append(dict(row=dict(arm=suite,suite=suite,method='frozen:Method',kwargs={'frozen':1},
            cell='groot_'+suite+'_cache',plugin_args=['--os-fit-artifact','/frozen/fit'],yaml='/frozen/config')))
    return dict(root=str(root),model='groot',k=1,phase='dev',entries=entries,group_sha256='synthetic')


def arguments(root,n,ports=None):
    return NS(group=root/'group.json',groot_servers_per_suite=n,ports=ports,port=23179)


@pytest.mark.parametrize('n',[3,4])
def test_resident_specs_per_suite_ports_and_checkpoints(tmp_path,monkeypatch,n):
    g=group(tmp_path);ports=list(range(23100,23100+2*n))
    specs=r.server_specs(g,ports,7,groot_servers_per_suite=n)
    assert [s['suite'] for s in specs]==['libero_10']*n+['libero_spatial']*n
    assert [s['port'] for s in specs]==ports and len({s['tag'] for s in specs})==2*n
    for suite in ('libero_10','libero_spatial'):
        original=r.server_specs(g,[23120,23121,23122,23123],1)
        reference=next(s for s in original if s['suite']==suite)
        for s in specs:
            if s['suite']==suite:
                for key in ('plugin','yaml','env','registry','model','need_mb'):
                    assert s[key]==reference[key]
                monkeypatch.setattr(r,'BASE_COMMAND',lambda spec:(['python','/serve_groot.py','--checkpoint','/ckpt/'+spec['suite']],{}))
                argv,_=r.resident_command(s)
                assert argv[argv.index('--checkpoint')+1]=='/ckpt/'+suite
                assert argv[argv.index('--resident-suite')+1]==suite
    with pytest.raises(ValueError):r.server_specs(g,ports[:-1]+ports[:1],7,groot_servers_per_suite=n)
    with pytest.raises(ValueError):r.server_specs(g,ports[:-1],7,groot_servers_per_suite=n)
    with pytest.raises(ValueError):r.server_count(g,0)


def fake_local(monkeypatch):
    from exp.offline_search.closed_loop.ops.h100 import local_server as local
    stopped=[];choices=[]
    monkeypatch.setattr(local,'status',lambda s:dict(running=True))
    monkeypatch.setattr(local,'stop',lambda s,**kw:stopped.append(s.copy()) or 'STOPPED mock')
    def choose(count=4,requested=None):
        choices.append((count,requested))
        return list(requested) if requested is not None else list(range(23104,23104+count))
    monkeypatch.setattr(local,'choose_ports',choose)
    return local,stopped,choices


@pytest.mark.parametrize('n',[3,4])
@pytest.mark.parametrize('explicit',[False,True])
def test_resize_keeps_receipts_and_shrink_stops_only_surplus(tmp_path,monkeypatch,n,explicit):
    g=group(tmp_path);old=r.server_specs(g,[23100,23101,23102,23103],1)
    plan=tmp_path/'runner_state.json';c.write_json(plan,dict(servers=old,generation=1,group_sha256=g['group_sha256']))
    local,stopped,choices=fake_local(monkeypatch)
    target=[23100,23101,*range(23104,23104+n-2),23102,23103,*range(23104+n-2,23104+2*(n-2))]
    new=r.prepare_servers(g,arguments(tmp_path,n,target if explicit else None))
    assert len(new)==2*n and len({s['port'] for s in new})==2*n
    assert not stopped and all(s in new for s in old) # keep exact live owner receipts/log tags
    assert [s['suite'] for s in new]==['libero_10']*n+['libero_spatial']*n
    assert json.loads(plan.read_bytes())['generation']==2
    again=r.prepare_servers(g,arguments(tmp_path,n))
    assert again==new and json.loads(plan.read_bytes())['generation']==2
    fallback=r.prepare_servers(g,arguments(tmp_path,2))
    assert fallback==old and stopped==[s for s in new if s not in old]
    assert json.loads(plan.read_bytes())['retired']==[]


def test_resize_failure_keeps_retirement_receipts_for_resume(tmp_path,monkeypatch):
    g=group(tmp_path);old=r.server_specs(g,list(range(23100,23106)),1,groot_servers_per_suite=3)
    plan=tmp_path/'runner_state.json';c.write_json(plan,dict(servers=old,generation=1,group_sha256=g['group_sha256']))
    _,stopped,_=fake_local(monkeypatch)
    real_stop=r.stop_servers
    monkeypatch.setattr(r,'stop_servers',lambda specs:(_ for _ in ()).throw(RuntimeError('mock interruption')))
    with pytest.raises(RuntimeError,match='mock interruption'):r.prepare_servers(g,arguments(tmp_path,2))
    state=json.loads(plan.read_bytes());assert len(state['servers'])==4 and len(state['retired'])==2
    monkeypatch.setattr(r,'stop_servers',real_stop)
    resumed=r.prepare_servers(g,arguments(tmp_path,2))
    assert len(resumed)==4 and stopped==state['retired']
    assert json.loads(plan.read_bytes())['retired']==[]


@pytest.mark.parametrize('released',[True,False])
def test_stop_attempts_all_servers_before_timeout_wait(tmp_path,monkeypatch,released):
    from exp.offline_search.closed_loop.ops.h100 import local_server as local
    specs=r.server_specs(group(tmp_path),[23100,23101,23102,23103],1)
    events=[]
    def stop(spec,timeout_s=60):
        events.append(('stop',spec['port'],timeout_s))
        if spec is specs[0] and timeout_s!=0:raise RuntimeError('LOCAL_PORT_RELEASE_TIMEOUT: mock slow exit')
        return 'STOPPED mock'
    monkeypatch.setattr(local,'stop',stop)
    monkeypatch.setattr(local,'matching',lambda s:dict(spec=s))
    monkeypatch.setattr(r.time,'monotonic',lambda:100)
    def wait(spec,receipts,deadline):
        assert [e[1] for e in events[:4]]==[s['port'] for s in specs]
        assert deadline==340 and receipts==[dict(spec=spec)]
        events.append(('wait',spec['port']))
        if not released:raise RuntimeError('LOCAL_PORT_RELEASE_TIMEOUT: still exiting')
    monkeypatch.setattr(local,'wait_port_release',wait)
    if released:r.stop_servers(specs)
    else:
        with pytest.raises(RuntimeError,match='R16_STOP_INCOMPLETE.*23100'):r.stop_servers(specs)
    assert events[:4]==[('stop',s['port'],60) for s in specs]
    assert ('wait',23100) in events


def test_changed_endpoint_count_keeps_completed_and_fences_inflight(tmp_path):
    # Existing strategy/driver, synthetic outcomes only; no worker/server launch.
    g=fixture_group(tmp_path);d=driver(g);d.drive_stages_once()
    completed=pull(d);inflight=pull(d);report(d,completed)
    endpoints={f'cpu:{i}':['libero_10'] for i in range(6)}
    resumed=c.R16Driver(c.R16Strategy(g),endpoints,yaml_weights={e['row']['arm']:300 for e in g['entries']},
        servers=[c.ServerEndpoint('cpu',i) for i in range(6)],journal_path=str(tmp_path/'journal.jsonl'),
        ctl_factory=lambda ep:NS())
    resumed.drive_stages_once();issued=[]
    for _ in range(301):
        ep=pull(resumed,'cpu:0')
        if ep is None:break
        issued.append(ep)
    assert completed.task_uid not in {ep.task_uid for ep in issued}
    retried=next(ep for ep in issued if ep.task_uid==inflight.task_uid)
    assert retried.attempt>inflight.attempt
    report(resumed,inflight,False)
    assert inflight.task_uid not in c.completed_rows(tmp_path/'journal.jsonl')
    report(resumed,retried,True)
    assert c.completed_rows(tmp_path/'journal.jsonl')[inflight.task_uid]['success']


def test_operational_registration_accepts_runner_and_rejects_measurement_tamper(tmp_path):
    from . import analyze as a,conductor_analysis as ca
    root=Path(r.__file__).resolve().parent
    old=json.loads((root/'before/groot_servers_CONDUCTOR_LOCK.json').read_bytes())
    current=json.loads((root/'CONDUCTOR_LOCK.json').read_bytes())
    runner_path=str(root/'conductor_runner.py')
    assert {k for k,v in old['new_sources'].items() if current['new_sources'][k]!=v}=={runner_path}
    assert old['adapted_sources']==current['adapted_sources']
    record=current['adapted_sources'][str(root/'analyze.py')]
    reader=a.Reader()
    ca.verify_amendment(reader,str(root/'analyze.py'),record['original_sha256'],reader.read(root/'analyze.py'))
    reader.stable()
    class Tampered(a.Reader):
        def read(self,path):
            data=super().read(path)
            return data+b'\n# changed gate\n' if Path(path)==root/'conductor.py' else data
    bad=Tampered()
    with pytest.raises(ValueError,match='Changed conductor implementation'):
        ca.verify_amendment(bad,str(root/'analyze.py'),record['original_sha256'],bad.read(root/'analyze.py'))
