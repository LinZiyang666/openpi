"""CPU socket-pair regression for worker loss, using the frozen DP worker/driver."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import socket
import sys
import subprocess
import tarfile
import threading
import time
import types
from unittest.mock import Mock

import pytest

from . import worker_resilience as r
from exp.r17_integration.ops import formal_eval as f


def test_1819_control_timeout_is_unknown_until_five_minutes_or_confirmed_failure(tmp_path,monkeypatch):
    reason=dict(control_probe_failed=True,status='FAIL',error=
        "RuntimeError('command failed (70): tether exec --safe --timeout 10s timan107 -- env\n"
        "error: exec: timed out after 10s waiting for chunk\n')")
    f.write_json(tmp_path/r.STATE,dict(measured={},nodes={'timan107':34},retired=[]))
    running=r.Running(dict(execution_worker_nodes={'timan107':34}),tmp_path,{}, {})
    running.retire=Mock();now=[0.]
    monkeypatch.setattr(r,'time',types.SimpleNamespace(monotonic=lambda:now[0]))
    owned=dict(node='timan107',label='workers')
    assert f.worker_fleet_failed(reason,dispatch_started=False) is False
    assert running.check(owned,reason) is True
    now[0]=299;running.check(owned,reason);running.retire.assert_not_called()
    assert running.check(owned,dict(status='RUNNING',alive=True,child_alive=True)) is False
    now[0]=400;running.check(owned,reason)
    now[0]=699;running.check(owned,reason);running.retire.assert_not_called()
    now[0]=700;running.check(owned,reason)
    assert running.retire.call_args.args[1]['probe_failure_seconds']==300
    running.retire.reset_mock();running.unknown_since.clear()
    running.check(owned,dict(reason,error='remote_fs_spawn_timeout: fork/exec stalled'))
    running.retire.assert_not_called();running.unknown_since.clear()
    running.check(owned,dict(reason,error='node_offline: status=OFFLINE'))
    running.retire.assert_called_once();running.retire.reset_mock()
    running.check(owned,dict(status='CLOSED',exit_code=2,alive=False))
    running.retire.assert_called_once()
    running.retire.reset_mock()
    running.check(owned,dict(status='FAIL',child_alive=False))
    running.retire.assert_called_once()
    # Exercise the actual batching boundary: it must classify an exception as
    # UNKNOWN and give read-only worker control the capacity probe's 90s bound.
    bounds=[]
    def remote(node,argv,seconds,**kw):
        bounds.append(seconds);raise RuntimeError(reason['error'])
    monkeypatch.setattr(f,'remote',remote)
    monkeypatch.setattr(f,'worker_resilience',r,raising=False)
    monkeypatch.setattr(f,'UNREACHABLE_UNTIL',{})
    status={}
    f.refresh_remote_owned([dict(owned,root='sealed',python='python',request=dict(lease='/read-only-fixture'))],
                           tmp_path,status,{},-1e9)
    assert bounds==[90] and status['timan107','workers']['status']=='UNKNOWN'


def test_local_capacity_reserves_whole_server_budget_cold_and_loaded(tmp_path, monkeypatch):
    info=tmp_path/'gpu/0000:01:00.0/information'
    info.parent.mkdir(parents=True);info.write_text('Device Minor: 0\n')
    glob,read_text=Path.glob,Path.read_text
    monkeypatch.setattr(Path,'glob',lambda p,pattern:iter([info])
                        if str(p)=='/proc/driver/nvidia/gpus' else glob(p,pattern))
    def read(p,*args,**kwargs):
        if str(p)=='/proc/meminfo':return 'MemTotal: 268435456 kB\nMemAvailable: 251658240 kB\n'
        if str(p)=='/proc/sys/kernel/random/boot_id':return 'cpu-boot'
        if str(p)=='/proc/172/stat':return '172 (replica) '+' '.join(['S']+['0']*18+['5'])
        return read_text(p,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    f.write_json(tmp_path/'RECEIPT.json',dict(owned_local=[dict(label='replica_0',pid=172,start_ticks='5')]))
    resident=0;other=1060
    def query(argv,**kwargs):
        assert argv[0]=='nvidia-smi'
        return types.SimpleNamespace(stdout='<nvidia_smi_log><gpu><pci><pci_bus_id>00000000:01:00.0</pci_bus_id></pci>'
            '<fb_memory_usage><total>49140 MiB</total><used>'+str(resident+other)+' MiB</used></fb_memory_usage>'
            '<processes><process_info><pid>172</pid><used_memory>'+str(resident)+' MiB</used_memory></process_info>'
            '<process_info><pid>173</pid><used_memory>'+str(other)+' MiB</used_memory></process_info></processes>'
            '</gpu></nvidia_smi_log>')
    monkeypatch.setattr(subprocess,'run',query)
    def probe(budget=40960):return r.capacity('local',str(tmp_path/'workers/run'),530,2048,budget,str(tmp_path))
    assert probe()['capacity']==4
    resident=32768
    loaded=probe();assert loaded['capacity']==4 and loaded['cards'][0]['server_pending_mib']==8192
    other+=530
    assert probe()['capacity']==3
    assert probe(54*1024)['capacity']==0
    monkeypatch.setattr(f,'worker_paths',lambda c,n:dict(run=str(tmp_path/'workers/run')))
    import contextlib,io
    def execute(argv,seconds):
        output=io.StringIO()
        with contextlib.redirect_stdout(output):exec(argv[-1],{})
        return output.getvalue()
    monkeypatch.setattr(f,'execute',execute)
    measured=r.measure(dict(domain='libero',server_node='local',replicas=4,memory_budget_gib=40,
                            local_worker_server_vram_headroom_mib=16384),tmp_path,excluded=('timan1','timan107','timan108'))
    assert measured['local']['server_budget_mib']==40960 and measured['local']['capacity']==3


def test_capacity_probe_retries_unknown_with_long_read_only_bounds(tmp_path, monkeypatch, capsys):
    # Exercise the real self-contained probe with CPU fixtures, including PCI
    # normalization and exclusion before querying; never invoke nvidia-smi.
    infos = []
    for minor in range(4):
        info = tmp_path/'gpus'/('0000:0%d:00.0' % (minor+1))/'information'
        info.parent.mkdir(parents=True)
        info.write_text('Device Minor: '+str(minor)+'\n')
        infos.append(info)
    glob, read_text = Path.glob, Path.read_text
    monkeypatch.setattr(Path, 'glob', lambda p, pattern: iter(infos)
                        if str(p) == '/proc/driver/nvidia/gpus' else glob(p, pattern))
    def read(p, *args, **kw):
        if str(p) == '/proc/meminfo':
            return 'MemTotal: 268435456 kB\nMemAvailable: 251658240 kB\n'
        if str(p) == '/proc/sys/kernel/random/boot_id':return 'cpu-boot'
        return read_text(p, *args, **kw)
    monkeypatch.setattr(Path, 'read_text', read)
    partial = False
    def query(argv, **kw):
        assert argv == ['nvidia-smi', '-i', '0000:01:00.0,0000:02:00.0,0000:03:00.0,0000:04:00.0', '-q', '-x']
        assert kw['timeout'] == 30
        cards = ''.join('<gpu><pci><pci_bus_id>00000000:0%d:00.0</pci_bus_id></pci>'
                        '<fb_memory_usage><total>24564 MiB</total><used>0 MiB</used></fb_memory_usage>'
                        '</gpu>' % bus for bus in ([1] if partial else [1, 2, 3, 4]))
        return types.SimpleNamespace(stdout='<nvidia_smi_log>'+cards+'</nvidia_smi_log>')
    monkeypatch.setattr(subprocess, 'run', query)
    measured = r.capacity('timan107', str(tmp_path/'runs'/'next'), 530, 2048)
    assert measured['gpu_slots'] == 164 and set(measured['cards']) == {0, 1, 2, 3}
    with pytest.raises(ValueError, match='disabled'):
        r.capacity('timan108', str(tmp_path/'disabled'), 530, 2048)
    partial = True
    with pytest.raises(KeyError):
        r.capacity('timan107', str(tmp_path/'runs'/'next'), 530, 2048)

    calls = {n: 0 for n in r.NODES}
    monkeypatch.setattr(f, 'worker_paths', lambda c, n: dict(run=str(tmp_path/n)))
    def execute(argv, seconds):
        assert seconds == 92
        node = argv[5] if argv[0] == 'tether' else 'local'
        if node != 'local':assert argv[:5] == ['tether', 'exec', '--safe', '--timeout', '90s']
        calls[node] += 1
        if node == 'timan107':
            raise subprocess.TimeoutExpired(argv[:6], 90)
        if node == 'timan1' and calls[node] == 1:return '{}'
        return json.dumps(dict(capacity={'timan1': 83, 'local': 0}[node]))
    monkeypatch.setattr(f, 'execute', execute)
    rows = r.measure(dict(domain='libero', server_node='h100'), tmp_path)
    assert calls == {'timan107': 2, 'timan1': 2, 'local': 1}
    assert rows['timan1']['capacity'] == 83 and rows['timan1']['probe_attempts'] == 2
    assert len(rows['timan1']['probe_errors']) == 1
    assert rows['timan107']['capacity'] is None and rows['timan107']['probe_status'] == 'UNKNOWN'
    assert len(rows['timan107']['probe_errors']) == 2
    assert rows['local']['capacity'] == 0 and rows['local']['probe_status'] == 'MEASURED'
    assert r.allocate({'timan107': 60}, rows) == {'timan1': 60}
    with pytest.raises(RuntimeError, match='no usable worker'):
        r.allocate({'timan107': 1}, {'timan107': rows['timan107']})
    log = capsys.readouterr().err
    assert 'node=timan1 attempt=1 status=UNKNOWN retrying' in log
    assert 'node=timan107 attempt=2 status=UNKNOWN excluded' in log

    # The longer bound is confined to measure(), not fleet spawn/control.
    bounds = []
    monkeypatch.setattr(f, 'execute', lambda argv, seconds: bounds.append((argv[4], seconds)) or '{}')
    with r.coordinator():
        f.remote('timan107', ['control'], seconds=5)
        f.remote('timan107', ['spawn'], seconds=60)
    assert bounds == [('5s', 7), ('10s', 12)]


def test_worker_node_loss_keeps_frozen_driver_and_survivor_running(tmp_path, monkeypatch):
    root=Path('/data/mw_finetune/eval/ffk1015_dp_mt50_reduced_descent')
    original=json.loads((root/'campaign.json').read_text())
    archive=root/'workers.tar'
    pin=hashlib.sha256(archive.read_bytes()).hexdigest()
    campaign=dict(run_name='cpu',domain='metaworld',model='dp',cache_study='frozen',replicas=4,
                  workers=2,worker_nodes={'timan107':1,'timan108':1},execution_workers=2,
                  execution_worker_nodes={'timan107':1,'timan108':1},seeds=[7,11],R=4,B=16,W=2)
    before=copy.deepcopy(campaign)
    (tmp_path/'campaign.json').write_text(json.dumps(campaign))
    rows={n:dict(capacity=c,boot_id='original-boot') for n,c in
          [('timan107',4),('timan108',0),('timan1',8),('local',4)]}
    assert r.allocate({'timan108':8},rows)=={'timan1':8}
    assert r.allocate({'timan107':20},{'timan107':dict(capacity=3)})=={'timan107':3}
    with pytest.raises(RuntimeError,match='no usable worker'):
        r.allocate({'timan108':40},{'timan108':dict(capacity=0)})
    rows['timan108']['capacity']=4
    f.write_json(tmp_path/r.STATE,dict(nodes=campaign['execution_worker_nodes'],retired=[],measured=rows))
    monkeypatch.setattr('exp.r17_integration.ops.cache_control.attach',lambda *a:None)
    # Load the actual sealed control module; do not rebuild/extract the archive.
    with tarfile.open(archive) as tar:
        member=next(m for m in tar if m.name.endswith('/ops/formal_process.py'))
        controls=types.ModuleType('cpu_frozen_formal_process')
        exec(compile(tar.extractfile(member).read(),str(archive)+'/'+member.name,'exec'),controls.__dict__)
    owned=[]
    for node in ('timan107','timan108'):
        owned.append(dict(node=node,label='workers',python='unused',root='unused',
            request=dict(lease=str(tmp_path/'vanished_after_reboot'/node/'workers.lease'),argv=['CPU-fixture','x'*9000])))
    calls=[]
    def remote(node,argv,**kw):
        calls.append(node)
        if node=='timan108':
            return json.dumps(controls.controls([dict(label='workers',request=owned[1]['request'])]))
        if node=='h100':raise RuntimeError('inference resident failure remains fatal')
        return json.dumps({'workers':dict(status='RUNNING',alive=True,child_alive=True)})
    monkeypatch.setattr(f,'remote',remote)
    monkeypatch.setattr(f,'transfer',lambda *a,**kw:pytest.fail('CPU test must never transfer files'))
    monkeypatch.setattr(f,'worker_resilience',r,raising=False)
    cache={}
    f.refresh_remote_owned(owned,tmp_path,cache,{},-1e9)
    assert cache['timan108','workers']['control_probe_failed']
    assert 'FileNotFoundError' in cache['timan108','workers']['error']
    assert cache['timan107','workers']['status']=='RUNNING'
    with pytest.raises(RuntimeError,match='resident failure'):
        f.refresh_remote_owned([dict(owned[0],node='h100',label='replica_0',request={})],tmp_path,{}, {},-1e9)

    # Launch-time loss is recorded through the native execution-topology knob;
    # failed admission during preflight is redistributed a second time.
    staging=tmp_path/'launch';staging.mkdir()
    (staging/'campaign.json').write_text(json.dumps(campaign))
    with monkeypatch.context() as m:
        m.setattr('exp.r17_integration.ops.timan1_workers.authorize',r.timan1_gpu2.authorize)
        measured={'timan107':dict(capacity=1),'timan108':dict(capacity=0),
                  'timan1':dict(capacity=4),'local':dict(capacity=0)}
        m.setattr(r,'measure',lambda *a,**kw:copy.deepcopy(measured))
        selected,deployment=r.select(campaign,staging,{'timan107':{},'timan1':{}})
        assert selected['execution_worker_nodes']=={'timan1':2}
        def preflight(node,item):
            if node=='timan107':raise RuntimeError('admission refused')
        assert set(r.launch_workers(selected,staging,deployment,preflight))=={'timan1'}
        assert selected['execution_worker_nodes']=={'timan1':2}
        for key in ('workers','worker_nodes','seeds','R','B','W'):
            assert selected[key]==before[key]
        assert json.loads((staging/'WORKER_REDISTRIBUTION.json').read_text())['worker_nodes']=={'timan1':2}
        fresh=tmp_path/'stage';fresh.mkdir()
        c=copy.deepcopy(before)
        def stage_measure(c,d,excluded=()):
            return {n:dict(capacity=0 if n in excluded else value) for n,value in
                    [('timan107',1),('timan108',1),('timan1',4),('local',0)]}
        m.setattr(r,'measure',stage_measure)
        def prepare_node(c,d):
            node=next(iter(c['execution_worker_nodes']))
            if node=='timan108':raise FileNotFoundError('agent node lost during stage')
            assert c['worker_ram_reservation_mib']==512
            return {node:dict(workers=c['execution_worker_nodes'][node])}
        assert set(r.prepare(c,fresh,prepare_node))=={'timan1'}
        assert c['workers']==before['workers'] and c['worker_nodes']==before['worker_nodes']
        # Exercise the real prepared-launch boundary, with every process and
        # transfer replaced; its resilience hook must run before any dispatch.
        from exp.r17_integration.ops import formal_prestage as prestage
        boundary=tmp_path/'prepared_boundary';boundary.mkdir()
        (boundary/'EPISODE_PLAN.json').write_text(json.dumps(dict(tasks=[],expected_episodes=0)))
        c=copy.deepcopy(before)
        c.update(server_node='local',runtime='sealed',runtime_identity='pin',endpoint='unused:1234')
        with m.context() as entry:
            entry.setattr(prestage,'validate_prepared',lambda d:(c,{'timan107':{},'timan1':{}}))
            entry.setattr(r,'measure',lambda *a,**kw:copy.deepcopy(measured))
            entry.setattr(prestage,'verify',lambda *a,**kw:dict(present=True))
            entry.setattr(f,'open_tunnels',lambda *a:[])
            entry.setattr('exp.r17_integration.ops.formal_driver.plan',lambda c:(None,None,[],[]))
            def preflight_only(c,d,deployment,preflight):
                assert c['execution_worker_nodes']=={'timan1':2}
                c.pop('_worker_node_measurements')
                return deployment
            entry.setattr(r,'launch_workers',preflight_only)
            run=Mock(return_value=0);entry.setattr(f,'run_campaign',run)
            assert prestage.launch_prepared(boundary)==0
            run.assert_called_once()
            assert run.call_args.args[0]['workers']==2

    # The command patch honors measured VRAM/RAM in the actual frozen worker
    # modules, without allowing protected timan108 GPUs into an assignment.
    from exp.r17_integration import ops
    from exp.r17_integration.ops import worker_gpu as native_gpu
    inventory_source=inspect.getsource(native_gpu.inventory)
    with monkeypatch.context() as m,tarfile.open(archive) as tar:
        frozen={}
        for name in ('worker_gpu','worker_memory'):
            fullname='exp.r17_integration.ops.'+name
            member=next(x for x in tar if x.name.endswith('/ops/'+name+'.py'))
            module=types.ModuleType(fullname);module.__package__='exp.r17_integration.ops'
            m.setitem(sys.modules,fullname,module);m.setattr(ops,name,module,raising=False)
            exec(compile(tar.extractfile(member).read(),member.name,'exec'),module.__dict__)
            frozen[name]=module
        r.install_worker(inventory_source)
        card=dict(device=3,uuid='allowed',usable=True,total_mib=49140,used_mib=10,processes=[])
        gpu_job=dict(worker_node='timan108',worker_egl_devices=[3],worker_egl_device=3,
                     node_workers=332,worker_vram_mib=133)
        admitted=frozen['worker_gpu'].admission(gpu_job,tmp_path/'gpu/next',observed=[card,dict(card,device=1)])
        assert admitted['admitted'] and {a['device'] for a in admitted['assignments']}=={3}
        assert not frozen['worker_gpu'].admission(dict(gpu_job,node_workers=333),tmp_path/'gpu/next',observed=[card])['admitted']
        assert frozen['worker_memory'].reservation_mib(dict(domain='metaworld',cache_study='frozen',worker_ram_reservation_mib=512))==512

    # Load the pinned driver mechanisms in memory, including its stock socket
    # disconnect/requeue path and asynchronous accepted-attempt commit fence.
    import importlib
    names=['openpi.conductor.driver','exp.r17_integration.conductor.driver',
           'exp.r17_integration.qualification.driver','exp.r17_integration.ops.formal_driver_fast']
    for name in names:
        package,_,leaf=name.rpartition('.')
        parent=importlib.import_module(package)
        relative=('src/' if name.startswith('openpi.') else '')+name.replace('.','/')+'.py'
        path=Path(original['driver_runtime'])/relative
        module=types.ModuleType(name)
        module.__package__=package;module.__file__=str(path)
        monkeypatch.setitem(sys.modules,name,module)
        monkeypatch.setattr(parent,leaf,module,raising=False)
        exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    from openpi.conductor import driver as base,protocol as wire
    from openpi.conductor.task import EpisodeTask,ServerEndpoint
    from exp.r17_integration.conductor.manifest import EndpointSpec
    from exp.r17_integration.conductor.strategy import MetaWorldStrategy
    from exp.r17_integration.qualification.driver import ProbeRecordingDriver
    from exp.r17_integration.ops.formal_driver_fast import AsyncCampaignDriver
    monkeypatch.setattr(sys,'argv',['driver','--campaign',str(tmp_path/'campaign.json')])
    r.install_driver()
    ep=ServerEndpoint('unused',1234)
    tasks=[EpisodeTask('task'+str(i),'yaml','eval','metaworld_mt50',0,i,7+i,ep.host,ep.port,
        'bundle',extra=dict(model_identity='model',norm_identity='norm',layout_identity='layout',
                           benchmark_protocol='benchmark',num_steps=100,original_init_idx=i)) for i in range(2)]
    endpoint=EndpointSpec(ep,'model','norm','layout','benchmark',allowed_k=(100,),capacity=2)
    strategy=MetaWorldStrategy(tasks,endpoints=[endpoint])
    d=AsyncCampaignDriver(strategy,evidence_directory=tmp_path/'driver',dispatch_path=tmp_path/'driver/dispatch.jsonl',
        rejection_path=tmp_path/'driver/rejections.json',expected_workers=2,shutdown_seconds=.2,
        yaml_weights={'yaml':1},servers=[ep],journal_path=str(tmp_path/'driver/journal.jsonl'),
        ctl_factory=lambda *a:Mock(),scheduler_kwargs=dict(eval_concurrency=2,max_episode_retries=3))
    d.drive_stages_once()
    d._shutdown_external_stop=lambda:False
    sockets=[];threads=[]
    def connect(node):
        client,server=socket.socketpair();client.settimeout(2)
        thread=threading.Thread(target=d._handle_conn,args=(server,lambda:False),daemon=True)
        thread.start();sockets.append(client);threads.append(thread)
        return client,dict(worker_id='cpu_'+node+'_0',server_host=ep.key)
    def pull(client,identity):
        wire.send_message(client,wire.MSG_PULL,identity)
        return wire.recv_message(client)
    def result(task):
        return dict(task_uid=task['task_uid'],attempt=task['attempt'],success=True,n_steps=1,
                    per_step_rows=[dict(actual_nfe=100,source='policy')],duration_s=.01)
    def wait(predicate):
        until=time.monotonic()+3
        while not predicate():
            assert time.monotonic()<until,'CPU fixture timed out'
            time.sleep(.005)
    try:
        a,ida=connect('timan107');b,idb=connect('timan108')
        _,first=pull(a,ida);_,lost=pull(b,idb)
        wire.send_message(a,wire.MSG_REPORT_RESULT,result(first['task']))
        wait(lambda:d.scheduler.progress()['episodes_done']==1)
        report={};state=r.Running(campaign,tmp_path,{},report)
        assert state.check(owned[1],cache['timan108','workers'])
        assert campaign['execution_worker_nodes']=={'timan107':1,'timan108':1}
        # A vanished stage alone is UNKNOWN; only sustained failure retires it.
        state.unknown_since['timan108']=time.monotonic()-301
        assert state.check(owned[1],cache['timan108','workers'])
        state.require_workers()
        assert campaign['execution_worker_nodes']=={'timan107':1}
        for key in ('workers','worker_nodes','seeds','R','B','W'):
            assert campaign[key]==before[key]
        # A surviving worker's pull closes only the lost node's driver socket.
        _,again=pull(a,ida)
        threads[1].join(2);assert not threads[1].is_alive()
        while 'task' not in again:
            _,again=pull(a,ida)
        assert again['task']['task_uid']==lost['task']['task_uid']
        assert again['task']['attempt']==lost['task']['attempt']+1
        assert {k:v for k,v in again['task'].items() if k!='attempt'}=={k:v for k,v in lost['task'].items() if k!='attempt'}
        wire.send_message(a,wire.MSG_REPORT_RESULT,result(again['task']))
        wait(lambda:d.scheduler.progress()['episodes_done']==2)
        d.drive_stages_once()
        wait(lambda:d.scheduler.all_done())
        # A late result from the lost generation cannot duplicate acceptance.
        d.handle_result(result(lost['task']))
        kind,_=pull(a,ida);assert kind==wire.MSG_SHUTDOWN
        a.close();threads[0].join(2);assert not threads[0].is_alive()
        # Exercise the frozen shutdown accounting without opening a listener.
        monkeypatch.setattr(base.ConductorDriver,'run',lambda *a,**kw:None)
        d.run(stop=lambda:False)
        assert d.expected_workers==1 and d.worker_shutdown['status']=='PASS'
        assert d.probe_rejections==[]
        assert state.complete()
        assert state.check(owned[0],dict(status='CLOSED',exit_code=0))
        assert not owned[0].get('retired_worker_node')
        d.close()
        accepted=[json.loads(x) for x in (tmp_path/'driver/journal.jsonl').read_text().splitlines()]
        assert sum(x['accepted'] for x in accepted)==2
        assert {x['task_uid'] for x in accepted if x['accepted']}=={t.task_uid for t in tasks}
        calls.clear();f.refresh_remote_owned(owned,tmp_path,cache,{},-1e9)
        assert calls==['timan107']  # dead fleet no longer renewed/counted
        with monkeypatch.context() as m:
            m.setattr(f,'remote',lambda *a,**kw:json.dumps('replacement-boot'))
            reboot=state.cleanup(owned[1],lambda *a:pytest.fail('reboot must not signal stale owners'))
            assert reboot['reboot_proven'] and reboot['alive'] is False
            def offline(*a,**kw):raise RuntimeError('agent offline')
            m.setattr(f,'remote',offline)
            assert state.cleanup(owned[1],Mock())['status']=='UNKNOWN'
        from exp.r17_integration.ops.formal_finalize import closed_proof
        normal=dict(status='CLOSED',alive=False,child_alive=False,supervisor_alive=False,
                    owned_worker_survivors=[],exit_code=0,closure_confirmed=True)
        proof=dict(all_owned_closed=True,owned_local=[],owned_remote=owned,
            worker_node_losses=state.retired,driver_completion=dict(verified_completion=True),
            cleanup=[dict(normal,node='timan107',label='workers'),
                     dict(reboot,node='timan108',label='workers',closure_confirmed=True)])
        assert set(closed_proof(proof))=={'timan107'}
        # The same retirement path handles restart-budget exhaustion; no worker
        # node left is fatal only while the plan still has work.
        state.check(owned[0],dict(status='CLOSED',exit_code=2,reason='worker_restart_budget_exhausted'))
        monkeypatch.setattr(state,'complete',lambda:False)
        with pytest.raises(RuntimeError,match='no usable worker'):
            state.require_workers()
    finally:
        d._shutdown_end.set()
        for client in sockets:client.close()
        for thread in threads:thread.join(2)
        d.close()
    assert pin==hashlib.sha256(archive.read_bytes()).hexdigest()
