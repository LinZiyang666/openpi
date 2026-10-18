"""Host loss must change placement only; no real transport/process/GPU calls."""
import json
import subprocess
import socket
import signal
from types import SimpleNamespace

from . import vla_workers as v


def test_legacy_host_loss_preserves_native_commands_and_r16_attestation(tmp_path, monkeypatch):
    job=dict(name='held',run=str(tmp_path),pipeline='R15',workers={'timan108':39},
             topology=dict(env=dict(SERVER_HOST='h100',WORKER_HOST='timan108',WPS='13',
                 PORTS='23230,23231,23232',STAGE1_ONLY='0')))
    arms=b'[{"arm":"held_c10"}]';(tmp_path/'arms.json').write_bytes(arms)
    (tmp_path/'h100_sync').mkdir();(tmp_path/'h100_sync/synced.json').write_text('{"worker_host":"timan108"}')
    admission=dict(capacity=64,cards={str(i):dict(total=8192,used=0,pending=0) for i in range(8)})
    monkeypatch.setattr(v,'probe',lambda h,p:dict(usable=h=='timan107',boot_id='cpu',host=h,admission=admission))
    calls=[]
    monkeypatch.setattr(v.subprocess,'run',lambda argv,**kw:calls.append((argv,kw)))
    v.r15(job,'launch')
    assert [c[0][3:5] for c in calls]==[['sync','--concurrent'],['chain',str(tmp_path)]]
    assert calls[0][1]['env']['OSCL_ISOLATE_GENERATED_ASSETS'] == '1'
    for argv,kw in calls:
        assert argv[-1]=='held_c10'
        assert {k:kw['env'][k] for k in job['topology']['env']}==dict(job['topology']['env'],WORKER_HOST='timan107')
    assert (tmp_path/'arms.json').read_bytes()==arms
    from exp.offline_search.rounds.r17.autoscale_1017.defaults import queue_workers
    assert json.loads((tmp_path/'WORKER_HOST_SELECTION.json').read_text())['execution_workers']=={
        'timan107': sum(queue_workers(job).values())}

    from exp.offline_search.rounds.r16 import conductor as c
    from openpi.conductor import agent as a
    code=c.R16Driver._record_pull.__code__
    routed=v.routed_code(code,'timan1','/new/lock')
    assert routed.co_consts==tuple('timan1' if x=='timan107' else x for x in code.co_consts)
    assert routed.co_code==code.co_code  # all non-host attestation bytecode unchanged
    monkeypatch.setattr(c.run_agent,'__code__',c.run_agent.__code__)
    monkeypatch.setattr(c,'main',lambda:None)
    monkeypatch.setattr(a,'WorkerSpec',a.WorkerSpec)
    monkeypatch.setattr(a,'subprocess',a.subprocess)
    spawned=[]
    monkeypatch.setattr(subprocess,'Popen',lambda argv,**kw:spawned.append((argv,kw)))
    boot=v.r16_agent_bootstrap('timan1',str(tmp_path/'island'));exec(compile(boot,'<cpu-agent>','exec'),{})
    spec=a.WorkerSpec('worker','server','7',seed=2026101401,replan_steps=5,
                      env=dict(MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='7'))
    assert spec.gpu_id=='2' and spec.env['MUJOCO_EGL_DEVICE_ID']=='2' and spec.env['CUDA_VISIBLE_DEVICES']==''
    assert spec.seed==2026101401 and spec.replan_steps==5
    a.subprocess.Popen(['conda','run','-p','/scratch/zixuans8/libero_sim','python','-m','examples.libero.worker_entry','--seed','2026101401'],env=spec.env)
    argv,kw=spawned[0];assert argv[:4]==['conda','run','-p','/scratch/zixuans8/libero_sim']
    assert argv[4:8]==['/usr/bin/python3','-B','-S','-c']
    compile(argv[8],'<cpu-gpu2-bootstrap>','exec')
    assert kw['env']==spec.env
    monkeypatch.setattr(v,'probe',lambda h,p:dict(usable=h=='timan1',host=h))
    assert v.choose(dict(pipeline='R16',workers={'timan107':64}))[0]=='timan1'
    # Exercise the actual archived agent's spec generator, without starting it.
    generated=[]
    class CPUAgent:
        def __init__(self,specs,*args):generated.extend(specs)
        def run(self):pass
        def stop(self):pass
    monkeypatch.setattr(a,'WorkerAgent',CPUAgent)
    monkeypatch.setattr(socket,'gethostname',lambda:'timan1')
    monkeypatch.setattr(signal,'signal',lambda *args:None)
    for key in ('OPENPI_WORKER_MEMORY_GUARD','OPENPI_WORKER_GOVERNOR','OPENPI_AUTOSCALE_MODEL','OPENPI_AUTOSCALE_RUN'):
        monkeypatch.setenv(key,'')
    c.run_agent(SimpleNamespace(phase='test',model='pi05',servers='a,b,c',workers=64,
        driver_host='ziyanglin.com',port=23159,lease_stdin=False,lease_file=None))
    assert len(generated)==64 and {s.gpu_id for s in generated}=={'2'}
    assert all(s.seed==2026101401 and s.replan_steps==5 and s.resize_size==224 and s.probe for s in generated)
