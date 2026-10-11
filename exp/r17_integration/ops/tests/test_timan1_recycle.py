"""CPU reproduction of a namespace wrapper forwarding a child's exit75."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from exp.r17_integration.ops import formal_eval,timan1_workers as t1
from exp.r17_integration.ops.formal_process import alive,write_json
from exp.r17_integration.ops.worker_lifetime import planned_recycle as direct_recycle


@pytest.fixture
def wrapped_fleet(tmp_path):
    worker=tmp_path/'native.py'
    worker.write_text('''import pathlib,os
from exp.r17_integration.ops.timan1_workers import record_native_process
from exp.r17_integration.ops.worker_lifetime import limited_loop
generation=int(pathlib.Path.cwd().name.split('_')[-1])
record_native_process()
if generation<4:
    class Base:
        def _run_one(self,*a):pass
        def run_forever(self,stop=None):
            while not stop():self._run_one(None,None)
    limited_loop(Base,4,os.environ['R17_WORKER_RECYCLE_RECEIPT'])().run_forever()
''')
    wrapper=tmp_path/'namespace-wrapper'
    wrapper.write_text(f'#!{sys.executable}\nimport subprocess,sys\nraise SystemExit(subprocess.call([sys.executable,{str(worker)!r}]))\n')
    wrapper.chmod(0o700)
    run=tmp_path/'runs/owned'
    campaign=dict(run_name='owned',worker_node='timan1',domain='libero',worker_run=str(run),
                  node_workers=1,seconds=30,worker_memory_mib=6144,worker_restart_limit=0,
                  worker_max_episodes=4)
    write_json(tmp_path/'campaign.json',campaign)
    harness=f'''import sys
from exp.r17_integration.ops import formal_workers as f,worker_memory as m,timan1_workers as t1
t1.WRAPPER={str(wrapper)!r}
m.admission=lambda *a,**kw:dict(admitted=True)
f.worker_command=lambda *a,**kw:('owned',[t1.WRAPPER])
sys.argv=['cpu','--campaign',{str(tmp_path/'campaign.json')!r}]
f.main()
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(formal_eval.ROOT))
    result=subprocess.run([sys.executable,'-c',harness],env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stdout+result.stderr
    receipt=json.loads((run/'receipt.json').read_text())
    return receipt,wrapper


def test_four_wrapped_recycles_spend_no_restart_budget(wrapped_fleet):
    receipt,wrapper=wrapped_fleet
    assert receipt['status']=='PASS' and receipt['reason']=='normal_exit'
    assert [p['exit_code'] for p in receipt['pids']]==[75,75,75,75,0]
    assert [p['generation'] for p in receipt['pids']]==list(range(5))
    assert len(receipt['worker_recycles'])==4 and receipt['worker_failures']==[]
    tokens={e['wrapped_generation_token'] for e in receipt['pids']}
    assert len(tokens)==5
    for entry in receipt['pids'][:4]:
        assert not direct_recycle(entry,75)  # Original regression: bwrap PID != native PID.
        assert entry['wrapped_recycle']['native']['pid']!=entry['pid']
        assert entry['wrapped_recycle']['generation_bound']
    assert all(not alive(e) for e in receipt['pids'])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(t1,'WRAPPER',str(wrapper))
        assert t1.planned_recycle(receipt['pids'][0],75)


def test_wrapped_receipt_must_match_generation_ancestry_native_birth_and_n4(wrapped_fleet):
    receipt,wrapper=wrapped_fleet
    entry=receipt['pids'][0]
    marker_path=Path(entry['recycle_receipt'])
    process_path=marker_path.with_name('NATIVE_PROCESS.json')
    marker=json.loads(marker_path.read_text())
    process=json.loads(process_path.read_text())
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(t1,'WRAPPER',str(wrapper))
        for bad in (dict(entry,node='local'),dict(entry,wrapped_generation_token='wrong'),
                    dict(entry,start_ticks='reused'),dict(entry,maximum_episodes=8),
                    dict(entry,argv=['unprotected-python'])):
            assert not t1.planned_recycle(bad,75)
        assert not t1.planned_recycle(entry,-9)
        for bad in (dict(marker,pid=entry['pid']),dict(marker,start_ticks='reused'),
                    dict(marker,completed_episodes=3),dict(marker,maximum_episodes=8),
                    dict(marker,reason='crash')):
            write_json(marker_path,bad)
            assert not t1.planned_recycle(entry,75)
        write_json(marker_path,marker)
        bad=copy.deepcopy(process);bad['ancestors']=[]
        write_json(process_path,bad)
        assert not t1.planned_recycle(entry,75)
        process_path.unlink()
        assert not t1.planned_recycle(entry,75)


def test_t1_command_stays_inside_pinned_guard_and_relay_refuses_unguarded_launch(monkeypatch):
    from exp.r17_integration.ops.formal_workers import worker_command
    campaign=dict(run_name='owned',worker_node='timan1',domain='libero',cache_study='sealed',
        worker_nodes={'timan1':1},node_workers=1,endpoint='host:1',driver_host='driver',driver_port=2,
        worker_benchmark='sealed',worker_egl_device=3,worker_max_episodes=4)
    _,argv=worker_command(campaign,0)
    assert argv[0]==t1.WRAPPER
    assert argv[argv.index('-m')+1]=='exp.r17_integration.ops.timan1_workers'
    assert argv[argv.index('--egl-device')+1]=='3'
    monkeypatch.delenv('R17_GPU3_DEVICE_ISOLATION',raising=False)
    with pytest.raises(ValueError,match='pinned GPU3 namespace'):
        t1.main(['--egl-device','3','--max-episodes','4'])


def test_guarded_relay_delegates_original_lifetime_without_changing_arguments(monkeypatch):
    from exp.r17_integration.ops import worker_lifetime
    monkeypatch.setenv('R17_GPU3_DEVICE_ISOLATION','1')
    monkeypatch.setenv('MUJOCO_EGL_DEVICE_ID','3')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    recorded=[];delegated=[]
    monkeypatch.setattr(t1,'record_native_process',lambda:recorded.append(True))
    monkeypatch.setattr(worker_lifetime,'main',lambda argv:delegated.append(argv))
    argv=['--worker-id','same','--egl-device','3','--max-episodes','4','--cache-study']
    t1.main(argv)
    assert recorded==[True] and delegated==[argv]
    for key,value in (('MUJOCO_EGL_DEVICE_ID','0'),('CUDA_VISIBLE_DEVICES','GPU-other')):
        with monkeypatch.context() as wrong:
            wrong.setenv(key,value)
            with pytest.raises(ValueError,match='pinned GPU3 namespace'):t1.main(argv)
    assert recorded==[True] and delegated==[argv]
