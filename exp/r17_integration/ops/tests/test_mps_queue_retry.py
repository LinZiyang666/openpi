"""CPU-only MPS CLI race repro and sealed, zero-acceptance queue retries."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
from unittest import mock

import pytest

from exp.r17_integration.ops import formal_eval as launch, formal_prestage as stage
from exp.r17_integration.ops import formal_queue as queue, mps_control_guard as guard, queue_retry as retry
from exp.r17_integration.ops.formal_process import write_json
from exp.r17_integration.ops.immutable_release import sha


def fake_control(tmp_path, transient=0, fatal=False):
    pipe = tmp_path/'pipe'
    pipe.mkdir(mode=0o700)
    binary = tmp_path/'native_control'
    binary.write_text('''#!/usr/bin/python3
import fcntl,json,os,sys,time
from pathlib import Path
p=Path(os.environ['CUDA_MPS_PIPE_DIRECTORY'])
with (p/'fake_daemon_lock').open('a') as lock:
 try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  print('concurrent control UI',file=sys.stderr); sys.exit(77)
 counter=p/'native_calls'; n=int(counter.read_text())+1 if counter.exists() else 1
 counter.write_text(str(n)); time.sleep(.03)
 if n <= '''+str(transient)+''' or '''+repr(fatal)+''':
  print('''+repr('real application error' if fatal else 'Cannot send command to MPS control daemon process')+''',file=sys.stderr); sys.exit(1)
 print('311')
''')
    binary.chmod(0o700)
    env = dict(os.environ,CUDA_MPS_PIPE_DIRECTORY=str(pipe),R17_MPS_CONTROL_BINARY=str(binary))
    return pipe,binary,env


def test_parallel_control_queries_are_serialized_and_retry_readiness(tmp_path):
    pipe,_,env = fake_control(tmp_path,transient=2)
    def call(_):
        return subprocess.run(['/usr/bin/python3',str(Path(guard.__file__))],env=env,
                              input=b'get_server_list\n',capture_output=True,timeout=10)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(call,range(6)))
    assert all(r.returncode==0 and r.stdout==b'311\n' for r in results)
    assert int((pipe/'native_calls').read_text()) == 8
    records = [json.loads(s) for s in (pipe/'r17_control_calls.jsonl').read_text().splitlines()]
    assert len(records)==6 and sorted(r['attempts'] for r in records)==[1,1,1,1,1,3]
    assert sum(len(r['backoff_seconds']) for r in records)==2


def test_unguarded_parallel_queries_reproduce_failure(tmp_path):
    _,binary,env = fake_control(tmp_path)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _:subprocess.run([str(binary)],env=env,input=b'get_server_list\n',
                                                capture_output=True,timeout=10),range(6)))
    assert any(r.returncode==77 for r in results)


def test_control_lock_is_per_campaign_not_shared_between_daemons(tmp_path):
    environments=[]
    for name in ('one','two'):
        root=tmp_path/name;root.mkdir()
        environments.append(fake_control(root)[2])
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda env:guard.forward([],b'get_server_list\n',env=env),environments))
    assert all(r.returncode==0 for r in results)
    assert all((Path(e['CUDA_MPS_PIPE_DIRECTORY'])/'r17_control.lock').is_file() for e in environments)


@pytest.mark.parametrize('argv,data', [([],b'quit\n'),(['-d'],b''),([],b'get_client_list 311\nquit\n')])
def test_mutating_or_multiline_commands_are_not_replayed(tmp_path,argv,data):
    pipe,_,env = fake_control(tmp_path,transient=10)
    result = guard.forward(argv,data,env=env)
    assert result.returncode==1 and int((pipe/'native_calls').read_text())==1


def test_fatal_error_is_not_retried(tmp_path):
    pipe,_,env = fake_control(tmp_path,fatal=True)
    result = guard.forward([],b'get_server_list\n',env=env)
    assert result.returncode==1 and int((pipe/'native_calls').read_text())==1


def test_readiness_retry_is_bounded(tmp_path):
    pipe,_,env = fake_control(tmp_path,transient=100)
    result = guard.forward([],b'get_server_list\n',env=env,bound=.25)
    assert result.returncode!=0 and int((pipe/'native_calls').read_text())<6


def test_private_pipe_lock_wait_is_bounded_and_permissions_checked(tmp_path):
    pipe,_,env = fake_control(tmp_path)
    with (pipe/'r17_control.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert guard.forward([],b'get_server_list\n',env=env,bound=.1).returncode==124
    assert not (pipe/'native_calls').exists()
    pipe.chmod(0o755)
    with pytest.raises(ValueError,match='ownership/permissions'):
        guard.forward([],b'get_server_list\n',env=env)


def test_remote_install_is_by_value_checksum_checked_and_does_not_overlay_runtime(tmp_path,monkeypatch):
    _,binary,_ = fake_control(tmp_path)
    (tmp_path/'nvidia-cuda-mps-control').symlink_to(binary)
    monkeypatch.setenv('PATH',str(tmp_path))
    source = Path(guard.__file__).read_text()
    namespace = {}
    exec(inspect.getsource(guard.install),namespace)
    request = dict(directory=str(tmp_path/'orchestration'),source=source,sha256=hashlib.sha256(source.encode()).hexdigest())
    installed = namespace['install'](request)
    assert installed['binary']==str(binary)
    assert sha(installed['script'])==request['sha256']
    assert installed['model_worker_runtime_unchanged']
    with pytest.raises(ValueError,match='checksum'):
        namespace['install'](dict(request,sha256='bad',directory=str(tmp_path/'bad')))
    assert not (tmp_path/'bad').exists()


def prepared(tmp_path,name='arm'):
    p = tmp_path/name
    p.mkdir()
    (p/'assets').mkdir()
    campaign = dict(run_name=name,prepare_only=True,model='fastwam',domain='libero',
        checkpoint='/immutable/weights',server_node='local',runtime='/immutable/code',runtime_identity='pin',
        runtime_from='/sealed/control',local_assets=str(p/'assets'),assets=str(p/'assets'),server_run=str(p/'servers'),
        worker_nodes={'timan107':24},workers=24,replicas=2,replica_startup_width=2,mps={'enabled':True},
        fastwam_batching={'max_batch':16},denoising_steps=10,commit_horizon=16,
        worker_max_episodes=4,worker_restart_limit=2,worker_egl_devices=list(range(8)),seconds=3600)
    write_json(p/'campaign.json',campaign)
    for n in ('PREFLIGHT.json','WORKER_DEPLOYMENT.json','timan107.json','RUNTIME_RELEASE.json','FROZEN_RUNTIME.json'):
        write_json(p/n,{'status':'PASS'})
    write_json(p/'EPISODE_PLAN.json',dict(tasks=[dict(task_uid='fixed_uid',seed=33,init=0)],expected_episodes=1))
    for n in ('model_manifest.json','model_spec.json','benchmark.json','full.yaml','manifest.json'):
        (p/'assets'/n).write_text('immutable native '+n)
    (p/'runtime.tar').write_bytes(b'frozen runtime')
    (p/'workers.tar').write_bytes(b'frozen workers')
    stage.seal_prepared(campaign,p,{'timan107':{}})
    return p,campaign


def failure(p,accepted=False,closed=True):
    write_json(p/'RECEIPT.json',dict(status='FAIL',all_owned_closed=closed,error='MPS Cannot send command'))
    if accepted:
        (p/'driver').mkdir(exist_ok=True)
        (p/'driver/journal.jsonl').write_text(json.dumps(dict(accepted=True,status='done'))+'\n')


def test_capture_and_clone_preserve_native_settings_and_plan(tmp_path,monkeypatch):
    p,c = prepared(tmp_path)
    snapshot = retry.capture(p,p/'.startup_inputs')
    native = {n:sha(p/n) for n in ('runtime.tar','workers.tar','EPISODE_PLAN.json','assets/full.yaml','assets/model_manifest.json')}
    failure(p)
    monkeypatch.setattr(launch,'FORMAL_ROOT',tmp_path)
    def deploy(campaign,directory):
        write_json(directory/'WORKER_DEPLOYMENT.json',{'timan107':{'worker_run':'/tmp/'+campaign['run_name']}})
        write_json(directory/'timan107.json',campaign)
        return {'timan107':{}}
    with mock.patch('exp.r17_integration.ops.formal_runtime.check_pins'),mock.patch.object(launch,'prepare_workers',side_effect=deploy):
        fresh = retry.clone(snapshot,'arm_r1',retry.zero_acceptance(p))
    candidate,_ = stage.validate_prepared(fresh)
    assert {n:sha(fresh/n) for n in native}==native
    for key in ('checkpoint','mps','runtime_identity','replicas','replica_startup_width','fastwam_batching',
                'worker_max_episodes','worker_restart_limit','denoising_steps','commit_horizon','worker_egl_devices'):
        assert candidate[key]==c[key]
    assert candidate['run_name']=='arm_r1' and candidate['local_assets']!=c['local_assets']
    assert json.loads((p/'RECEIPT.json').read_text())['status']=='FAIL'
    assert 'STARTUP_RETRY.json' in json.loads((fresh/'PREPARED.json').read_text())['files']


def test_snapshot_or_native_changes_refuse_retry(tmp_path,monkeypatch):
    p,_ = prepared(tmp_path)
    snapshot = retry.capture(p,p/'.startup_inputs')
    (snapshot/'workers.tar').write_bytes(b'changed code')
    with pytest.raises(ValueError,match='snapshot changed'):
        retry.validate_inputs(snapshot)


@pytest.mark.parametrize('name',['../outside','bad name',''])
def test_retry_run_paths_cannot_escape_owned_formal_root(tmp_path,name):
    p,_=prepared(tmp_path);snapshot=retry.capture(p,p/'.startup_inputs')
    with pytest.raises(ValueError,match='plain run identifier'):
        retry.clone(snapshot,name,{})


def test_first_control_retry_inherits_captured_native_release_not_checkout(tmp_path,monkeypatch):
    p,c=prepared(tmp_path)
    c.pop('runtime_from');c.pop('fastwam_batching')
    write_json(p/'campaign.json',c);stage.seal_prepared(c,p,{'timan107':{}})
    snapshot=retry.capture(p,p/'.startup_inputs');failure(p)
    monkeypatch.setattr(launch,'FORMAL_ROOT',tmp_path)
    def attach(campaign,source):
        campaign['runtime_from']=str(source.resolve())
    def deploy(campaign,directory):
        assert campaign['runtime_from']==str(snapshot.resolve())
        assert sha(directory/'workers.tar')==sha(snapshot/'workers.tar')
        return {'timan107':{}}
    with mock.patch('exp.r17_integration.ops.formal_runtime.attach',side_effect=attach) as bound, \
         mock.patch('exp.r17_integration.ops.formal_runtime.check_pins'), \
         mock.patch('exp.r17_integration.ops.immutable_release.install') as installed, \
         mock.patch.object(launch,'prepare_workers',side_effect=deploy):
        fresh=retry.clone(snapshot,'control_r1',retry.zero_acceptance(p))
    bound.assert_called_once();installed.assert_called_once()
    assert sha(fresh/'runtime.tar')==sha(snapshot/'runtime.tar')
    assert json.loads((fresh/'campaign.json').read_text())['coordinator_runtime']==c['runtime']


@pytest.mark.parametrize('kind',['accepted','unclosed','malformed','driver_no_journal','summary_accepted','accepted_error_attempt'])
def test_retry_refuses_accepted_or_unproven_work(tmp_path,kind):
    p,_ = prepared(tmp_path)
    failure(p,accepted=kind=='accepted',closed=kind!='unclosed')
    if kind=='malformed':
        (p/'driver').mkdir();(p/'driver/journal.jsonl').write_text('{partial')
    if kind=='driver_no_journal':
        r=json.loads((p/'RECEIPT.json').read_text());r['owned_local']=[{'label':'driver'}];write_json(p/'RECEIPT.json',r)
    if kind=='summary_accepted':write_json(p/'SUMMARY.json',{'completed_episodes':1})
    if kind=='accepted_error_attempt':
        (p/'driver').mkdir();(p/'driver/attempts.jsonl').write_text('{"accepted":true,"error":"EGL"}\n')
    with pytest.raises((ValueError,RuntimeError)):
        retry.zero_acceptance(p)


def test_queue_retries_fresh_copies_and_records_attempts(tmp_path,monkeypatch):
    p,_ = prepared(tmp_path)
    retry.capture(p,p/'.startup_inputs')
    job=dict(name='arm',run=str(p),args=argparse.Namespace(seconds=1))
    calls=[];clones=[]
    class Runner:
        stopped=False
        def event(self,**kw):pass
        def invoke(self,job,phase,argv):
            calls.append(job['run']);run=Path(job['run'])
            if len(calls)<3:
                failure(run);raise RuntimeError('startup failure')
            write_json(run/'RECEIPT.json',dict(status='PASS',all_owned_closed=True))
    def clone(snapshot,name,proof):
        clones.append((snapshot,name,proof));fresh,_=prepared(tmp_path,name);return fresh
    with mock.patch.object(retry,'clone',side_effect=clone):
        result=retry.run_job(job,Runner(),limit=2,delay=0)
    assert len(calls)==3 and len(set(calls))==3 and len(clones)==2
    assert result['name']=='arm' and len(result['startup_retries'])==2
    assert result['run']!=str(p)


def test_queue_retry_limit_and_closed_pass_bypass(tmp_path):
    p,_=prepared(tmp_path);retry.capture(p,p/'.startup_inputs')
    job=dict(name='arm',run=str(p),args=argparse.Namespace(seconds=1))
    class Runner:
        stopped=False
        def invoke(self,*args):failure(p);raise RuntimeError('bounded fail')
    with mock.patch.object(retry,'clone') as clone:
        with pytest.raises(RuntimeError,match='bounded fail'):retry.run_job(job,Runner(),limit=0,delay=0)
        clone.assert_not_called()
    class Good:
        def invoke(self,*args):write_json(p/'RECEIPT.json',dict(status='PASS',all_owned_closed=True))
    assert retry.run_job(job,Good())['startup_retries']==[]


def test_two_retry_limit_stops_after_three_closed_zero_acceptance_failures(tmp_path):
    p,_=prepared(tmp_path);retry.capture(p,p/'.startup_inputs')
    job=dict(name='arm',run=str(p),args=argparse.Namespace(seconds=1))
    calls=[]
    class Runner:
        stopped=False
        def event(self,**kw):pass
        def invoke(self,job,*args):
            calls.append(job['run']);failure(Path(job['run']));raise RuntimeError('real startup failure')
    def clone(snapshot,name,proof):
        return prepared(tmp_path,name)[0]
    with mock.patch.object(retry,'clone',side_effect=clone) as cloned:
        with pytest.raises(RuntimeError,match='real startup failure'):
            retry.run_job(job,Runner(),limit=2,delay=0)
    assert len(calls)==3 and len(set(calls))==3 and cloned.call_count==2


def test_launch_preserves_inputs_before_one_shot_claim(tmp_path,monkeypatch):
    p,c=prepared(tmp_path)
    c['server_node']='local'
    # Stop just after the exclusive claim, before any GPU/remote operation.
    original_validation=stage.validate_prepared
    monkeypatch.setattr(launch,'FORMAL_ROOT',tmp_path)
    with mock.patch('exp.r17_integration.ops.worker_disk.prepared',return_value={'timan107':{'admitted':True}}), \
         mock.patch('exp.r17_integration.ops.formal_runtime.check_pins',side_effect=ValueError('CPU boundary')):
        with pytest.raises(ValueError,match='CPU boundary'):stage.launch_prepared(p)
    assert (p/'LAUNCH.json').exists()
    snapshot=p/'.startup_inputs'
    before,_=retry.validate_inputs(snapshot)
    assert before['prepare_only'] is True
    assert sha(snapshot/'assets/model_manifest.json')==sha(p/'assets/model_manifest.json')
    assert json.loads((p/'RECEIPT.json').read_text())['all_owned_closed'] is True
    with pytest.raises(ValueError,match='already launched'):original_validation(p)


def test_continuation_waits_for_old_queue_closure_and_skips_pass(tmp_path,monkeypatch):
    p,_=prepared(tmp_path,'complete');write_json(p/'RECEIPT.json',dict(status='PASS',all_owned_closed=True))
    failed,_=prepared(tmp_path,'failed');retry.capture(failed,failed/'.startup_inputs');failure(failed)
    pending,_=prepared(tmp_path,'pending')
    source=tmp_path/'queue';source.mkdir()
    write_json(source/'QUEUE_PLAN.json',dict(jobs=[dict(name=p.name,run=str(p)) for p in (p,failed,pending)]))
    write_json(source/'QUEUE_RECEIPT.json',dict(status='FAIL',cpu_only=False,all_owned_closed=True))
    write_json(source/'QUEUE_OWNED.json',[{'pid':123,'start_ticks':'7'}])
    monkeypatch.setattr(launch,'FORMAL_ROOT',tmp_path)
    with mock.patch.object(retry,'alive',side_effect=[True,False]),mock.patch.object(retry.time,'sleep') as sleep:
        jobs=retry.continued_jobs(source)
    assert sleep.call_count==1 and [j['name'] for j in jobs]==['failed','pending']
    assert jobs[0]['initial_failure'] is True and not jobs[1].get('initial_failure')


def test_cli_retry_bounds_and_continuation_are_explicit_run_only():
    for args in (['--jobs','unused','--run','--startup-retries','4'],
                 ['--continue-queue','unused','--prepare-only']):
        with pytest.raises(SystemExit):queue.main(args)
