"""Deterministic CPU transport failures; no tether/GPU/process signals."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from exp.r17_integration.ops import formal_eval, library_staging as staging, tether_retry as tr
from exp.r17_integration.ops.deploy import tether_transfer
from exp.r17_integration.ops.formal_process import lease_expired
from exp.r17_integration.tests.ops.test_library_staging import campaign, closed_shard

OFFLINE = 'error: exec: the agent is OFFLINE (no recent heartbeat). (node_offline: status=STALE)\n'


class Clock:
    def __init__(self):
        self.value, self.delays = 0, []
    def now(self):
        return self.value
    def sleep(self, delay):
        self.delays.append(delay)
        self.value += delay


def error(code=64, stderr=OFFLINE):
    return subprocess.CalledProcessError(code, ['tether', 'CPU-fixture'], 'stdout', stderr)


@pytest.mark.parametrize('code,stderr', (
    (64, OFFLINE), (64, 'error: exec: OFFLINE (status=STALE)'),
    (64, 'error: pull: relay_unavailable'), (64, 'error: push: transport_timeout'),
    (64, 'error: exec: connection_lost'), (75, 'too_many_in_flight'),
))
def test_transport_recovers_with_backoff_and_each_blip_in_receipt(tmp_path, code, stderr):
    clock, calls = Clock(), []
    def run():
        calls.append(True)
        if len(calls) < 3:
            raise error(code, stderr)
        return 'complete'
    with tr.retry_scope(tmp_path):
        assert tr.retry(run,label='renew',node='timan1',clock=clock.now,sleeper=clock.sleep,replay_safe=True)=='complete'
    receipt=json.loads((tmp_path/'TETHER_RETRY_RECEIPT.json').read_text())
    assert calls==[True]*3 and clock.delays==[2,4]
    assert receipt['blips']==2 and receipt['window_seconds']==300
    assert [r['event'] for r in receipt['events']]==['blip','blip','recovered']
    assert all(r['stderr']==stderr and r['node']=='timan1' for r in receipt['events'][:2])
    assert receipt['events']==[json.loads(r) for r in (tmp_path/'TETHER_BLIPS.jsonl').read_text().splitlines()]


@pytest.mark.parametrize('failure', (
    error(1, OFFLINE), error(64,'error: exec: remote command exited1: EGL STALE'),
    error(75,'unrelated'), error(124,'remote execution deadline exceeded'),
    subprocess.TimeoutExpired(['tether'],10), ValueError('SHA mismatch'),
))
def test_real_command_ambiguous_timeout_and_sha_errors_never_retry(failure):
    calls,clock=[],Clock()
    def run():
        calls.append(True); raise failure
    with pytest.raises(type(failure)):
        tr.retry(run,label='CPU',clock=clock.now,sleeper=clock.sleep)
    assert calls==[True] and clock.delays==[]


def test_sustained_offline_expires_at_five_minutes_and_outer_layer_does_not_retry(tmp_path):
    clock,calls,events=Clock(),[],[]
    def run():
        calls.append(clock.now()); raise error()
    with tr.retry_scope(tmp_path), pytest.raises(subprocess.CalledProcessError) as failed:
        tr.retry(lambda:tr.retry(run,label='inner',clock=clock.now,sleeper=clock.sleep,on_event=events.append),
                 label='outer',clock=clock.now,sleeper=clock.sleep,on_event=events.append)
    assert failed.value.stderr==OFFLINE and failed.value.r17_retry_exhausted
    assert clock.now()==300 and max(clock.delays)<=30
    assert len(calls)>8 and all(r['operation']=='inner' for r in events)
    assert events[-1]['event']=='expired'


@pytest.mark.parametrize('mode',('renew','status','stop','worker_snapshot'))
def test_formal_remote_lease_close_snapshot_paths_recover(monkeypatch,tmp_path,mode):
    clock,calls=Clock(),[]
    monkeypatch.setattr(tr.time,'sleep',clock.sleep)
    monkeypatch.setattr(tr.time,'monotonic',clock.now)
    def execute(argv,seconds):
        calls.append((argv,seconds))
        if len(calls)==1:raise error()
        return '{"complete":true}'
    monkeypatch.setattr(formal_eval,'execute',execute)
    argv=['python3','-m','exp.r17_integration.ops.formal_process',mode]
    with tr.retry_scope(tmp_path):
        assert json.loads(formal_eval.remote('timan1',argv,seconds=10))['complete']
    assert len(calls)==2 and calls[0]==calls[1] and clock.delays==[2]
    assert json.loads((tmp_path/'TETHER_RETRY_RECEIPT.json').read_text())['blips']==1


def test_transfer_retry_preserves_default_saturation_contract_and_real_failures(tmp_path):
    clock,calls=Clock(),[]
    def runner(argv,**unused):
        calls.append(argv)
        return SimpleNamespace(returncode=64 if len(calls)<3 else 0,stdout='',stderr=OFFLINE)
    with tr.retry_scope(tmp_path):
        result=tether_transfer(['tether','pull','h100:/tmp/CPU','/tmp/CPU'],lane='tether',runner=runner,
            retry_seconds=300,clock=clock.now,sleeper=clock.sleep)
    assert result==dict(status='ok',attempts=3,lane='tether') and clock.delays==[2,4]
    default_calls=[]
    def default_runner(*a,**k):
        default_calls.append(True)
        return SimpleNamespace(returncode=64,stdout='',stderr=OFFLINE)
    # Default-off API behavior is still a single failed attempt for exit64.
    with pytest.raises(subprocess.CalledProcessError):
        tether_transfer(['tether'],lane='tether',runner=default_runner)
    assert default_calls==[True]
    with pytest.raises(subprocess.CalledProcessError):
        tether_transfer(['tether'],lane='tether',retry_seconds=300,clock=clock.now,sleeper=clock.sleep,
            runner=lambda *a,**k:SimpleNamespace(returncode=1,stdout='',stderr='real remote failure'))
    assert clock.delays==[2,4]


def test_legacy_lease_120_seconds_and_new_launch_600_seconds(tmp_path):
    lease=tmp_path/'lease';lease.touch();os.utime(lease,(100,100))
    request=dict(lease=str(lease))
    assert not lease_expired(request,now=220)
    assert lease_expired(request,now=221)
    request['lease_seconds']=tr.LEASE_SECONDS
    assert not lease_expired(request,now=100+25+tr.RETRY_SECONDS+60)
    assert lease_expired(request,now=701)
    with pytest.raises(ValueError):lease_expired(dict(request,lease_seconds=601))
    assert lease_expired(dict(request,lease=str(tmp_path/'missing')),now=100)


def test_background_shard_stage_pull_ack_blips_recover_without_losing_or_duplicating(campaign,tmp_path,monkeypatch):
    shard,leaf,*_=closed_shard(campaign)
    clock,calls=Clock(),{}
    monkeypatch.setattr(tr.time,'sleep',clock.sleep)
    monkeypatch.setattr(tr.time,'monotonic',clock.now)
    def remote(action,*args):
        calls[action]=calls.get(action,0)+1
        if calls[action]==1:raise error()
        result=getattr(staging,action)(campaign,*args)
        if action=='acknowledge_shard':pump.stop.set()
        return result
    def pull(source,target):
        calls['pull']=calls.get('pull',0)+1
        if calls['pull']==1:raise error(64,'error: pull: node_offline: status=STALE')
        shutil.copyfile(source,target)
    directory=tmp_path/'local'
    with tr.retry_scope(directory):
        pump=staging.ShardPump(campaign,directory,remote,pull)
        pump.start();pump.thread.join(5)
        assert not pump.thread.is_alive()
        assert pump.failure is None
        result=pump.finish()
    assert not leaf.exists() and list(result['shards'])==[shard]
    assert result['shards'][shard]['receipt']['status']=='verified'
    assert len(list((directory/'shard_transfers').glob('*.receipt.json')))==1
    assert not list((directory/'shard_transfers').glob('*.tar'))
    assert result['high_water_reserved_bytes']<=20_000_000_000
    receipt=json.loads((directory/'TETHER_RETRY_RECEIPT.json').read_text())
    assert receipt['blips']==4
    blips=[r for r in receipt['events'] if r['event']=='blip']
    assert {r['context']['step'] for r in blips}=={'pending','stage_shard','pull','acknowledge_shard'}
    assert all(r['context']['shard_id']==shard for r in blips if r['context']['step']!='pending')


@pytest.mark.parametrize('recovers',(True,False))
def test_failed_campaign_cleanup_retries_and_records_provable_closure(tmp_path,monkeypatch,recovers):
    # Fail BEFORE any resident starts. The only owned process is a fake guard.
    campaign=dict(library=True,model='fastwam',server_node='h100',run_name='CPU',
        runtime=str(tmp_path/'runtime'),server_run=str(tmp_path/'servers'),seconds=900,
        server_campaign=str(tmp_path/'campaign.json'),h100_staging=dict(staging.DEFAULT_POLICY),
        ports=[23340,23341,23342,23359],driver_port=23359,smoke=False)
    monkeypatch.setattr(staging.socket,'gethostname',lambda:'weilandserver')
    class Pump:
        failure=None
        def __init__(self,*a):pass
        def start(self):pass
        def check(self):pass
        def abort(self):pass
        def summary(self):return {}
    monkeypatch.setattr(staging,'ShardPump',Pump)
    clock,stops=Clock(),[]
    monkeypatch.setattr(tr.time,'sleep',clock.sleep)
    monkeypatch.setattr(tr.time,'monotonic',clock.now)
    def execute(argv,seconds=60,**kwargs):
        assert argv[:2]==['tether','exec']
        if 'exp.r17_integration.ops.formal_process' in argv:
            mode=argv[argv.index('exp.r17_integration.ops.formal_process')+1]
            request=json.loads(argv[-1])
            assert request['lease_seconds']==600
            if mode=='spawn':return json.dumps(dict(child={'pid':12345}))
            if mode=='status':return json.dumps(dict(status='RUNNING'))
            if mode=='renew':return '{}'
            assert mode=='stop'
            stops.append(True)
            if not recovers or len(stops)==1:raise error()
            return json.dumps(dict(status='CLOSED',alive=False,child_alive=False,supervisor_alive=False,
                                   error='earlier supervisor error'))
        code=argv[-1]
        if 'controls(' in code:return json.dumps(dict(gpu_lock=dict(status='RUNNING')))
        if 's.admit(' in code:return json.dumps({'status':'PASS'})
        if 'GPU_LOCK_READY.json' in code:
            return json.dumps(dict(status='PASS',campaign='CPU',pid=12345,lock='/tmp/r17_gpu.lock'))
        assert 'sockets' in code
        raise error(1,'CPU fixture real port conflict')
    monkeypatch.setattr(formal_eval,'execute',execute)
    monkeypatch.setattr(formal_eval.subprocess,'Popen',lambda *a,**k:pytest.fail('process/GPU launch forbidden'))
    assert formal_eval.run_campaign(campaign,tmp_path/'campaign',{})==2
    receipt=json.loads((tmp_path/'campaign/RECEIPT.json').read_text())
    assert receipt['status']=='FAIL' and 'real port conflict' in receipt['error']
    assert receipt['all_owned_closed'] is recovers
    assert receipt['tether_retry']['blips']==len(stops)-int(recovers)
    if recovers:
        assert len(stops)==2 and clock.now()==2 and receipt['cleanup'][0]['alive'] is False
        assert receipt['cleanup'][0]['closure_confirmed'] and receipt['cleanup'][0]['error']=='earlier supervisor error'
    else:
        assert clock.now()==300 and receipt['cleanup'][0]['error']


def test_h100_server_accepts_either_worker_node():
    from exp.r17_integration.ops.library_campaign import parse_args
    # Legacy input metadata is parsed here; shared preparation removes the
    # disabled node from execution targets before any transport.
    for worker in ('timan107','timan108'):
        args=parse_args(['--model','fastwam_libero','--suite','libero_10','--run',
            '--server-node','h100','--worker-node',worker,'--replicas','2','--workers','24',
            '--memory-budget-gib','64','--port','23340','--driver-port','23359'])
        assert args.server_node=='h100' and args.worker_node==worker and args.replicas==2


def test_capacity_backpressure_allows_a_four_minute_transport_outage(campaign,tmp_path,monkeypatch):
    campaign['h100_staging'].update(shard_bytes=2*1024**2,limit_bytes=2*staging.GIB+4*1024**2)
    shard,leaf,*_=closed_shard(campaign)
    budget=staging.ShardBudget(campaign)
    active=budget.root/'replica_1/capture/shards'/('a'*64)
    budget.reserve(active.name,active,wait_seconds=0)
    clock=Clock()
    pump=staging.ShardPump(campaign,tmp_path/'local',lambda op,*args:getattr(staging,op)(campaign,*args),
        lambda source,target:shutil.copyfile(source,target))
    def sleep(delay):
        clock.sleep(delay)
        if clock.now()>=240 and leaf.exists():pump.drain()
    monkeypatch.setattr(tr.time,'sleep',sleep)
    monkeypatch.setattr(tr.time,'monotonic',clock.now)
    waiting=budget.root/'replica_0/capture/shards'/('b'*64)
    budget.reserve(waiting.name,waiting,poll_seconds=5)
    assert clock.now()==240 and not leaf.exists() and shard in pump.receipts
    assert budget.check()['failure'] is None and budget.reserved(budget.check())<=20_000_000_000


def test_formal_worker_payload_includes_retry_module_and_imports_without_model_or_simulator(tmp_path):
    from exp.r17_integration.ops.deploy import install_payload
    from exp.r17_integration.ops.build_client_bundle import sha256
    formal_eval.prepare_workers(dict(worker_nodes={}),tmp_path)
    destination=tmp_path/'client'
    manifest=install_payload(tmp_path/'workers.tar',destination,sha256(tmp_path/'workers.tar'))
    assert 'exp/r17_integration/ops/tether_retry.py' in manifest['files']
    code='import sys;sys.path.insert(0,'+repr(str(destination))+');from exp.r17_integration.ops import tether_retry,formal_process;assert not any(m in sys.modules for m in ("torch","jax","mujoco","lerobot","gr00t"));assert tether_retry.retry(lambda:42,label="CPU")==42'
    result=subprocess.run([sys.executable,'-I','-c',code],cwd=destination,capture_output=True,text=True,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('mode',('renew','status','stop'))
@pytest.mark.parametrize('transport',('rpc_deadline','outer_client_timeout'))
def test_idempotent_remote_controls_retry_transient_client_timeouts(monkeypatch,tmp_path,mode,transport):
    clock,calls=Clock(),[]
    monkeypatch.setattr(tr.time,'sleep',clock.sleep)
    monkeypatch.setattr(tr.time,'monotonic',clock.now)
    def execute(argv,seconds):
        calls.append(argv)
        if len(calls)==1:
            if transport=='rpc_deadline':raise error(64,'error: exec: context deadline exceeded')
            raise subprocess.TimeoutExpired(argv,seconds)
        return '{"alive":false}'
    monkeypatch.setattr(formal_eval,'execute',execute)
    with tr.retry_scope(tmp_path):
        result=formal_eval.remote('timan1',['python3','-m','exp.r17_integration.ops.formal_process',mode],seconds=10)
    assert json.loads(result)['alive'] is False and len(calls)==2
    assert json.loads((tmp_path/'TETHER_RETRY_RECEIPT.json').read_text())['blips']==1


def test_ambiguous_spawn_response_is_never_reexecuted(monkeypatch):
    calls=[]
    def execute(argv,seconds):
        calls.append(argv);raise error(64,'error: exec: connection_lost')
    monkeypatch.setattr(formal_eval,'execute',execute)
    with pytest.raises(subprocess.CalledProcessError):
        formal_eval.remote('h100',['python3','-m','exp.r17_integration.ops.formal_process','spawn'])
    assert len(calls)==1


def test_lost_shard_ack_reply_replays_only_exact_verified_receipt(campaign,tmp_path,monkeypatch):
    shard,leaf,*_=closed_shard(campaign)
    clock,acks=Clock(),[]
    monkeypatch.setattr(tr.time,'sleep',clock.sleep)
    monkeypatch.setattr(tr.time,'monotonic',clock.now)
    def remote(action,*args):
        result=getattr(staging,action)(campaign,*args)
        if action=='acknowledge_shard':
            acks.append(result)
            if len(acks)==1:raise error(64,'error: exec: connection reset by peer')
        return result
    directory=tmp_path/'local'
    with tr.retry_scope(directory):
        pump=staging.ShardPump(campaign,directory,remote,lambda source,target:shutil.copyfile(source,target))
        pump.drain()
    assert not leaf.exists() and len(acks)==2 and acks[-1]['already_cleaned']
    assert pump.failure is None and list(pump.receipts)==[shard]
    receipt=pump.receipts[shard]['receipt']
    assert json.loads((staging.ShardBudget(campaign).root/'SHARD_RECEIPTS'/(shard+'.json')).read_text())==receipt
    assert json.loads((directory/'TETHER_RETRY_RECEIPT.json').read_text())['blips']==1
    with pytest.raises(ValueError,match='differs'):
        staging.acknowledge_shard(campaign,shard,dict(receipt,archive_sha256='0'*64))
    # Receipt tampering is a real integrity error and latches the quota closed.
    with pytest.raises(RuntimeError,match='capture stopped'):
        staging.acknowledge_shard(campaign,'a'*64,receipt)
