"""Production backlog shape, bounded pressure and concrete transport failures."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

import pytest

from exp.r17_integration.ops import library_staging as staging
from exp.r17_integration.debug.validate import digest
from exp.r17_integration.tests.ops.test_library_staging import campaign,closed_shard


def wait_pressure(budget):
    deadline=time.monotonic()+2
    while time.monotonic()<deadline:
        state=json.loads(budget.state_path.read_text())
        if state.get('backpressure_events'):return
        time.sleep(.01)
    pytest.fail('reservation did not apply backpressure')


def test_real_19_active_47_tiny_closed_backlog_can_cross_episode_67(campaign):
    budget=staging.ShardBudget(campaign)
    for i in range(19):
        leaf=budget.root/('replica_'+str(i%2))/'capture/shards'/digest(('active',i))
        budget.reserve(leaf.name,leaf,wait_seconds=0)
    closed=[closed_shard(campaign,i,i%2) for i in range(47)]
    # Old code exhausted its66 full256MiB reservations at this next episode.
    for i in range(47,110):closed.append(closed_shard(campaign,i,i%2))
    state=budget.check()
    assert len(state['shards'])==129 and state['failure'] is None
    assert state['high_water_reserved_bytes']<8*staging.GIB
    assert all(e['reserved_bytes']>=e['actual_bytes'] for e in state['shards'].values() if e['status']=='closed')
    assert all(leaf.exists() for _,leaf,*_ in closed)  # Unreceived data retained.


def test_capacity_wait_releases_flock_so_verified_transfer_can_ack(campaign,tmp_path):
    campaign['h100_staging'].update(shard_bytes=2*1024**2,
        limit_bytes=2*staging.GIB+4*1024**2)
    shard_id,leaf,*_=closed_shard(campaign)
    budget=staging.ShardBudget(campaign)
    active=budget.root/'replica_1/capture/shards'/digest('active')
    budget.reserve(active.name,active,wait_seconds=0)
    waiting=budget.root/'replica_0/capture/shards'/digest('waiting')
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(budget.reserve,waiting.name,waiting,wait_seconds=2,poll_seconds=.01)
        wait_pressure(budget)
        assert not future.done() and budget.check()['failure'] is None
        # No deadlock or fatal latch: the pump can use this same quota and ack.
        pump=staging.ShardPump(campaign,tmp_path/'local',lambda op,*args:getattr(staging,op)(campaign,*args),
            lambda src,dst:shutil.copyfile(src,dst))
        pump.drain()
        future.result(timeout=2)
    assert not leaf.exists() and shard_id in pump.receipts
    state=budget.check()
    assert set(state['shards'])=={active.name,waiting.name}
    assert budget.reserved(state)<=budget.limits['limit_bytes'] and state['failure'] is None


def test_capacity_stall_deadline_fails_closed_with_shard_and_bytes(campaign):
    campaign['h100_staging']['limit_bytes']=2*staging.GIB+staging.DEFAULT_POLICY['shard_bytes']
    budget=staging.ShardBudget(campaign)
    for i in range(2):
        leaf=budget.root/'replica_0/capture/shards'/digest(i)
        if i==0:budget.reserve(leaf.name,leaf,wait_seconds=0)
        else:
            with pytest.raises(RuntimeError,match='reserved=.*requested=.*limit=') as error:
                budget.reserve(leaf.name,leaf,wait_seconds=.02,poll_seconds=.01)
            assert leaf.name in str(error.value)
    assert json.loads(budget.state_path.read_text())['failure']


@pytest.mark.parametrize('step',('pending','stage_shard','pull','install_snapshot','partial',
    'persist_receipt','acknowledge_shard','persist_summary','remove_local_archive'))
def test_each_transfer_failure_records_shard_step_and_stderr(campaign,tmp_path,monkeypatch,capsys,step):
    shard_id,leaf,*_=closed_shard(campaign)
    if step=='partial':(leaf/'interrupted.part').write_text('incomplete')
    directory=tmp_path/'local'
    def failed():
        raise subprocess.CalledProcessError(75,['tether','CPU-fixture'],output='partial stdout',stderr='concrete tether stderr')
    def remote(op,*args):
        if op==step:failed()
        return getattr(staging,op)(campaign,*args)
    def pull(src,dst):
        if step=='pull':failed()
        shutil.copyfile(src,dst)
        if step=='install_snapshot':dst.write_bytes(b'corrupt')
    original_atomic=staging.atomic_bytes
    def atomic(path,data):
        if step=='persist_receipt' and str(path).endswith('.receipt.json'):failed()
        if step=='persist_summary' and Path(path).name=='H100_SHARD_TRANSFERS.json':failed()
        return original_atomic(path,data)
    monkeypatch.setattr(staging,'atomic_bytes',atomic)
    original_unlink=Path.unlink
    def unlink(path,*a,**k):
        if step=='remove_local_archive' and path==directory/'shard_transfers'/(shard_id+'.tar'):failed()
        return original_unlink(path,*a,**k)
    monkeypatch.setattr(Path,'unlink',unlink)
    pump=staging.ShardPump(campaign,directory,remote,pull)
    pump.start();pump.thread.join(3)
    assert not pump.thread.is_alive()
    with pytest.raises(RuntimeError,match='shard=.*step=.*cause=') as error:pump.check()
    failure=json.loads((directory/'H100_SHARD_FAILURE.json').read_text())
    expected='install_snapshot' if step=='partial' else step
    assert failure['step']==expected
    assert failure['shard_id']==(None if step=='pending' else shard_id)
    assert str(error.value).endswith('stderr='+failure['stderr'])
    output=capsys.readouterr().err
    assert expected in output and failure['cause'] in output
    if step not in ('install_snapshot','partial'):
        assert 'concrete tether stderr' in failure['stderr'] and 'concrete tether stderr' in output
    else:assert 'SHA' in output or 'partial shard' in output
    if step not in ('persist_summary','remove_local_archive'):assert leaf.exists()
    else:assert not leaf.exists() and (directory/'shard_transfers'/(shard_id+'.receipt.json')).exists()
    assert pump.summary()['status']=='FAIL' and pump.summary()['failure']==failure


def test_remote_command_error_preserves_full_stderr_without_gpu_or_remote_call(monkeypatch):
    from exp.r17_integration.ops import formal_eval
    stderr='CPU traceback: quota shard failure\n'+('detail '*2000)
    monkeypatch.setattr(formal_eval.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout='stdout',stderr=stderr))
    with pytest.raises(RuntimeError) as error:formal_eval.execute(['CPU-fixture'])
    assert error.value.stderr==stderr and error.value.stdout=='stdout' and error.value.returncode==1
