"""Large repair controls and bounded multi-shard transfers: CPU, fake nodes."""
import ast
import errno
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from exp.r17_integration.ops import formal_eval as f, library_batch as batch, library_staging as staging
from exp.r17_integration.ops import remote_json as control, formal_process
from exp.r17_integration.ops.build_client_bundle import sha256
from exp.r17_integration.ops.formal_process import write_json
from exp.r17_integration.ops.tether_retry import retry_scope
from exp.r17_integration.tests.ops.test_library_staging import campaign, closed_shard


def large_campaign(tmp_path):
    return dict(library=True,model='cosmos',server_node='h100',run_name='CPU_LARGE_REPAIR',
        runtime=str(tmp_path/'runtime'),server_run=str(tmp_path/'servers'),seconds=900,
        server_campaign=str(tmp_path/'campaign.json'),h100_staging=dict(staging.DEFAULT_POLICY),
        repair_pin={'coordinates':[dict(task_uid='eval:'+str(i),binding='/data/'+('capture_binding_'*40)+str(i),
                                       sha256='a'*64) for i in range(500)]},
        ports=[23360,23361,23362,23379],driver_port=23379,smoke=False)


def test_real_linux_large_argument_repro_and_sha_file_repair_admission(tmp_path,monkeypatch):
    c=large_campaign(tmp_path)
    assert len(repr(c).encode())>131072
    # This is the original failure before any subprocess starts, not a mock.
    with pytest.raises(OSError) as exc:
        subprocess.run([sys.executable,'-c','x='+repr(c)],capture_output=True,check=True)
    assert exc.value.errno==errno.E2BIG
    monkeypatch.setattr(staging.socket,'gethostname',lambda:'weilandserver')
    monkeypatch.setattr(f,'host_python',lambda c:sys.executable)
    monkeypatch.setattr(f,'server_env',lambda *a:dict(CUDA_VISIBLE_DEVICES=''))
    transfers=[];commands=[];references=[]
    def push(argv):
        assert argv[:2]==['tether','push'];transfers.append(argv)
        source=Path(argv[2]);target=tmp_path/'fake_h100'/Path(argv[3].split(':',1)[1]).name
        target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target)
    def remote(node,argv,**kwargs):
        assert node=='h100' and max(len(str(a).encode()) for a in argv)<8192
        commands.append(argv);code=argv[-1]
        if 'staged control JSON SHA mismatch' in code:
            assert len(transfers)==1
            references.append(Path(transfers[0][2]))
            return json.dumps(dict(sha256=sha256(references[-1])))
        assert 's.admit(load_json(' in code
        # Execute the actual SHA-checked file read (a fake node maps /tmp).
        reference=ast.literal_eval(code.split('s.admit(load_json(',1)[1].split('),*',1)[0])
        original=Path(reference['path']);reference['path']=str(tmp_path/'fake_h100'/original.name)
        assert control.load_json(reference)==c
        raise RuntimeError('CPU admission reached; no GPU launch')
    monkeypatch.setattr(f,'remote',remote);monkeypatch.setattr(f,'transfer',push)
    monkeypatch.setattr(f.subprocess,'Popen',lambda *a,**k:pytest.fail('GPU/process launched'))
    directory=tmp_path/'campaign';directory.mkdir()
    assert f.run_campaign(c,directory,{})==2
    assert len(transfers)==1 and len(commands)==2
    assert 'CPU admission reached' in json.loads((directory/'RECEIPT.json').read_text())['error']
    pin=json.loads((directory/'H100_CONTROL_INPUT.json').read_text())
    assert pin['bytes']>131072 and pin['sha256']==sha256(references[0])


@pytest.mark.parametrize('mutation',['bytes','returned_sha','symlink'])
def test_staged_control_is_verified_and_rechecked_at_use(tmp_path,mutation):
    value=large_campaign(tmp_path)
    def push(argv):shutil.copyfile(argv[2],argv[3].split(':',1)[1])
    def remote(node,argv):
        source=next((tmp_path/'control_inputs').glob('*.json'))
        return json.dumps(dict(sha256='0'*64 if mutation=='returned_sha' else sha256(source)))
    if mutation=='returned_sha':
        with pytest.raises(ValueError,match='verification differs'):
            control.stage_json('CPU',value,tmp_path,str(tmp_path),remote=remote,transfer=push)
        return
    ref=control.stage_json('CPU',value,tmp_path,str(tmp_path),remote=remote,transfer=push)
    path=Path(ref['path'])
    if mutation=='bytes':path.write_bytes(b'changed')
    else:
        other=tmp_path/'other.json';path.rename(other);path.symlink_to(other)
    with pytest.raises(ValueError,match='SHA/path'):control.load_json(ref)


def test_probe_reuses_node_session_every_thirty_seconds_and_renews_each_lease(tmp_path,monkeypatch):
    owned=[]
    for node,labels in [('h100',['gpu_lock','replica_0','replica_1','proxy']),('timan108',['workers'])]:
        for label in labels:
            receipt=tmp_path/(node+'_'+label+'.json');lease=receipt.with_suffix('.lease')
            write_json(receipt,dict(status='RUNNING',supervisor=dict(pid=1),child=dict(pid=2)))
            owned.append(dict(node=node,label=label,root=str(tmp_path),python=sys.executable,
                              request=dict(receipt=str(receipt),lease=str(lease))))
    monkeypatch.setattr(formal_process,'alive',lambda p:True)
    clock=[0.];calls=[]
    monkeypatch.setattr(f.time,'monotonic',lambda:clock[0])
    def remote(node,argv,**kw):
        assert kw==dict(seconds=10,replay_safe=True)
        entries=ast.literal_eval(argv[-1].split('print(json.dumps(controls(',1)[1][:-3])
        calls.append((node,clock[0]));return json.dumps(formal_process.controls(entries))
    monkeypatch.setattr(f,'remote',remote)
    cache,files,last={},{},-float('inf')
    for t in range(0,120,2):
        clock[0]=t;last=f.refresh_remote_owned(owned,tmp_path,cache,files,last)
    assert calls==[(node,t) for t in (0,30,60,90) for node in ('h100','timan108')]
    assert len(cache)==5 and all(Path(o['request']['lease']).exists() for o in owned)
    # Old loops at this topology made9 calls/2s +5/26s, new path4/min.
    assert len(calls)/2==4


def test_probe_missing_remote_label_fails_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(f,'remote',lambda *a,**k:'{}')
    entry=dict(node='h100',label='replica_0',root=str(tmp_path),python=sys.executable,request={})
    with pytest.raises(ValueError,match='omitted'):
        f.refresh_remote_owned([entry],tmp_path,{}, {},-float('inf'))


def test_batch_one_pull_for_eight_then_forced_tail_with_per_shard_receipts(campaign,tmp_path):
    campaign['h100_batch']=True
    closed=[closed_shard(campaign,i,i%2) for i in range(9)]
    directory=tmp_path/'weilandserver';calls=[];pulls=[]
    def remote(action,*args):
        calls.append(action);return getattr(staging,action)(campaign,*args)
    def pull(source,target):pulls.append(source);shutil.copyfile(source,target)
    pump=staging.ShardPump(campaign,directory,remote,pull)
    value=pump.drain()
    assert len(pulls)==1 and len(pump.receipts)==8 and value['transfer'] is None
    assert sum(row[1].exists() for row in closed)==1
    # finish normally follows a running thread; use an already-closed CPU
    # pump thread to test the final forced drain without30s real waiting.
    pump.stop.set();pump.start();pump.thread.join(5)
    result=pump.finish()
    assert len(pulls)==2 and result['batch_transfers']==2 and len(result['shards'])==9
    assert result['poll_seconds']==30 and result['high_water_reserved_bytes']<=20_000_000_000
    assert all(not row[1].exists() for row in closed)
    assert not list(staging.TRANSFER_ROOT.rglob('*.tar'))
    sources=[Path(row['receipt']['destination'])/'binding_cpu' for row in result['shards'].values()]
    from exp.r17_integration.ops.library_salvage import verify_received
    verify_received(sources,campaign,result['shards'])
    from exp.r17_integration.debug.attempts import capture_counts
    counts,_=capture_counts(sources,binding=closed[0][2]);assert sum(counts.values())==18
    assert calls.count('batch_next')==5  # prepare, ack+idle, tail, ack+idle, final


@pytest.mark.parametrize('mutation',['archive','receipt','remote_changed','duplicate','destination_symlink'])
def test_batch_corruption_never_releases_any_remote_copy(campaign,tmp_path,mutation):
    closed=[closed_shard(campaign,i) for i in range(4)]
    transfer=batch.next_batch(campaign)['transfer'];assert len(transfer['shards'])==4
    local=tmp_path/'local';local.mkdir();archive=local/'batch.tar';shutil.copyfile(transfer['archive'],archive)
    if mutation=='archive':
        with archive.open('r+b') as stream:stream.write(b'bad')
        with pytest.raises(ValueError,match='SHA mismatch'):batch.install_batch(archive,local,transfer)
    elif mutation=='destination_symlink':
        relative=next(iter(transfer['shards'].values()))['relative'];target=local/'servers'/relative
        target.parent.mkdir(parents=True);other=local/'other';other.mkdir();target.symlink_to(other,target_is_directory=True)
        with pytest.raises(ValueError,match='symlink'):batch.install_batch(archive,local,transfer)
    else:
        ack=batch.install_batch(archive,local,transfer)
        if mutation=='receipt':ack['receipts'][-1]['archive_sha256']='0'*64
        elif mutation=='duplicate':ack['receipts'].append(ack['receipts'][0])
        else:(closed[-1][1]/'manifest.json').write_bytes(b'changed')
        with pytest.raises(ValueError,match='receipt|changed|exactly once'):batch.next_batch(campaign,ack)
    assert all(row[1].exists() for row in closed) and Path(transfer['archive']).exists()
    assert len(json.loads((Path(campaign['server_run'])/'SHARD_BUDGET.json').read_text())['shards'])==4


def test_batch_ack_lost_response_replay_and_salvage_failed_latch(campaign,tmp_path):
    closed=[closed_shard(campaign,i) for i in range(4)]
    transfer=batch.next_batch(campaign)['transfer']
    archive=tmp_path/'batch.tar';shutil.copyfile(transfer['archive'],archive)
    ack=batch.install_batch(archive,tmp_path/'local',transfer)
    from exp.r17_integration.ops.library_salvage import remote_shards
    state_path=Path(campaign['server_run'])/'SHARD_BUDGET.json'
    state=json.loads(state_path.read_text());state['failure']='node_offline: status=STALE';write_json(state_path,state)
    assert remote_shards(campaign,'batch_ack',receipt=ack)['removed_after_verified_receipt']
    assert remote_shards(campaign,'batch_ack',receipt=ack)['already_cleaned']
    assert not any(r[1].exists() for r in closed)
    changed=dict(ack,receipts=ack['receipts'][:-1])
    with pytest.raises(ValueError,match='exact'):remote_shards(campaign,'batch_ack',receipt=changed)


def test_batch_waits_for_cohort_but_flushes_after_two_minutes_and_ignores_open(campaign,tmp_path,monkeypatch):
    closed_shard(campaign,0)
    now=[0.];monkeypatch.setattr(batch.time,'time',lambda:now[0])
    assert batch.next_batch(campaign)['transfer'] is None
    now[0]=119;assert batch.next_batch(campaign)['transfer'] is None
    budget=staging.ShardBudget(campaign)
    shard='a'*64;leaf=budget.root/'replica_1/capture/shards'/shard;budget.reserve(shard,leaf)
    now[0]=120;value=batch.next_batch(campaign)
    assert len(value['transfer']['shards'])==1 and shard not in value['transfer']['shards'] and value['active']==1
    assert Path(value['transfer']['source_archive']).stat().st_size<=511*1024**2


def test_actual_tether_calls_are_counted_without_running_cli(tmp_path,monkeypatch):
    monkeypatch.setattr(f.subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a[0],0,stdout='{}',stderr=''))
    with retry_scope(tmp_path) as journal:
        for node in ('h100','timan108'):
            f.execute(['tether','exec','--safe','--timeout','10s',node,'--','env','CPU=1'])
    summary=json.loads((tmp_path/'TETHER_RETRY_RECEIPT.json').read_text())
    assert summary['call_metrics']['total']==2 and summary['call_metrics']['by_node']=={'h100':1,'timan108':1}
    assert len((tmp_path/'TETHER_CALLS.jsonl').read_text().splitlines())==2


def test_closed_receipt_arrival_replay_measures_reduced_production_call_rate(campaign,tmp_path,monkeypatch):
    """CPU fake-node replay; actual closed receipt mtimes, no live campaign."""
    evidence=Path(__file__).parent/'receipts/TETHER_TRAFFIC_EVIDENCE.json'
    trace=json.loads(evidence.read_text());arrivals=trace['arrival_seconds']
    # Every real group of8 fits the511MiB archive bound. Synthetic payloads
    # produce the same group count without rereading GB of camera arrays.
    assert max(trace['shard_tar_estimate_bytes'])*8+16384<511*1024**2
    campaign['h100_batch']=True
    clock=[0.];calls=[];pulls=[];created=0
    monkeypatch.setattr(f.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(batch.time,'time',lambda:clock[0])
    owned=[dict(node=node,label=label,root=str(tmp_path),python=sys.executable,request={})
           for node,labels in [('h100',['gpu_lock','replica_0','replica_1','proxy']),('timan108',['workers'])]
           for label in labels]
    def lease_remote(node,argv,**kwargs):
        entries=ast.literal_eval(argv[-1].split('print(json.dumps(controls(',1)[1][:-3])
        calls.append(('control',node,clock[0]));return json.dumps({e['label']:dict(status='RUNNING') for e in entries})
    monkeypatch.setattr(f,'remote',lease_remote)
    def shard_remote(action,*args):
        calls.append((action,'h100',clock[0]));return getattr(staging,action)(campaign,*args)
    def pull(source,target):
        calls.append(('pull','h100',clock[0]));pulls.append(str(target));shutil.copyfile(source,target)
    directory=tmp_path/'local';pump=staging.ShardPump(campaign,directory,shard_remote,pull)
    cache,files,last={},{},-float('inf');horizon=math.ceil(arrivals[-1]/30)*30
    for t in range(0,horizon+1,30):
        clock[0]=t
        while created<len(arrivals) and arrivals[created]<=t:
            closed_shard(campaign,created,created%2);created+=1
        last=f.refresh_remote_owned(owned,directory,cache,files,last);pump.drain()
    pump.stop.set();pump.start();pump.thread.join(5);summary=pump.finish()
    assert len(summary['shards'])==54 and summary['batch_transfers']==len(pulls)<20
    rate=len(calls)*60/horizon
    assert rate<10 and trace['old_estimated_total_calls_per_minute']>300
    result=dict(schema='r17.tether_traffic_cpu_replay.v1',evidence=str(evidence),gpu_executed=False,
        real_network_executed=False,source_receipts=54,window_seconds=horizon,
        old_code_receipt_estimated_calls_per_minute=trace['old_estimated_total_calls_per_minute'],
        new_cpu_replay_calls=len(calls),new_cpu_replay_calls_per_minute=rate,
        pulls=len(pulls),shards_per_pull=[len(json.loads((directory/'batch_transfers'/
            (Path(p).name.replace('.tar','.receipt.json'))).read_text())['receipts']) for p in pulls],
        counters={kind:sum(r[0]==kind for r in calls) for kind in {r[0] for r in calls}},
        blip_tolerance_seconds=300,probe_seconds=30,hypothesis_proved=False)
    write_json(Path(__file__).parent/'receipts/TETHER_TRAFFIC_CPU_REPLAY.json',result)
