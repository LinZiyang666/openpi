"""No sockets/process services: reproduce real key reuse through Store + sink.

Also exercise the real stock scheduler as a pure in-process data structure.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import ModuleType
from unittest.mock import patch

from .dispatch_fence import DispatchFence, install as install_fence
from .repair_stream import prepare, install, inventory
from .stream_collect import replay_spills, verify_arm
from .stream_protocol import attempt_key
from .stream_receiver import Store
from .stream_sink import StreamSink


def line(uid,ev,**kw):
    return json.dumps(dict(task_uid=uid,attempt=1,ev=ev,**kw),separators=(',',':'))+'\n'


def run_tests(out):
    out.mkdir(parents=True,exist_ok=False)
    run=out/'run';run.mkdir();(run/'arms.json').write_text('[{"arm":"test"}]')
    uid='test:eval:4:1';key=attempt_key(uid,1);root=run/'runs/test'
    store=Store(run);identity=dict(run=run.name,arm='test',uid=uid,attempt=1,key=key)
    prefix=line(uid,'attempt_start')
    old=(prefix+line(uid,'reset',rng=[1,2,3])+line(uid,'profile_error',error='connection lost')+line(uid,'rollout_error')).encode()
    old_info=dict(bytes=len(old),sha256=hashlib.sha256(old).hexdigest())
    store.apply(dict(**identity,op='data',file='controls.jsonl',offset=0),old)
    store.apply(dict(**identity,op='finish',file='controls.jsonl',**old_info))
    store.apply(dict(**identity,op='attempt',files={'controls.jsonl':old_info}))
    marker=dict(p3_step=0,p3_anchor=True,source='cache',parent_anchor=0,commit_controls=10)
    timing=dict(env_ms=19.25,infers=1,steps=1,termination_reason='success')
    new=prefix+line(uid,'reset',rng=[4,5,6])+line(uid,'decision',decision_step=0,marker=marker)+line(uid,'rollout_end',success=True,timing=timing)
    # No socket/listener: exercise identical sender retry/spill control flow
    # against the receiver's real durable Store implementation in-process.
    def send(self,event):
        return store.apply(*event)
    with patch.object(StreamSink,'_send',send):
        sink=StreamSink('127.0.0.1:1','test',run.name,'test',uid,1,root/'client_spills',fail_seconds=.01,timeout=.01,close_seconds=.25)
        sink.write(prefix)
        deadline=time.monotonic()+2
        while sink.stats['frames_acked']<1:
            assert time.monotonic()<deadline;time.sleep(.005)
        sink.write(new[len(prefix):]);sink.write_file('step_000000.npz',b'exact binary snapshot\0\xff');sink.close()
    assert sink.spilled and sink.stats['frames_acked']==1
    try:replay_spills(run,'test',root/'client_spills')
    except ValueError as e:assert str(e)=='conflicting duplicate bytes'
    else:raise AssertionError('did not reproduce pilot collision')
    (root/'client').mkdir()
    journal=dict(task_uid=uid,attempt=1,run_id='newdriver',accepted=True,status='done',success=True)
    (root/'client/journal.jsonl').write_text(json.dumps(journal)+'\n')
    per=[dict(**journal,_kind='client_timing',**timing),dict(**journal,step_idx=0,winner_id='winner:0')]
    (root/'client/per_step.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in per))
    (root/'server_fixture').mkdir()
    server=[dict(ev='dec',uid=uid,attempt=1,step=0,tag='t',conn=2,winner='winner:0',vision=True),
            dict(ev='p3_decision',uid=uid,attempt=1,step=0,tag='t',conn=2,source='cache',parent_anchor=0,commit_controls=10)]
    (root/'server_fixture/decisions_fixture.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in server))
    before=inventory(root/'client_telemetry'/key)
    prepared=prepare(run,'test',out/'repair')
    assert inventory(root/'client_telemetry'/key)==before
    assert prepared['verified']['accepted_attempts']==1
    assert prepared['attempts'][0]['proof']['driver_run_id']=='newdriver'
    installed=install(out/'repair');assert installed['accepted_attempts']==1
    assert (root/'client_telemetry'/key/'controls.jsonl').read_bytes()==new.encode()
    assert replay_spills(run,'test',root/'client_spills')==1
    assert verify_arm(run,'test')['files']==2
    install(out/'repair')
    assert (out/'repair/originals'/key/'client_telemetry/controls.jsonl').read_bytes()==old
    # Missing prefix and a wrong accepted incarnation fail before installation.
    trace=root/'client_telemetry'/key/'controls.jsonl';trace.write_bytes(b'X'+new.encode()[1:])
    try:prepare(run,'test',out/'bad_prefix')
    except ValueError as e:assert 'SHA mismatch' in str(e)
    else:raise AssertionError('guessed a missing prefix')
    trace.write_bytes(new.encode())
    per[0]['env_ms']=99
    (root/'client/per_step.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in per))
    try:prepare(run,'test',out/'bad_incarnation')
    except ValueError as e:assert 'different driver incarnation' in str(e)
    else:raise AssertionError('accepted outcome alone was treated as proof')
    # Real scheduler: first fresh dispatch unchanged; retried/resumed dispatch
    # advances the generation and still rejects a stale result.
    from openpi.conductor.scheduler import EpisodeScheduler
    from openpi.conductor.task import EpisodeTask,TaskGraph,Stage,ServerEndpoint
    ep=EpisodeTask(uid,'test','eval','fixture',4,1,1,'fixture',1,'test')
    endpoint=ServerEndpoint('fixture',1)
    def scheduler():
        graph=TaskGraph();graph.add_stage(Stage('s','test','eval',endpoint,[ep]))
        s=EpisodeScheduler(graph);s.pending_setups();s.mark_setup_running('s');s.mark_setup_done('s');return s
    original=EpisodeScheduler.next_task
    baseline=scheduler().next_task(endpoint.key)
    try:
        install_fence(out/'fence')
        first=scheduler();task1=first.next_task(endpoint.key)
        assert task1==baseline and task1.attempt==1
        assert first.mark_result(uid,success=False,retriable=True,attempt=1)
        task2=first.next_task(endpoint.key);assert task2.attempt==2
        resumed=scheduler();task3=resumed.next_task(endpoint.key);assert task3.attempt==3
        assert not resumed.mark_result(uid,success=True,retriable=False,attempt=1)
        assert resumed.mark_result(uid,success=True,retriable=False,attempt=3)
    finally:EpisodeScheduler.next_task=original
    fence=DispatchFence(out/'concurrent')
    with ThreadPoolExecutor(max_workers=4) as pool:
        generations=list(pool.map(lambda _:fence.reserve(uid,1),range(40)))
    assert sorted(generations)==list(range(1,41))
    assert DispatchFence(out/'concurrent').reserve(uid,1)==41
    legacy=out/'old_journal';legacy.touch()
    assert DispatchFence(out/'upgrade',legacy).reserve(uid,1)==100001
    # Storage failure cannot return/reuse a dispatch generation.
    with patch.object(DispatchFence,'_write',side_effect=OSError('disk failed')):
        try:fence.reserve(uid,1)
        except OSError:pass
        else:raise AssertionError('dispatch returned before durable reservation')
    from . import dispatch_fence, run_gtp_v2
    wiring=[]
    for enabled in (False,True):
        pkg=ModuleType('exp.gate_threshold_pareto');driver=ModuleType(pkg.__name__+'.run_gtp');pkg.run_gtp=driver
        calls=[];driver.WorkerSpec=lambda **kw:kw
        driver.ConductorDriver=lambda **kw:calls.append(('construct',kw))
        driver.main=lambda:driver.ConductorDriver(journal_path='journal-fixture')
        env=dict(P3_SNAPSHOT_DIR=str(out/'island/test/p3_telemetry'))
        if enabled:env['P3_STREAM']='127.0.0.1:1'
        with patch.dict(sys.modules,{pkg.__name__:pkg,driver.__name__:driver}),patch.dict(os.environ,env,clear=True),patch.object(sys,'argv',['test']),patch.object(dispatch_fence,'install',side_effect=lambda *args:calls.append(('fence',args))):
            run_gtp_v2.main()
        assert [x[0] for x in calls]==(['fence','construct'] if enabled else ['construct'])
        if enabled:assert calls[0][1]==(out/'island/test/.p3_dispatch','journal-fixture')
        wiring.append(enabled)
    return dict(PASS=True,sockets_opened=0,services_started=0,pilot_collision_reproduced=True,
        same_prefix_ack_then_conflict=True,exact_repair=True,source_unchanged_until_install=True,
        repeated_install_and_normal_replay=True,missing_prefix_rejected=True,wrong_driver_rejected=True,
        fresh_dispatch_byte_equal=True,generations=[1,2,3],stale_result_rejected=True,
        concurrent_reservations=40,process_reopen_next=41,legacy_upgrade_first=100001,disk_failure_closed=True,
        driver_wiring_stream_only=True,reservation_before_dispatch=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();result=run_tests(a.out)
    (a.out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
