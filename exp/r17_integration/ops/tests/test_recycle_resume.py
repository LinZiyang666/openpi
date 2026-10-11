"""Reserved N4 slot transfer and accepted-cell retry, with CPU-only children."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest

from exp.r17_integration.ops import formal_workers as fleet, worker_gpu as gpu
from exp.r17_integration.ops import queue_retry, cache_control
from exp.r17_integration.ops.formal_process import ticks, write_json


def test_real_structured_terminal_ack_and_tampering():
    from exp.r17_integration.ops.cache_resume import terminal_ack_matches
    decision=dict(attempt=2,decision_seq=14,task_id=0,task_name='sp/task',bundle_identity='bundle',
        bundle_generation='gen',arm='c',benchmark_protocol='A',original_init_idx=0,campaign='segment')
    server=dict(terminal=True,decision_seq=14,executed_count=4,cursor_end=4,bundle_identity='bundle',
                request={k:v for k,v in decision.items() if k!='bundle_identity'})
    ack=dict(server_terminal_ack=server,decision_seq=14,executed_count=4,cursor_end=4)
    assert terminal_ack_matches(ack,decision)
    for key in ('attempt','decision_seq','task_id','task_name','bundle_generation','arm','benchmark_protocol','original_init_idx','campaign'):
        assert not terminal_ack_matches(dict(ack,server_terminal_ack=dict(server,request=dict(server['request'],**{key:'wrong'}))),decision)
    assert not terminal_ack_matches(dict(ack,server_terminal_ack=True),decision)
    assert not terminal_ack_matches(dict(ack,server_terminal_ack=dict(server,bundle_identity='wrong')),decision)


def test_recycle_reuses_held_gpu_slot_without_nvidia_or_fresh_headroom(tmp_path,monkeypatch):
    result=dict(assignments=[dict(index=0,device=3,uuid='GPU-3',worker_vram_mib=576)],worker_vram_mib=576)
    lease=gpu.Reservation({},tmp_path,result)
    child=subprocess.Popen([sys.executable,'-c','raise SystemExit(75)'])
    entry=dict(pid=child.pid,start_ticks=ticks(child.pid),index=0,egl_device=3,
               gpu_uuid='GPU-3',worker_vram_mib=576,exit_code=None)
    child.wait(timeout=10)
    monkeypatch.setattr(gpu,'inventory',lambda *a:pytest.fail('reserved recycle must not re-admit GPU'))
    with mock.patch('exp.r17_integration.ops.worker_memory.admission',side_effect=AssertionError('fresh RAM headroom')):
        proof=fleet.recycle_slot(dict(memory_admission=dict(admitted=True)),child,entry,lease)
    assert proof['admitted'] and proof['ram_slot_reused']
    assert proof['requested_extra_ram_bytes']==proof['requested_extra_gpu_mib']==0
    assert proof['gpu']['device']==3 and proof['gpu']['old_pid']==child.pid
    assert entry['exit_code']==75
    lease.close()
    with pytest.raises(ValueError,match='reservation lost'):
        lease.reuse(0,entry)


def test_slot_transfer_refuses_live_old_pid_or_changed_device(tmp_path):
    result=dict(assignments=[dict(index=0,device=3,uuid='GPU-3',worker_vram_mib=576)],worker_vram_mib=576)
    lease=gpu.Reservation({},tmp_path,result)
    live=dict(pid=os.getpid(),start_ticks=ticks(os.getpid()),index=0,exit_code=75)
    assert not lease.reuse(0,live)['admitted']
    dead=dict(live,start_ticks='different',egl_device=4,gpu_uuid='GPU-3',worker_vram_mib=576)
    with pytest.raises(ValueError,match='changed GPU slot'):lease.reuse(0,dead)
    lease.close()


def test_fleet_planned_recycle_never_calls_fresh_ram_or_gpu_admission(tmp_path,monkeypatch):
    config=dict(run_name='cpu',worker_node='cpu',worker_run=str(tmp_path/'run'),domain='libero',
        node_workers=1,seconds=30,endpoint='host:1',driver_host='host',driver_port=2,
        worker_benchmark='/sealed',worker_egl_device=0,worker_max_episodes=4,cache_study='STUDY')
    write_json(tmp_path/'campaign.json',config)
    monkeypatch.setattr(sys,'argv',['fleet','--campaign',str(tmp_path/'campaign.json')])
    old=mock.Mock(pid=987654);old.poll.return_value=75;old.wait.return_value=75
    new=mock.Mock(pid=987653);new.poll.return_value=0;new.wait.return_value=0
    lease=mock.Mock();lease.result=dict(assignments=[dict(index=0,device=0,uuid='GPU-0',worker_vram_mib=576)])
    lease.reuse.return_value=dict(admitted=True,reused=True)
    lease.check.side_effect=AssertionError('fresh GPU admission')
    with mock.patch.object(gpu,'acquire',return_value=lease), \
         mock.patch.object(fleet.subprocess,'Popen',side_effect=[old,new]), \
         mock.patch.object(fleet.time,'sleep'), \
         mock.patch('exp.r17_integration.ops.worker_lifetime.planned_recycle',return_value=True), \
         mock.patch('exp.r17_integration.ops.worker_memory.admission',side_effect=[dict(admitted=True),AssertionError('fresh RAM admission')]) as admission, \
         mock.patch('exp.r17_integration.ops.sync_back.snapshot_run',return_value=dict(archive_sha256='cpu')) as snapshot, \
         mock.patch('exp.r17_integration.qualification.archive_stage.stage_archive',return_value={}):
        with pytest.raises(SystemExit) as done:fleet.main()
    assert done.value.code==0
    assert admission.call_count==1 and lease.reuse.call_count==1 and lease.check.call_count==0
    assert snapshot.call_args.kwargs['compressed'] is True
    receipt=json.loads((tmp_path/'run/receipt.json').read_text())
    assert receipt['status']=='PASS' and not receipt['worker_failures']
    assert len(receipt['worker_recycles'])==1
    assert receipt['last_replacement']['admission']['requested_extra_ram_bytes']==0
    assert fleet.RECYCLE_TIMEOUT_SECONDS>=600


def test_accepted_job_uses_explicit_resume_and_new_log_name(tmp_path,monkeypatch):
    old=tmp_path/'old';old.mkdir();(old/'driver').mkdir()
    write_json(old/'campaign.json',dict(cache_study='sealed'))
    (old/'driver/journal.jsonl').write_text(json.dumps(dict(accepted=True))+'\n')
    fresh=tmp_path/'dpc_resume';fresh.mkdir()
    calls=[];events=[]
    class Runner:
        stopped=False
        def invoke(self,job,phase,argv):
            calls.append(dict(job))
            if len(calls)==1:
                write_json(old/'RECEIPT.json',dict(status='FAIL',all_owned_closed=True))
                raise RuntimeError('planned recycle transient failure')
            write_json(fresh/'RECEIPT.json',dict(status='PASS',all_owned_closed=True))
        def event(self,**kw):events.append(kw)
    monkeypatch.setattr(queue_retry,'clone',lambda *a,**k:pytest.fail('accepted outcomes cannot startup-retry'))
    resumed=[]
    result=queue_retry.run_job(dict(name='old',run=str(old),model='dp_libero'),Runner(),delay=0,
        resume_handler=lambda job:(resumed.append(job['run']) or fresh))
    assert resumed==[str(old)] and result['run']==str(fresh)
    assert calls[1]['name']==fresh.name and calls[1]['model']=='dp_libero'
    assert events[0]['event']=='cache_resume'


def test_refresh_changes_only_two_ops_files_and_keeps_all_native_bytes(tmp_path,monkeypatch):
    from exp.r17_integration.ops.build_client_bundle import build_client_bundle,verified_members
    source=tmp_path/'source';source.mkdir();new=tmp_path/'new';new.mkdir();root=tmp_path/'root';root.mkdir()
    names=['exp/r17_integration/ops/formal_workers.py','exp/r17_integration/ops/worker_gpu.py',
           'packages/openpi-client/src/openpi_client/base_policy.py']
    for name in names:
        p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# frozen\n')
        q=root/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_text('# refreshed\n')
    build_client_bundle(source,names,source/'workers.tar')
    c=dict(cache_worker_refresh=dict(archive=str(source/'workers.tar'),archive_sha256=cache_control.sha(source/'workers.tar'),
        files={n:cache_control.sha(root/n) for n in names[:2]}))
    monkeypatch.setattr(cache_control,'check',lambda *a:None)
    cache_control.worker_payload(c,new,root,names)
    old,_=verified_members(source/'workers.tar','PAYLOAD.json');fresh,_=verified_members(new/'workers.tar','PAYLOAD.json')
    assert old['files'][names[2]]==fresh['files'][names[2]]
    assert all(old['files'][n]!=fresh['files'][n] for n in names[:2])
    c['cache_worker_refresh']['files'][names[2]]=cache_control.sha(root/names[2])
    with pytest.raises(ValueError,match='orchestration boundary'):cache_control.worker_payload(c,new,root,names)


def test_prestaged_cold_cache_workers_refresh_before_claim_preserving_native(tmp_path,monkeypatch):
    from exp.r17_integration.ops import formal_eval as launch,formal_prestage as stage
    from exp.r17_integration.ops.build_client_bundle import build_client_bundle
    root=tmp_path/'root';root.mkdir();p=tmp_path/'cold';p.mkdir();(p/'assets').mkdir()
    names=['exp/r17_integration/ops/formal_workers.py','exp/r17_integration/ops/worker_gpu.py','exp/r17_integration/ops/worker_disk.py',
           'exp/r17_integration/ops/sync_back.py',
           'packages/openpi-client/src/openpi_client/base_policy.py']
    for n in names:
        f=root/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('# original')
    # Existing frozen controls predate the disk helper; refresh must add it
    # without altering any native simulator/client file.
    build_client_bundle(root,[n for n in names if not n.endswith('/worker_disk.py')],p/'workers.tar')
    for n in names[:2]:(root/n).write_text('# recycled')
    c=dict(cache_study='seal',cache_control='control',runtime='native',runtime_identity='pin')
    for n in ('campaign.json','WORKER_DEPLOYMENT.json','PREFLIGHT.json','EPISODE_PLAN.json','ORCHESTRATION.json'):
        write_json(p/n,c if n=='campaign.json' else {})
    (p/'runtime.tar').write_bytes(b'native bytes')
    (p/'assets/model_spec.json').write_text('native seal')
    stage.seal_prepared(c,p,{})
    before={n:(p/n).read_bytes() for n in ('runtime.tar','EPISODE_PLAN.json','ORCHESTRATION.json','assets/model_spec.json')}
    monkeypatch.setattr(launch,'ROOT',root);monkeypatch.setattr(cache_control,'check',lambda *a:None)
    def prepare(c,d):
        cache_control.worker_payload(c,d,root,names);write_json(d/'WORKER_DEPLOYMENT.json',{});return {}
    monkeypatch.setattr(launch,'prepare_workers',prepare)
    updated,deployment=cache_control.refresh_prepared_workers(c,p,{})
    assert all((p/n).read_bytes()==b for n,b in before.items())
    assert cache_control.read(p/'CACHE_RECYCLE_REFRESH.json')['cpu_only']
    assert cache_control.refresh_prepared_workers(updated,p,deployment)==(updated,deployment)
    (p/'LAUNCH.json').write_text('{}')
    with pytest.raises(ValueError,match='dispatched'):cache_control.refresh_prepared_workers(updated,p,deployment)
    assert cache_control.refresh_prepared_workers({},p,{})==({},{})


def test_lane_resumes_accepted_failure_then_continues_next_model(tmp_path,monkeypatch):
    from exp.r17_integration.ops import cache_lane as lane,formal_queue,formal_prestage,formal_finalize
    directory=tmp_path/'lane';directory.mkdir();manifest=tmp_path/'study.json';manifest.write_text('{}')
    fw=tmp_path/'fastwam';fw.mkdir();(fw/'driver').mkdir()
    fresh=tmp_path/'fw_continuation';fresh.mkdir();cosmos=tmp_path/'cosmos';cosmos.mkdir()
    write_json(directory/'LANE_PLAN.json',dict(lane='h100',long_dp=10,long_fast=10,
        study=str(manifest),study_sha256=cache_control.sha(manifest),jobs=[
            dict(model='fastwam_libero',name=fw.name,run=str(fw),argv=['--seconds','86400']),
            dict(model='cosmos_libero',name=cosmos.name,run=str(cosmos),argv=['--seconds','86400'])]))
    launched=[];events=[]
    class Runner:
        stopped=False
        def __init__(self,d):pass
        def invoke(self,job,phase,argv):
            launched.append((job['name'],job['model']))
            p=Path(job['run'])
            if p==fw:
                write_json(p/'campaign.json',dict(cache_study='sealed'))
                write_json(p/'RECEIPT.json',dict(status='FAIL',all_owned_closed=True))
                (p/'driver/journal.jsonl').write_text(json.dumps(dict(accepted=True))+'\n')
                raise RuntimeError('planned recycle timed out')
            write_json(p/'RECEIPT.json',dict(status='PASS',all_owned_closed=True))
        def event(self,**kw):events.append(kw)
        def close(self):return True
    monkeypatch.setattr(formal_queue,'Runner',Runner)
    monkeypatch.setattr(formal_prestage,'validate_prepared',lambda p:({},{}))
    monkeypatch.setattr(formal_finalize,'wait_seal',lambda p:None)
    monkeypatch.setattr(lane,'resume_job',lambda job,*args:fresh)
    assert lane.main(['--lane','h100','--directory',str(directory),'--run'])==0
    assert launched==[(fw.name,'fastwam_libero'),(fresh.name,'fastwam_libero'),(cosmos.name,'cosmos_libero')]
    assert events[0]['event']=='cache_resume'
    state=cache_control.read(directory/'LANE_STATE.json')
    assert state['status']=='PASS' and state['jobs']['fastwam_libero']['run']==str(fresh)
    assert state['jobs']['cosmos_libero']['status']=='PASS'


def test_many_real_cpu_recycles_fit_low_fd_limit_and_keep_every_log(tmp_path):
    """Old logs.append leaked one FD per generation and failed under RLIMIT."""
    config=dict(run_name='cpu',worker_node='cpu',worker_run=str(tmp_path/'run'),domain='libero',
        node_workers=1,seconds=25,endpoint='host:1',driver_host='host',driver_port=2,cache_study='study')
    write_json(tmp_path/'campaign.json',config)
    code='''
import json,resource,sys
from pathlib import Path
from unittest import mock
from exp.r17_integration.ops import formal_workers as f,worker_gpu as g
resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
class Lease:
    def close(self):pass
child_code="import sys; from pathlib import Path; n=int(Path.cwd().name.split('_')[-1]); print(n,flush=True); sys.exit(75 if n<80 else 0)"
f.worker_command=lambda c,i:('cpu',[sys.executable,'-c',child_code])
with mock.patch.object(g,'acquire',return_value=Lease()), \\
     mock.patch('exp.r17_integration.ops.worker_memory.admission',return_value={'admitted':True}), \\
     mock.patch('exp.r17_integration.ops.worker_lifetime.planned_recycle',side_effect=lambda e,c:c==75), \\
     mock.patch.object(f.time,'sleep'), \\
     mock.patch('exp.r17_integration.ops.sync_back.snapshot_run',return_value={'archive_sha256':'cpu'}), \\
     mock.patch('exp.r17_integration.qualification.archive_stage.stage_archive',return_value={}):
    f.main()
'''
    result=subprocess.run([sys.executable,'-c',code,'--campaign',str(tmp_path/'campaign.json')],
        text=True,capture_output=True,timeout=30,cwd=Path(__file__).resolve().parents[4])
    assert result.returncode==0,result.stderr[-3000:]+result.stdout[-2000:]
    receipt=json.loads((tmp_path/'run/receipt.json').read_text())
    assert receipt['status']=='PASS' and len(receipt['worker_recycles'])==80
    assert len(receipt['pids'])==81
    for n in range(81):assert (tmp_path/'run'/f'worker_0_generation_{n}.log').read_text().strip()==str(n)
