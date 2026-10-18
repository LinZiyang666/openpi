"""CPU reproduction of 12:05 stale-stage failure and safe closure/reset."""
import json
from pathlib import Path

import pytest

from exp.offline_search.rounds.r17.autoscale_1017 import prestart_closure as closure
from exp.offline_search.rounds.r17.autoscale_1017 import reset_remote, reset_zero
from exp.offline_search.rounds.r17.autoscale_1017.node import atomic
from exp.r17_integration.ops import formal_process


@pytest.fixture(autouse=True)
def synthetic_copy_only(monkeypatch):
    # Synthetic byte-sized fixtures must not contend with a live prepare's
    # production bulk-reader lock. Production code retains the bounded copier.
    import shutil
    monkeypatch.setattr('exp.offline_search.rounds.r17.longseg_1017.io_budget.copy',shutil.copyfile)


def process(proc,pid=123,*,pgid=123,ticks='99'):
    d=proc/str(pid);d.mkdir(parents=True)
    fields=['S','0',str(pgid)]+['0']*16+[ticks,'0']
    (d/'stat').write_text(str(pid)+' (cpu child) '+' '.join(fields)+'\n')
    (d/'cmdline').write_bytes(b'python\0unrelated\0')
    return d


def test_real_spawn_collision_occurs_before_popen_or_lease(tmp_path,monkeypatch):
    path=tmp_path/'server/mps_guard.process.json';atomic(path,dict(status='CLOSED'))
    monkeypatch.setattr(formal_process.subprocess,'Popen',lambda *a,**kw:pytest.fail('must not spawn'))
    with pytest.raises(FileExistsError,match='mps_guard.process.json'):
        formal_process.spawn(dict(receipt=str(path),lease=str(tmp_path/'new.lease')))
    assert not (tmp_path/'new.lease').exists()


def test_stage_archive_preserves_stale_receipt_and_real_spawn_gets_past_collision(tmp_path,monkeypatch):
    root=tmp_path/'fa_cpu';path=root/'server/mps_guard.process.json'
    atomic(path,dict(status='CLOSED',child=dict(pid=999999999,start_ticks='1')))
    old=path.read_bytes();backup=tmp_path/'fa_cpu.failed'
    result=reset_remote.archive(dict(root=str(root),backup=str(backup)),campaign_parent=tmp_path,proc_root=tmp_path/'proc')
    assert result['status']=='ARCHIVED' and (backup/'server/mps_guard.process.json').read_bytes()==old
    path.parent.mkdir(parents=True)
    class AtSpawn(Exception):pass
    def popen(*a,**kw):raise AtSpawn('CPU proof: next native operation is spawn')
    monkeypatch.setattr(formal_process.subprocess,'Popen',popen)
    with pytest.raises(AtSpawn):
        formal_process.spawn(dict(receipt=str(path),lease=str(path.with_suffix('.lease')),
            log=str(path.with_suffix('.log')),env={},cwd=str(root)))


@pytest.mark.parametrize('filename',['server/mps_guard.process.json','MPS_READY.json'])
def test_archive_refuses_live_staging_identity(tmp_path,filename):
    root=tmp_path/'fa_cpu';proc=tmp_path/'proc';process(proc)
    atomic(root/filename,dict(child=dict(pid=123,start_ticks='99')))
    with pytest.raises(ValueError,match='live remote receipt'):
        reset_remote.archive(dict(root=str(root),backup=str(tmp_path/'fa_cpu.failed')),
                             campaign_parent=tmp_path,proc_root=proc)
    assert root.exists()


def test_dead_guard_with_live_group_member_cannot_archive_or_close(tmp_path):
    root=tmp_path/'fa_cpu';proc=tmp_path/'proc';process(proc,pid=124,pgid=123)
    atomic(root/'server/mps_guard.process.json',dict(child=dict(pid=123,start_ticks='1',pgid=123)))
    assert closure.probe(dict(roots=[str(root)]),proc_root=proc)['status']=='FAIL'
    with pytest.raises(ValueError,match='process group'):
        reset_remote.archive(dict(root=str(root),backup=str(tmp_path/'fa_cpu.failed')),
                             campaign_parent=tmp_path,proc_root=proc)


def test_preflight_is_read_only_and_absent_stage_is_allowed(tmp_path):
    root=tmp_path/'fa_cpu';root.mkdir();(root/'evidence').write_bytes(b'keep')
    request=dict(root=str(root),backup=str(tmp_path/'fa_cpu.failed'))
    result=reset_remote.archive(request,campaign_parent=tmp_path,proc_root=tmp_path/'proc',check_only=True)
    assert result['status']=='READY' and sorted(p.name for p in root.iterdir())==['evidence']
    missing=dict(root=str(tmp_path/'fa_missing'),backup=str(tmp_path/'fa_missing.failed'))
    assert reset_remote.archive(missing,campaign_parent=tmp_path,check_only=True)['status']=='ABSENT'


def fixture(root):
    atomic(root/'campaign.json',dict(run_name=root.name,server_node='h100',mps={'enabled':True},server_run=str(root/'servers')))
    atomic(root/'EPISODE_PLAN.json',dict(tasks=[dict(task_uid='same',seed=7,init=1)]))
    atomic(root/'LAUNCH.json',dict(pid=999999999))
    atomic(root/'LAUNCH_WORKER_DEPLOYMENT.json',{})
    receipt=dict(status='PARTIAL',all_owned_closed=False,owned_local=[],owned_remote=[],backend_evidence={},
        cleanup=[dict(label='mps_daemon',closure_confirmed=False,error='missing MPS_READY.json')],
        error='FileExistsError: /tmp/r17_formal/'+root.name+'/server/mps_guard.process.json')
    atomic(root/'RECEIPT.json',receipt)
    return (root/'RECEIPT.json').read_bytes()


def finalize(root,tmp_path,monkeypatch,*,live=False):
    def probe(request,**kwargs):return dict(status='FAIL' if live else 'PASS',request=request,live=[123] if live else [])
    monkeypatch.setattr(closure,'probe',probe)
    def remote(node,argv,**kwargs):
        assert node=='h100' and len(argv[-1])<32768
        return json.dumps(dict(status='FAIL' if live else 'PASS'))
    return closure.finalize(root,moved_stage='/tmp/r17_formal/'+root.name+'.moved',
                            evidence_dir=tmp_path/'audit',remote=remote)


def test_partial_prestart_closure_preserves_original_and_error(tmp_path,monkeypatch):
    root=tmp_path/'fa_cpu';original=fixture(root)
    proof=finalize(root,tmp_path,monkeypatch)
    now=json.loads((root/'RECEIPT.json').read_text())
    assert now['status']=='FAIL' and now['all_owned_closed'] and proof['accepted']==0
    assert (root/now['closure_finalization']['original_receipt']).read_bytes()==original
    assert now['error']==json.loads(original)['error']
    assert now['cleanup'][0]['original_closure_confirmed'] is False


@pytest.mark.parametrize('unsafe',['live','journal','owned','different_error'])
def test_closure_refuses_unsafe_state_without_changing_receipt(tmp_path,monkeypatch,unsafe):
    root=tmp_path/'fa_cpu';fixture(root)
    r=json.loads((root/'RECEIPT.json').read_text())
    if unsafe=='journal':atomic(root/'driver/journal.jsonl',{'accepted':True})
    if unsafe=='owned':r['owned_local']=[dict(pid=123)]
    if unsafe=='different_error':r['error']='something else failed'
    atomic(root/'RECEIPT.json',r);before=(root/'RECEIPT.json').read_bytes()
    with pytest.raises(ValueError):finalize(root,tmp_path,monkeypatch,live=unsafe=='live')
    assert (root/'RECEIPT.json').read_bytes()==before


def test_reset_preflights_then_archives_server_stage_and_worker_state(tmp_path,monkeypatch):
    root=tmp_path/'fa_cpu';fixture(root)
    r=json.loads((root/'RECEIPT.json').read_text());r.update(status='FAIL',all_owned_closed=True);atomic(root/'RECEIPT.json',r)
    atomic(root/'LAUNCH_WORKER_DEPLOYMENT.json',{'timan1':{
        'stage':'/scratch/zixuans8/mw_r17/formal/stage/fa_cpu/timan1',
        'worker_run':'/scratch/zixuans8/mw_r17/formal/runs/fa_cpu_timan1'}})
    calls=[]
    def run(node,path,backup,**kw):
        calls.append((node,str(path),str(backup),kw.get('check_only',False)))
        return dict(status='READY' if kw.get('check_only') else 'ARCHIVED',request={'root':str(path)})
    monkeypatch.setattr(reset_remote,'run',run)
    out=reset_zero.reset(root,tmp_path/'fa_cpu.failed',remote_backup=tmp_path/'fa_cpu.failed')
    assert len(calls)==8 and all(c[3] for c in calls[:4]) and not any(c[3] for c in calls[4:])
    assert calls[1][1:] == ('/tmp/r17_formal/fa_cpu','/tmp/r17_formal/fa_cpu.failed',True)
    assert len(out['node_backups'])==4
    assert json.loads((root/'EPISODE_PLAN.json').read_text())['tasks']==[dict(task_uid='same',seed=7,init=1)]


def test_failed_node_preflight_leaves_all_directories_untouched(tmp_path,monkeypatch):
    root=tmp_path/'fa_cpu';fixture(root)
    r=json.loads((root/'RECEIPT.json').read_text());r.update(status='FAIL',all_owned_closed=True);atomic(root/'RECEIPT.json',r)
    calls=[]
    def run(*args,**kwargs):
        assert kwargs['check_only'];calls.append(args)
        if len(calls)==2:raise ValueError('live remote receipt')
        return dict(status='READY')
    monkeypatch.setattr(reset_remote,'run',run)
    with pytest.raises(ValueError,match='live'):
        reset_zero.reset(root,tmp_path/'fa_cpu.failed',remote_backup=tmp_path/'fa_cpu.failed')
    assert (root/'LAUNCH.json').exists() and not (tmp_path/'fa_cpu.failed').exists()


def test_archive_dispatch_refuses_unapproved_node_and_shared_payload():
    for node,root in [('timan108','/tmp/r17_formal/fa_cpu'),('h100','/data/mw_finetune/qualification/fa_cpu')]:
        with pytest.raises(ValueError,match='unsupported'):
            reset_remote.run(node,root,root+'.backup')
