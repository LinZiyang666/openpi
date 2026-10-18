"""CPU reproductions of the owner acceptance failures; no transports or binds."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from exp.offline_search.rounds.r17.autoscale_1017 import kernel_evidence as k, trial_runtime, trial_lane
from exp.offline_search.rounds.r17.autoscale_1017.reset_remote import archive


def snapshot(at, value, boot='boot', cg=0, inode=1):
    return dict(time=at,host='timan107',uid=1,boot_id=boot,boot_epoch=0,vmstat={'oom_kill':value},
                cgroups={'user.slice/memory.events':dict(device=1,inode=inode,oom=cg,oom_kill=cg,covers_self=True)})


@pytest.mark.parametrize('case,status', [('zero','PASS'),('kill','FAIL'),('cgroup','FAIL'),('reboot','UNKNOWN'),
    ('decrease','UNKNOWN'),('missing','UNKNOWN'),('lifetime_zero','PASS'),('late_start','UNKNOWN'),('short_end','UNKNOWN')])
def test_counter_window_without_kernel_permissions(case,status):
    first,last=snapshot(99,15),snapshot(201,15)
    if case=='kill':last['vmstat']['oom_kill']=16
    if case=='cgroup':last['cgroups']['user.slice/memory.events']['oom']=1
    if case=='reboot':last['boot_id']='different'
    if case=='decrease':last['vmstat']['oom_kill']=14;last['cgroups']={}
    if case=='missing':first=None
    if case=='lifetime_zero':first=None;last['vmstat']['oom_kill']=0
    if case=='late_start':first['time']=101
    if case=='short_end':last['time']=199
    result=k.counter_evidence(100,200,first,last)
    assert result['status']==status
    assert result['coverage']==(status!='UNKNOWN')


def test_recreated_cgroup_and_boot_after_lane_do_not_certify_zero():
    first,last=snapshot(99,15),snapshot(201,15,inode=2)
    first['vmstat']=last['vmstat']={}
    assert k.counter_evidence(100,200,first,last)['status']=='UNKNOWN'
    last['vmstat']={'oom_kill':0};last['boot_epoch']=150
    assert k.counter_evidence(100,200,None,last)['status']=='UNKNOWN'


def test_actual_collector_falls_back_and_preserves_source(tmp_path):
    (tmp_path/'AUTOSCALE_KERNEL_START.json').write_text(json.dumps(snapshot(99,15)))
    calls=[]
    def denied(argv,**kw):calls.append(argv);return SimpleNamespace(returncode=1,stderr='permission denied',stdout='')
    result=k.collect(tmp_path,100,execute=denied,now=lambda:200,snapshot=snapshot(201,15))
    assert result['status']=='PASS' and result['source']=='counters' and len(calls)==2
    assert result['counter_sources'][0]['delta']==0
    before=(tmp_path/'AUTOSCALE_KERNEL.json').read_bytes()
    result=k.collect(tmp_path,100,execute=denied,now=lambda:200,snapshot=snapshot(201,16),persist=False)
    assert result['status']=='FAIL' and (tmp_path/'AUTOSCALE_KERNEL.json').read_bytes()==before


@pytest.mark.parametrize('case,status', [('historical','PASS'),('new_log','FAIL'),
    ('counter_increase','FAIL'),('rotated','PASS'),('reboot','UNKNOWN')])
def test_start_cursor_and_vmstat_cover_lane_without_counting_historical_oom(tmp_path,monkeypatch,case,status):
    original=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda p,*a,**kw:'300 0' if str(p)=='/proc/uptime' else original(p,*a,**kw))
    anchor='[ 99.000000] acceptance start anchor\n'
    start='[ 90.000000] Out of memory: Killed process 123\n'+anchor
    calls=[]
    def execute(argv,**kw):
        calls.append(argv)
        if argv!=['dmesg']:return SimpleNamespace(returncode=1,stderr='permission denied',stdout='')
        text=start
        if len(calls)>1:
            text = anchor if case!='rotated' else '[ 150.000000] quiet after rotation\n'
            if case=='new_log':text+='[ 150.000000] Out of memory: Killed process 456\n'
            text+='[ 250.000000] Out of memory: Killed process 789\n'
        return SimpleNamespace(returncode=0,stdout=text,stderr='')
    baseline=k.begin(tmp_path,execute=execute,snapshot=snapshot(99,45))
    assert baseline['vmstat']['oom_kill']==45 and baseline['dmesg_cursor']['status']=='PASS'
    end=snapshot(301,46 if case=='counter_increase' else 45)
    if case=='reboot':end['boot_id']='new-boot';end['boot_epoch']=101
    result=k.collect(tmp_path,100,ended_at=200,now=lambda:300,execute=execute,snapshot=end)
    assert result['status']==status
    assert [r['time'] for r in result['oom_lines']]==([150.] if case=='new_log' else [])
    assert json.loads((tmp_path/'AUTOSCALE_KERNEL_START.json').read_text())==baseline
    assert sum(c==['dmesg'] for c in calls)==2


def test_normal_user_counter_reader(tmp_path):
    proc=tmp_path/'proc';cg=tmp_path/'cgroup'
    values={'vmstat':'oom_kill 15\n','uptime':'1000 20\n','sys/kernel/random/boot_id':'boot\n',
            'self/cgroup':f'0::/user.slice/user-{os.getuid()}.slice/session-1.scope\n'}
    for name,data in values.items():p=proc/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(data)
    events=cg/f'user.slice/user-{os.getuid()}.slice/memory.events';events.parent.mkdir(parents=True)
    events.write_text('oom 1\noom_kill 0\nhigh 40\n')
    result=k.counters(proc,cg)
    assert result['vmstat']['oom_kill']==15
    assert result['cgroups'][str(events)]['oom']==1


def frozen(root):
    root.mkdir()
    with tarfile.open(root/'runtime.tar','w') as out:
        data=b'def plan(c): return None\n';member=tarfile.TarInfo('exp/r17_integration/ops/formal_driver.py')
        member.size=len(data);out.addfile(member,io.BytesIO(data))
    archive_sha=hashlib.sha256((root/'runtime.tar').read_bytes()).hexdigest()
    return dict(run_name=root.name,server_node='h100',runtime='/data/mw_finetune/qualification/formal_runtime_'+archive_sha,
                autoscale_validation=True)


def test_old_runtime_missing_identity_is_promoted_from_exact_archive(tmp_path):
    root=tmp_path/'fa_asv_old';c=frozen(root);data=(root/'runtime.tar').read_bytes()
    info=trial_runtime.metadata(c,root)
    assert c['runtime_identity']==info['identity'] and c['runtime'].endswith(info['identity'])
    request=dict(info,root=str(tmp_path/'release'),archive=str(root/'runtime.tar'))
    assert not trial_runtime.install_checked(request)['reused']
    assert trial_runtime.install_checked(request)['reused']
    assert (root/'runtime.tar').read_bytes()==data
    c['runtime_identity']='wrong'
    with pytest.raises(ValueError,match='identity'):trial_runtime.metadata(c,root)


def test_existing_runtime_corruption_is_never_accepted_or_overwritten(tmp_path):
    root=tmp_path/'fa_asv_old';c=frozen(root);info=trial_runtime.metadata(c,root)
    target=tmp_path/'release';target.mkdir();(target/'RUNTIME_RELEASE.json').write_text('{}')
    with pytest.raises(ValueError,match='manifest mismatch'):
        trial_runtime.install_checked(dict(info,root=str(target),archive=str(root/'runtime.tar')))
    assert (target/'RUNTIME_RELEASE.json').read_text()=='{}'


def test_runtime_prepare_stages_same_archive_not_checkout(tmp_path,monkeypatch):
    root=tmp_path/'fa_asv_old';c=frozen(root);info=trial_runtime.metadata(c,root)
    c['runtime']=str(tmp_path/'local-release')
    calls=[]
    def remote(host,argv,**kw):
        calls.append(('remote',host,argv));return json.dumps(dict(reused=False))
    f=SimpleNamespace(remote=remote,transfer=lambda argv:calls.append(('transfer',argv)))
    result=trial_runtime.prepare(f,c,root)
    assert result['identity']==info['identity'] and c['coordinator_runtime']==c['runtime']
    assert calls[1][1][2]==str(root/'runtime.tar')
    assert json.loads((root/'RUNTIME_RELEASE.json').read_text())['archive_sha256']==info['archive_sha256']


def test_remote_reset_archives_all_leftovers_without_deleting_and_replays(tmp_path):
    root=tmp_path/'fa_test';root.mkdir();(root/'mps_control').mkdir();(root/'mps_control/control').write_text('old code')
    backup=tmp_path/'fa_test.failed';request=dict(root=str(root),backup=str(backup))
    result=archive(request,campaign_parent=tmp_path,proc_root=tmp_path/'proc')
    assert result['status']=='ARCHIVED' and not root.exists()
    assert (backup/'mps_control/control').read_text()=='old code'
    assert archive(request,campaign_parent=tmp_path,proc_root=tmp_path/'proc')==result


@pytest.mark.parametrize('filename',['receipt.json','RECEIPT.json'])
def test_remote_reset_refuses_live_receipt(tmp_path,filename):
    root=tmp_path/'fa_test';root.mkdir();backup=tmp_path/'fa_test.failed'
    (root/filename).write_text(json.dumps(dict(child=dict(pid=123,start_ticks='99'))))
    proc=tmp_path/'proc';p=proc/'123';p.mkdir(parents=True)
    (p/'stat').write_text('123 (child) S '+'0 '*18+'99 0\n');(p/'cmdline').write_bytes(b'python\0')
    with pytest.raises(ValueError,match='live remote receipt'):
        archive(dict(root=str(root),backup=str(backup)),campaign_parent=tmp_path,proc_root=proc)
    assert root.exists() and not backup.exists()


def test_pre_driver_reset_requires_closed_no_owned_processes_and_remote_backup(tmp_path,monkeypatch):
    from exp.offline_search.rounds.r17.autoscale_1017.tests.test_reset_zero import fixture
    from exp.offline_search.rounds.r17.autoscale_1017 import reset_zero,reset_remote
    root=tmp_path/'fa_test';fixture(root)
    # Preserve the synthetic empty journal as evidence rather than deleting it.
    (root/'driver/journal.jsonl').rename(root/'driver/old.empty')
    trial_lane.write(root/'RECEIPT.json',dict(status='FAIL',all_owned_closed=True,owned_local=[],owned_remote=[],backend_evidence={}))
    trial_lane.write(root/'campaign.json',dict(server_node='h100',server_run=str(root/'servers')))
    with pytest.raises(ValueError,match='remote-backup'):reset_zero.reset(root,tmp_path/'backup')
    calls=[];monkeypatch.setattr(reset_remote,'run',lambda *a,**kw:calls.append(a) or dict(status='ARCHIVED'))
    result=reset_zero.reset(root,tmp_path/'backup',remote_backup=tmp_path/'fa_test.failed')
    assert result['pre_driver_failure'] and calls[0][0]=='h100'


def test_real_old_fw_source_prepare_keeps_all_256_task_fields(tmp_path,monkeypatch):
    from exp.r17_integration.ops import formal_eval as f, driver_release, immutable_release, formal_prestage, formal_ports
    from exp.offline_search.rounds.r17.filter_failed import stage
    from exp.offline_search.rounds.r17.autoscale_1017 import formal_trial, release
    from contextlib import nullcontext
    source=Path('/data/mw_finetune/eval/formal_fastwam_mw_mt50_500_20261009_223917_0b845f00')
    original=(source/'campaign.json').read_bytes()
    root=tmp_path/'fa_asv_cpu_old'
    trial_lane.build(source,root,'r17')
    expected=trial_lane.read(root/'EPISODE_PLAN.json')['tasks']
    # Reproduce the already-built partial lane: metadata still has the old
    # archive-SHA address and lacks runtime_identity when prepare starts.
    c=trial_lane.read(root/'campaign.json');c.pop('runtime_identity')
    c['runtime']=json.loads(original)['runtime'];trial_lane.write(root/'campaign.json',c)
    pinned=(root/'runtime.tar').read_bytes()
    installs={}
    def install(archive,path,archive_sha,pin):
        target=tmp_path/'installed'/Path(path).name
        doc=immutable_release.archive_manifest(archive)
        proof=trial_runtime.install_checked(dict(root=str(target),archive=str(archive),document=doc,
            archive_sha256=archive_sha,identity=pin))
        installs[path]=str(target)
        return proof
    native_prepare=driver_release.prepare
    def prepare_driver(c,r,checkout):
        native_prepare(c,r,checkout)
        c['driver_runtime']=installs[c['driver_runtime']]
        c['orchestration']['driver_runtime']=c['driver_runtime']
        trial_lane.write(r/'ORCHESTRATION.json',c['orchestration'])
    def runtime_prepare(launch,c,r):
        info=trial_runtime.metadata(c,r)
        install(r/'runtime.tar',info['root'],info['archive_sha256'],info['identity'])
        trial_lane.write(r/'RUNTIME_RELEASE.json',{k:v for k,v in info.items() if k!='document'})
    monkeypatch.setattr(trial_runtime,'prepare',runtime_prepare)
    monkeypatch.setattr(immutable_release,'install',install)
    monkeypatch.setattr(driver_release,'install',install)
    monkeypatch.setattr(driver_release,'prepare',prepare_driver)
    monkeypatch.setattr(formal_ports,'prepare_admission',lambda c:None)
    monkeypatch.setattr(f,'remote',lambda *a,**kw:pytest.fail('unmocked remote'))
    monkeypatch.setattr(f,'transfer',lambda *a,**kw:pytest.fail('unmocked transfer'))
    bootstrap=[]
    def prepare_workers(c,r,*,payload_override):
        # Real immutable bootstrap on the old 10-09 source, not a getmember mock.
        import subprocess,sys
        destination=tmp_path/'worker-installed'
        command=[sys.executable,str(trial_lane.REPO/'exp/r17_integration/qualification/bootstrap_install.py'),
            str(payload_override),str(destination),immutable_release.sha(payload_override),'--immutable']
        result=subprocess.run(command,capture_output=True,text=True,check=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1'))
        manifest=json.loads(result.stdout)
        assert 'exp/r17_integration/ops/worker_installs.py' in manifest['files']
        assert (destination/'exp/r17_integration/ops/worker_installs.py').is_file()
        assert (destination/'exp/r17_integration/ops/worker_disk.py').is_file()
        # The install is immutable and reusable, including unchanged science files.
        subprocess.run(command,capture_output=True,text=True,check=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1',
                 PYTHONPATH=f.pythonpath(str(destination)))
        code=('import json;from exp.r17_integration.ops import formal_workers;'
              'from exp.r17_integration.conductor.worker_entry import import_probe;'
              'print(json.dumps(import_probe()))')
        probe=subprocess.run([sys.executable,'-c',code],env=env,cwd=destination,
                             capture_output=True,text=True,check=True)
        assert json.loads(probe.stdout)['client_only'] is True
        bootstrap.append(manifest)
        return {'timan1':{}}
    monkeypatch.setattr(f,'prepare_workers',prepare_workers)
    monkeypatch.setattr(f,'retry_scope',lambda *a:nullcontext())
    monkeypatch.setattr(stage,'stage_server_inputs',lambda *a:None)
    monkeypatch.setattr(stage,'stage_worker_inputs',lambda *a:None)
    monkeypatch.setattr(formal_prestage,'seal_prepared',lambda c,r,d:trial_lane.write(r/'PREPARED.json',dict(cpu_dry_run=True)))
    # Explicit test fixtures normally select candidate ON; this call tests the
    # real isolated OFF->private-ON entry and never changes the physical file.
    trial_lane.run('prepare',root)
    updated=trial_lane.read(root/'campaign.json')
    assert updated['runtime_identity']==immutable_release.identity(immutable_release.archive_manifest(root/'runtime.tar'))
    assert driver_release.planned_tasks(updated,root)==expected
    assert (root/'runtime.tar').read_bytes()==pinned
    assert (source/'campaign.json').read_bytes()==original
    assert (trial_lane.BASE/'ENABLED').read_text().strip()=='off'
    assert len(bootstrap)==1
