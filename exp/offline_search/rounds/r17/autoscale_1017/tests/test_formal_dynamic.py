"""Large real-formal orchestration: CPU only, no transport or original-run writes."""
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from exp.offline_search.rounds.r17.autoscale_1017 import formal_dynamic as f, node as n
from exp.offline_search.rounds.r17.autoscale_1017.policy import GIB
from exp.offline_search.rounds.r17.autoscale_1017.tests.test_runtime import Clock, node


def test_private_switch_never_changes_global_file():
    path=f.HERE/'ENABLED';before=path.read_bytes()
    with f.private_on():
        assert path.read_text().strip()=='on'
        with pytest.raises(PermissionError):path.write_text('on\n')
    assert path.read_bytes()==before==b'off\n'


def test_formal_maxima_are_physical_lower_bounds_and_science_equal():
    c=dict(domain='metaworld',run_name='formal',execution_worker_nodes={'timan1':60,'timan107':60},
           worker_nodes={'timan1':40,'timan107':40,'timan108':40},workers=120,seed=7,
           checkpoint='unchanged',runtime='fixed',cache_study='fixed',replicas=3,miss_steps=10)
    d=f.dynamic_campaign(c)
    assert d['execution_worker_nodes']=={'timan1':332,'timan107':440}
    assert d['autoscale_maximum']==772 and 'worker_hard_limits' not in d
    assert {k:d[k] for k in ('seed','checkpoint','runtime','cache_study','replicas','miss_steps')}=={
        k:c[k] for k in ('seed','checkpoint','runtime','cache_study','replicas','miss_steps')}
    assert d['autoscale_node_config']['timan107']['floor']==96*GIB


def test_dispatch_targets_fresh_formal_prepare_without_rewriting_definition(tmp_path,monkeypatch):
    r=tmp_path/'formal';r.mkdir();c=dict(domain='metaworld',worker_nodes={'timan1':60,'timan107':60},workers=120)
    n.atomic(r/'campaign.json',c);data=(r/'campaign.json').read_bytes();calls=[]
    monkeypatch.setattr(f,'jobs',lambda name:[dict(run=str(r),pipeline='R17')])
    monkeypatch.setattr(f.formal_trial,'prepare',lambda *a,**kw:calls.append((a,kw)))
    monkeypatch.setattr(f,'record_r17',lambda *a:None)
    f.dispatch('prepare',f.NAMES[0])
    assert calls[0][0][1]=={'timan1':332,'timan107':440}
    assert calls[0][1]['preserve_science'] and not calls[0][1]['reprepare']
    assert (r/'campaign.json').read_bytes()==data


def test_dynamic_prepare_refuses_already_launched_before_touching_its_directory(tmp_path,monkeypatch):
    r=tmp_path/'launched';r.mkdir();n.atomic(r/'campaign.json',{'domain':'metaworld'});n.atomic(r/'LAUNCH.json',{})
    before={p.name:p.read_bytes() for p in r.iterdir()}
    monkeypatch.setattr(f,'jobs',lambda name:[dict(run=str(r),pipeline='R17')])
    monkeypatch.setattr(f.formal_trial,'prepare',lambda *a,**kw:pytest.fail('must refuse before prepare lock/write'))
    with pytest.raises(ValueError,match='already launched'):f.dispatch('prepare',f.NAMES[0])
    assert {p.name:p.read_bytes() for p in r.iterdir()}==before


def test_external_unledgered_legacy_high_water_reserved_without_conda_double_count(tmp_path):
    proc=tmp_path/'proc'
    for pid,argv in ((10,b'python\0-m\0examples.libero.worker_entry\0'),
                     (11,b'python\0/opt/conda\0run\0python\0-m\0examples.libero.worker_entry\0')):
        p=proc/str(pid);(p/'task'/str(pid)).mkdir(parents=True)
        (p/'cmdline').write_bytes(argv)
        (p/'stat').write_text(str(pid)+' (worker) S '+'0 '*18+'123 0\n')
        (p/'status').write_text('VmRSS: 128 kB\nVmHWM: 256 kB\n')
        (p/'task'/str(pid)/'children').write_text('')
    workers=n.external_legacy(set(),1.15,proc)
    assert [w['pid'] for w in workers]==[10]
    assert workers[0]['rss']==128*1024 and workers[0]['projected_peak']>=256*1024*1.15
    assert not workers[0]['owned'] and not n.external_legacy({10},1.15,proc)


def governor(tmp_path,clock,name,available,**extras):
    c=dict(run_name=name,model='fastwam',domain='metaworld',worker_node='timan107',node_workers=440,
           autoscale={},autoscale_node_config=f.NODE_CONFIG,**extras)
    g=n.LocalGovernor(c,tmp_path/name,ledger=tmp_path/'ledger',clock=clock,
        reader=lambda:(available[0],{'full':{'avg10':0}}),process_reader=lambda *a:(GIB,GIB))
    n.atomic(g.history_path,[dict(peak=GIB,complete=True)])
    n.atomic(g.control_path,dict(campaign=name,time=clock.wall(),targets={'timan107':440}))
    return g


def test_t107_external_growth_prevents_observed_30_to_3_gib_incident(tmp_path,monkeypatch):
    clock=Clock();monkeypatch.setattr(n.time,'time',clock.wall)
    available=[107*GIB]
    # Measured: 64 unmanaged workers, RSS104GiB / summed HWM180GiB.
    external=[dict(pid=i,worker_id='legacy'+str(i),rss=104*GIB//64,
        peak=180*GIB//64,projected_peak=int(180*GIB/64*1.15),owned=False,born=0) for i in range(64)]
    monkeypatch.setattr(n,'external_legacy',lambda *a:external)
    g=governor(tmp_path,clock,'lane',available,autoscale_external_legacy_nodes=['timan107'])
    assert not g.tick()['admit'] and g.snapshot['external_legacy_workers']==64
    assert g.snapshot['external_legacy_growth']>100*GIB
    for amount in (107,80,60,31,3):
        available[0]=amount*GIB;clock.advance(2)
        with g.permit() as allowed:assert not allowed
    assert not g.entries  # no self-induced burst on top of untracked legacy growth


def test_cold_legacy_pid_turnover_does_not_release_reserved_future_growth(tmp_path,monkeypatch):
    clock=Clock();monkeypatch.setattr(n.time,'time',clock.wall);available=[200*GIB]
    workers=[dict(pid=i,worker_id='cold'+str(i),rss=GIB//4,peak=GIB//4,
        projected_peak=int(GIB/4*1.15),owned=False,born=0) for i in range(64)]
    monkeypatch.setattr(n,'external_legacy',lambda *a:workers)
    g=governor(tmp_path,clock,'lane',available,autoscale_external_legacy_nodes=['timan107'])
    assert not g.tick()['admit']
    assert g.snapshot['external_legacy_growth']==240*GIB  # 64*(4GiB-.25GiB)
    workers[0]['projected_peak']=5*GIB;clock.advance(2);g.tick()
    workers[0]['projected_peak']=GIB//4;clock.advance(2);g.tick()
    assert g.snapshot['external_legacy_growth']==304*GIB  # high-water survives recycling


def test_t107_ramp_is_one_per_10s_shared_between_dynamic_fleets(tmp_path,monkeypatch):
    clock=Clock();monkeypatch.setattr(n.time,'time',clock.wall)
    available=[200*GIB]
    a=governor(tmp_path,clock,'a',available);b=governor(tmp_path,clock,'b',available)
    with a.permit() as allowed:
        assert allowed;a.register(dict(pid=os.getpid(),start_ticks=n.identity(os.getpid()),worker_id='a'))
    with b.permit() as allowed:assert not allowed
    assert b.snapshot['reason']=='spawn_ramp_wait'
    clock.advance(9)
    a.tick()  # A live peer keeps publishing while B waits for the ramp token.
    with b.permit() as allowed:assert not allowed
    clock.advance(1)
    with b.permit() as allowed:
        assert allowed;b.register(dict(pid=os.getpid(),start_ticks=n.identity(os.getpid()),worker_id='b'))
    with a.permit() as allowed:assert not allowed


def test_existing_coordinator_guard_is_respected_by_dynamic_controller(tmp_path):
    from exp.offline_search.rounds.r17.autoscale_1017.driver import DriverController
    clock=Clock();c=dict(run_name='lane',workers=772,execution_worker_nodes={'timan1':332,'timan107':440},
                        worker_nodes={'timan1':332,'timan107':440},autoscale={})
    d=DriverController(NS(scheduler=NS(all_done=lambda:False)),c,tmp_path,clock=clock,wall=clock.wall)
    n.atomic(tmp_path/'WORKER_NODE_STATE.json',dict(nodes={'timan1':332},retired=['timan107'],phase='running'))
    clock.advance(2)
    for host,count in c['worker_nodes'].items():d.ingest(dict(node=node(clock,host,active=0,headroom=2000,target=count)))
    d.tick()
    assert d.targets['timan107']==d.demand_targets['timan107']==0
    clock.advance(20);d.tick();assert d.targets['timan107']==0


def test_accepted_rate_curve_is_per_node_and_scheduler_accepted_only(tmp_path):
    from exp.offline_search.rounds.r17.autoscale_1017.formal_dynamic_audit import curve
    from exp.offline_search.rounds.r17.autoscale_1017.driver import DriverController
    scope=tmp_path/'workers/timan107';scope.mkdir(parents=True)
    for t in (0,30,60,90):
        row=dict(time=t,node='timan107',delta=0,reason='node_sample',signals={},counts=dict(active=24,target=104))
        with (scope/'AUTOSCALE_SAMPLES.jsonl').open('a') as s:s.write(json.dumps(row)+'\n')
    rate=[dict(time=t,node='timan107',signals={'task_uid':str(t)},counts={},delta=0,reason='accepted_episode') for t in (20,65)]
    (scope/'AUTOSCALE_ACCEPTED.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rate))
    values,_=curve([scope],tmp_path/'curve.csv')
    assert [(r['time'],r['accepted_episodes_per_min']) for r in values]==[(0,0),(30,1),(60,1),(90,1)]
    assert all(r['active']==24 for r in values)
    clock=Clock();d=DriverController(NS(scheduler=NS(all_done=lambda:False)),
        dict(run_name='lane',workers=1,worker_nodes={'timan107':1},autoscale={}),tmp_path/'driver',clock=clock,wall=clock.wall)
    p=dict(_autoscale_episode=dict(node='timan107',task_uid='uid',attempt=1))
    d.result(p,False);assert not (tmp_path/'driver/AUTOSCALE_ACCEPTED.jsonl').exists()
    d.result(p,True);assert len((tmp_path/'driver/AUTOSCALE_ACCEPTED.jsonl').read_text().splitlines())==1


def test_native_filtered_continuation_has_298_missing_no_overlap():
    evidence=f.read(f.HERE/'followup_1018/formal_dynamic_122x/FILTERED_RESUME_REMOTE.json')
    root=Path('/data/mw_finetune/eval/fa_pi_f_f_l_l50')
    row=next(r for r in f.read(root/'arms.json') if r['arm'].endswith('_ir25'))
    all_tasks=f.legacy_plan(root,[row]);kept=set(evidence['uids'])
    after=f.legacy_plan(root,[row],done=kept)
    assert len(kept)==202 and len(after)==298 and len(all_tasks)==500
    assert not kept & {r['task_uid'] for r in after}
    assert kept | {r['task_uid'] for r in after} == {r['task_uid'] for r in all_tasks}


def test_partial_r15_prepare_preserves_old_closed_plan_and_only_stages_ops(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from exp.offline_search.closed_loop.ops.h100 import control
    from exp.offline_search.rounds.r17.autoscale_1017 import legacy_release
    from exp.offline_search.rounds.r17.filter_failed_knob import vla_shared
    root=tmp_path/'fa_partial';(root/'state').mkdir(parents=True);(root/'h100_sync').mkdir()
    n.atomic(root/'arms.json',[dict(arm='closed'),dict(arm='partial')])
    n.atomic(root/'h100_sync/synced.json',dict(worker_host='timan107',arms=['closed','partial']))
    (root/'state/closed.manifest_hash.DONE').write_text('closed')
    (root/'runs/partial').mkdir(parents=True);n.atomic(root/'runs/partial/h100_launch.json',dict(old='receipt'))
    before=(root/'h100_sync/synced.json').read_bytes();calls=[]
    monkeypatch.setattr(vla_shared,'r15',lambda *a:nullcontext())
    monkeypatch.setattr(control,'sync',lambda *a,**kw:pytest.fail('must not rewrite a closed serving plan'))
    monkeypatch.setattr(legacy_release,'stage',lambda *a,**kw:calls.append((a,kw)))
    job=dict(run=str(root),topology={'env':{'SERVER_HOST':'h100','PORTS':'23230,23231,23232','WPS':'13'}})
    f.legacy_job('prepare',job)
    assert calls[0][1]['maximum']==104
    assert (root/'h100_sync/synced.json').read_bytes()==before
    assert len(list((root/'runs/partial').glob('h100_launch.before_dynamic_*.json')))==1
    assert f.read(root/'FORMAL_DYNAMIC.json')['retained_done_arms']==['closed.manifest_hash.DONE']


def test_formal_audit_requires_unique_outcomes_recycling_and_covered_zero_oom(tmp_path,monkeypatch):
    from exp.offline_search.rounds.r17.autoscale_1017.formal_dynamic_audit import audit
    root=tmp_path/'formal';scope=root/'workers/timan107';scope.mkdir(parents=True);(root/'driver').mkdir()
    monkeypatch.setattr(f,'jobs',lambda name:[dict(run=str(root),pipeline='R17')])
    n.atomic(root/'FORMAL_DYNAMIC.json',dict(desired_maxima={'timan107':440}))
    n.atomic(root/'EPISODE_PLAN.json',dict(tasks=[dict(task_uid=str(i)) for i in range(64)]))
    n.atomic(root/'RECEIPT.json',dict(status='PASS',all_owned_closed=True))
    journal=root/'driver/journal.jsonl'
    journal.write_text(''.join(json.dumps(dict(task_uid=str(i),accepted=True,status='done',error=None))+'\n' for i in range(64)))
    for at in (0,65):
        n.event(scope/'AUTOSCALE_SAMPLES.jsonl','timan107',0,'node_sample',dict(minimum_available=100*GIB),dict(active=20,target=440))
    for i in range(2):n.event(scope/'AUTOSCALE_DECISIONS.jsonl','timan107',-1,'generation_exit',dict(complete=True),dict(active=20,target=440))
    for i in range(64):n.event(root/'AUTOSCALE_ACCEPTED.jsonl','timan107',0,'accepted_episode',dict(task_uid=str(i)),dict(accepted=1))
    n.atomic(scope/'AUTOSCALE_KERNEL.json',dict(status='PASS',coverage=True,oom_lines=[]))
    result=audit(f.NAMES[0],tmp_path/'audit');assert result['status']=='PASS' and result['recycled_generations']==2
    n.atomic(scope/'AUTOSCALE_KERNEL.json',dict(status='UNKNOWN',coverage=False,oom_lines=[]))
    assert not audit(f.NAMES[0],tmp_path/'audit')['checks']['zero_oom']
    with journal.open('a') as s:s.write(json.dumps(dict(task_uid='0',accepted=True,status='done',error=None))+'\n')
    assert not audit(f.NAMES[0],tmp_path/'audit')['checks']['every_uid_once']


def test_actual_gr_formal_prepare_uses_its_fixed_runtime_not_pre_longseg_parent(tmp_path,monkeypatch):
    import shutil
    import tarfile
    from contextlib import nullcontext
    from exp.r17_integration.ops import formal_eval as launch,driver_release,immutable_release,formal_ports,formal_prestage
    from exp.r17_integration.ops import direct_transfer
    from exp.offline_search.rounds.r17.autoscale_1017 import trial_runtime
    from exp.offline_search.rounds.r17.filter_failed import stage
    from exp.offline_search.rounds.r17.longseg_1017.runtime_fix import patched_files
    source=Path('/data/mw_finetune/eval/fa_ls_gr_f_p_m');root=tmp_path/'formal';root.mkdir()
    for name in ('campaign.json','runtime.tar','workers.tar','EPISODE_PLAN.json','WORKER_DEPLOYMENT.json'):
        if (source/name).exists():shutil.copy2(source/name,root/name)
    old=f.read(root/'campaign.json');expected=f.read(root/'EPISODE_PLAN.json')['tasks']
    # Start from the frozen pre-fix parent archive (byte-identical to this
    # campaign's own archive before any prepare), whatever the live dir's state.
    parent=Path(old['filter_failed_source']);unfixed=f.read(parent/'RUNTIME_RELEASE.json')
    shutil.copy2(parent/'runtime.tar',root/'runtime.tar')
    pin=immutable_release.sha(root/'runtime.tar');installs={};calls=[]
    def install(archive,path,archive_sha,identity):
        assert immutable_release.sha(archive)==archive_sha
        target=tmp_path/'installed'/Path(path).name
        doc=immutable_release.archive_manifest(archive)
        trial_runtime.install_checked(dict(root=str(target),archive=str(archive),document=doc,
            archive_sha256=archive_sha,identity=identity))
        installs[path]=str(target);calls.append((str(archive),path))
    native=driver_release.prepare
    def driver(c,r,checkout):
        native(c,r,checkout);c['driver_runtime']=installs[c['driver_runtime']]
        c['orchestration']['driver_runtime']=c['driver_runtime'];n.atomic(r/'ORCHESTRATION.json',c['orchestration'])
    monkeypatch.setattr(immutable_release,'install',install);monkeypatch.setattr(driver_release,'install',install)
    monkeypatch.setattr(driver_release,'prepare',driver)
    monkeypatch.setattr(formal_ports,'prepare_admission',lambda *a:None)
    # Only the long-segment fixed-runtime install may reach H100 (emulated).
    server='/tmp/r17_formal/'+old['run_name']+'/server';remotes,transfers=[],[]
    def remote(node,argv,**kw):
        assert node=='h100' and direct_transfer.BWLIMIT_KIBPS==4096
        remotes.append(argv);return json.dumps(dict(present=False)) if argv[1:2]==['-c'] else ''
    def transfer(argv,**kw):
        assert argv[:2]==['rsync','push'] and argv[3] in ('h100:'+server+'/runtime.tar','h100:'+server+'/release_install.py')
        transfers.append(argv)
    monkeypatch.setattr(launch,'remote',remote)
    monkeypatch.setattr(launch,'transfer',transfer)
    monkeypatch.setattr(launch,'retry_scope',lambda *a:nullcontext())
    monkeypatch.setattr(launch,'prepare_workers',lambda c,r,**kw:{n:{} for n in c['execution_worker_nodes']})
    monkeypatch.setattr(stage,'stage_server_inputs',lambda *a:None)
    monkeypatch.setattr(stage,'stage_worker_inputs',lambda *a:None)
    monkeypatch.setattr(formal_prestage,'seal_prepared',lambda c,r,d:n.atomic(r/'PREPARED.json',{'cpu_dry_run':True}))
    with f.private_on():
        f.formal_trial.prepare(root,f.desired(old),node_config=f.NODE_CONFIG,preserve_science=True)
    new=f.read(root/'campaign.json');fixed=f.read(root/'LONGSEG_RUNTIME_FIX.json')
    # Same fixed runtime as filter_failed.stage: never the frozen pre-fix archive.
    assert immutable_release.sha(root/'runtime.tar')==fixed['archive_sha256']!=pin
    assert new['runtime']==fixed['runtime']!=unfixed['root']
    assert new['runtime_identity']==fixed['identity']==immutable_release.identity(immutable_release.archive_manifest(root/'runtime.tar'))!=unfixed['identity']
    with tarfile.open(root/'runtime.tar') as tar:data={m.name:tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    assert fixed['members'] and all(data[m].decode()==text for m,text in patched_files(lambda m:data[m].decode()).items())
    assert transfers[0][2]==str(root/'runtime.tar') and fixed['identity'] in remotes[-1]
    assert new['orchestration']['source_runtime_identity']==new['runtime_identity']
    assert driver_release.planned_tasks(new,root)==expected
    assert new['execution_worker_nodes']=={'timan1':332,'timan107':440}
    assert (str(root/'runtime.tar'),new['runtime']) in calls and unfixed['root'] not in installs
