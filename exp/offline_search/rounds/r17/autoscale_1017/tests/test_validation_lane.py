import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from exp.offline_search.rounds.r17.autoscale_1017 import trial_lane as t


def test_r17_validation_subset_preserves_uids_indices_seeds_and_global_off(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'assets').mkdir()
    c=dict(run_name='original',model='fastwam',domain='metaworld')
    tasks=[dict(task_uid=f'uid:{i}',orig_init_state_idx=i%50,extra=dict(seed=42)) for i in range(500)]
    t.write(source/'campaign.json',c);t.write(source/'EPISODE_PLAN.json',dict(tasks=tasks))
    (source/'assets/full.yaml').write_text('original science\n')
    import tarfile,io
    with tarfile.open(source/'runtime.tar','w') as archive:
        data=b'def plan(c): return None\n';m=tarfile.TarInfo('exp/r17_integration/ops/formal_driver.py');m.size=len(data);archive.addfile(m,io.BytesIO(data))
    root=tmp_path/'fa_asv_r17';t.build(source,root,'r17')
    assert t.read(root/'EPISODE_PLAN.json')['tasks']==tasks[:256]
    assert t.read(root/'campaign.json')['semantic_campaign']=='original'
    raw=(t.BASE/'ENABLED').read_bytes()
    with t.validation_process(root):assert (t.BASE/'ENABLED').read_text().strip()=='on'
    assert (t.BASE/'ENABLED').read_bytes()==raw==b'off\n'
    assert t.read(source/'campaign.json')==c


def test_legacy_validation_keeps_scientific_config_and_first_256_pairs(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'config').mkdir()
    (source/'config/a.yaml').write_text('seed: 2026101401\n')
    (source/'config/m.yaml').write_text('model: pi05\n')
    pairs=[[i//50,i%50] for i in range(500)]
    t.write(source/'eval500.json',pairs)
    t.write(source/'arms.json',[dict(arm='original',yaml=str(source/'config/a.yaml'),matrix=str(source/'config/m.yaml'),manifest=str(source/'eval500.json'),seed=7)])
    root=tmp_path/'fa_asv_r15';t.build(source,root,'r15')
    assert t.read(root/'validation_manifest.json')==pairs[:256]
    assert t.read(root/'arms.json')[0]['arm']=='original'
    assert (root/'config/a.yaml').read_bytes()==(source/'config/a.yaml').read_bytes()
    assert t.read(root/'EPISODE_PLAN.json')['tasks'][255]['orig_init_state_idx']==5


def test_validation_never_enables_ordinary_campaign(tmp_path):
    root=tmp_path/'fa_pi_f_u_o';root.mkdir()
    t.write(root/'AUTOSCALE_VALIDATION.json',dict(schema='autoscale.owner_validation.v1',run=str(root),path='r15'))
    with pytest.raises(ValueError,match='isolated'): 
        with t.validation_process(root):pytest.fail('ordinary campaign entered ON context')


@pytest.mark.parametrize('busy', [False, True])
def test_acceptance_launch_waits_for_legacy_timan107_and_never_starts_it(monkeypatch,busy):
    import subprocess
    calls=[]
    def run(argv,**kw):
        calls.append(argv)
        assert argv[:7]==['tether','exec','--safe','--timeout','90s','timan107','--']
        assert argv[7:11]==['/usr/bin/python3','-B','-S','-c']
        compile(argv[-1],'<read-only-process-probe>','exec')
        return NS(stdout=json.dumps([dict(pid=42,argv=['worker'])] if busy else []))
    monkeypatch.setattr(subprocess,'run',run)
    if busy:
        with pytest.raises(RuntimeError,match='waits for LIBERO'):t.require_legacy_idle()
    else:t.require_legacy_idle()
    assert len(calls)==1


@pytest.mark.parametrize('listener',[False,True])
def test_rsync_reuseaddr_probe_matches_daemon_without_binding_any_real_port(tmp_path,monkeypatch,listener):
    from exp.offline_search.closed_loop.ops.h100 import control
    import socket,errno
    calls=[]
    class Sock:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def setsockopt(self,*args):calls.append(('reuse',args))
        def bind(self,address):
            assert calls==[('reuse',(socket.SOL_SOCKET,socket.SO_REUSEADDR,1))]
            calls.append(('bind',address))
            if listener:raise OSError(errno.EADDRINUSE,'live listener')
    monkeypatch.setattr(socket,'socket',Sock)
    monkeypatch.setenv('SYNC_PORT','23197')
    monkeypatch.setattr(control.time,'sleep',lambda *_:None)
    monkeypatch.setattr(control.subprocess,'Popen',lambda *a,**kw:NS(poll=lambda:None,terminate=lambda:calls.append(('terminate',)),wait=lambda **kw:None))
    if listener:
        with pytest.raises(OSError):
            with control.daemon(tmp_path,[]):pytest.fail('live listener accepted')
    else:
        with control.daemon(tmp_path,[]) as url:assert ':23197/' in url
    assert ('bind',('0.0.0.0',23197)) in calls


def test_legacy_shutdown_collects_once_not_on_ticks(tmp_path,monkeypatch):
    from exp.offline_search.rounds.r17.autoscale_1017 import legacy,kernel_evidence
    from unittest.mock import Mock
    g=object.__new__(legacy.AgentGovernor);g.agent=NS(_stop=NS(is_set=lambda:True))
    g.local=NS(run=tmp_path,started_wall=10);g.slots=NS(close=Mock())
    collect=Mock();monkeypatch.setattr(kernel_evidence,'collect',collect)
    g.supervise();collect.assert_called_once_with(tmp_path,10);g.slots.close.assert_called_once()


def test_validation_seals_on_only_inside_its_own_overlay(tmp_path):
    from exp.offline_search.rounds.r17.autoscale_1017.legacy_release import build
    root=tmp_path/'fa_asv_seal';root.mkdir()
    t.write(root/'AUTOSCALE_VALIDATION.json',dict(schema='autoscale.owner_validation.v1',run=str(root),path='r15'))
    with t.validation_process(root):
        release=build(root)
    sealed=Path(release['local'])/'exp/offline_search/rounds/r17/autoscale_1017/ENABLED'
    assert sealed.read_text().strip()=='on'
    import subprocess
    # run_arm.sh strips trailing newlines in command substitution, not spaces.
    probe=subprocess.run(['bash','-c','[ "$(cat "$1")" = on ]','probe',str(sealed)],capture_output=True)
    assert probe.returncode==0
    assert (t.BASE/'ENABLED').read_text().strip()=='off'
    assert b'__generation_' in (Path(release['local'])/'src/openpi/conductor/worker.py').read_bytes()


def test_legacy_evidence_join_is_bounded_and_missing_evidence_never_passes(tmp_path):
    from exp.offline_search.rounds.r17.autoscale_1017.legacy import AgentGovernor
    from unittest.mock import Mock
    g=object.__new__(AgentGovernor);g.local=NS(run=tmp_path)
    g.heartbeat=NS(join=Mock(),is_alive=lambda:True)
    g.close();g.heartbeat.join.assert_called_once_with(timeout=65)
    assert t.read(tmp_path/'AUTOSCALE_KERNEL.json')['status']=='UNKNOWN'


def test_legacy_validation_uses_native_full_pool_tasks_without_renumbering(tmp_path):
    from dataclasses import asdict
    from exp.gate_threshold_pareto.run_gtp import SweepStrategy
    from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import selected_tasks,load_manifest
    from openpi.conductor.task import ServerEndpoint
    s=SweepStrategy('libero_10',{'native_arm':'native.yaml'},50,parent_pool_sha256='original-pool')
    tasks=s._episodes('native_arm',ServerEndpoint('h100',23230))
    pairs=[[i//50,i%50] for i in range(256)];t.write(tmp_path/'pairs.json',pairs)
    selected=selected_tasks(tasks,load_manifest(tmp_path/'pairs.json'))
    assert [asdict(v) for v in selected]==[asdict(v) for v in tasks[:256]]
    assert all(v.extra['num_trials_per_task']==50 for v in selected)


@pytest.mark.parametrize('path',['r17','r15'])
def test_live_audit_requires_collected_coverage_and_uses_cumulative_memory_minimum(tmp_path,path):
    root=tmp_path/'fa_asv_audit';root.mkdir()
    t.write(root/'AUTOSCALE_VALIDATION.json',dict(path=path,episodes=2,
        task_uid_sha256=__import__('hashlib').sha256(b'a\nb').hexdigest()))
    t.write(root/'EPISODE_PLAN.json',dict(tasks=[dict(task_uid='a'),dict(task_uid='b')]))
    if path=='r17':
        journal=root/'driver/journal.jsonl';scope=root/'workers/timan1'
        t.write(root/'campaign.json',dict(execution_worker_nodes={'timan1':4}))
        t.write(root/'RECEIPT.json',dict(status='PASS',all_owned_closed=True))
    else:
        journal=root/'runs/native_arm/client/journal.jsonl';scope=journal.parent/'autoscale'
        (root/'state').mkdir();(root/'state/native_arm.DONE').write_text('done')
    journal.parent.mkdir(parents=True,exist_ok=True);scope.mkdir(parents=True)
    journal.write_text('\n'.join(json.dumps(dict(task_uid=uid,status='done',accepted=True,error=None)) for uid in ('a','b'))+'\n')
    events=[dict(time=i,node='timan1',delta=-1,reason='generation_exit',
        signals=dict(complete=True,available=30*1024**3),counts=dict(active=2-i,target=4)) for i in (0,1)]
    (scope/'AUTOSCALE_DECISIONS.jsonl').write_text('\n'.join(map(json.dumps,events))+'\n')
    assert t.audit(root)['checks']['zero_oom'] is False  # Missing evidence is never success.
    t.write(scope/'AUTOSCALE_KERNEL.json',dict(status='PASS',coverage=True,oom_lines=[]))
    sample=dict(time=2,node='timan1',reason='node_sample',delta=0,counts=dict(active=1,target=4),
        signals=dict(available=30*1024**3,minimum_available=9*1024**3))
    (scope/'AUTOSCALE_SAMPLES.jsonl').write_text(json.dumps(sample)+'\n')
    result=t.audit(root)
    assert result['status']=='FAIL' and not result['checks']['above_critical']
    sample['signals']['minimum_available']=25*1024**3
    (scope/'AUTOSCALE_SAMPLES.jsonl').write_text(json.dumps(sample)+'\n')
    result=t.audit(root);assert result['status']=='PASS'
    assert len((root/'AUTOSCALE_ACTIVE.csv').read_text().splitlines())==4
    with journal.open('a') as out:out.write(json.dumps(dict(task_uid='a',status='done',accepted=True))+'\n')
    assert not t.audit(root)['checks']['every_uid_once']

    t.write(root/'EPISODE_PLAN.json',dict(tasks=[dict(task_uid='a')]))
    assert not t.audit(root)['checks']['original_uid_plan_bound']
