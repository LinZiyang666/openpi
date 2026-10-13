"""CPU regressions for matched topology, immutable prestage and continuation."""
import copy
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import io

import pytest

from exp.r17_integration.ops import cache_lane as lane, cache_study as study
from exp.r17_integration.ops import cache_control, cache_resume
from exp.r17_integration.ops.formal_eval import parse_args
from exp.r17_integration.ops.formal_process import write_json
from exp.r17_integration.ops.build_client_bundle import add_bytes,build_client_bundle,verified_members


@pytest.mark.parametrize('model,short', [('dp_libero','dp'),('fastwam_libero','fastwam'),('cosmos_libero','cosmos')])
def test_smoke_exact_control_resources(model,short):
    text=subprocess.check_output(['bash',str(study.HERE/'smoke.sh'),short,'--print'],text=True)
    tokens=shlex.split(text);tokens=tokens[tokens.index('exp.r17_integration.ops.cache_study')+1:]
    assert tokens.pop(0)=='smoke';tokens.remove('--study');tokens.pop(0)
    args=parse_args(['--official-50','--cache-study',str(study.HERE/'yaml/STUDY.json'),
                    '--smoke','--smoke-episodes','2',*tokens])
    profile=lane.PROFILES[model]
    assert (args.replicas,args.max_batch,args.batch_wait_ms,args.mps,args.driver_mode)==(2,profile['batch'],5,profile['mps'],profile['mode'])
    assert (args.workers,args.worker_memory_mib,args.worker_vram_mib,args.worker_max_episodes)==(profile['workers'],profile['ram'],profile['vram'],4)
    assert args.replica_startup_width==profile['startup']
    assert args.episode_seconds==(3600 if short=='fastwam' else 1800)
    assert args.worker_egl_devices==(list(range(8)) if short=='fastwam' else None)
    assert args.worker_node=='timan107' and args.worker_restart_limit==2


def test_single_argument_smoke_default_print_is_safe():
    result=subprocess.run(['bash',str(study.HERE/'smoke.sh'),'dp'],capture_output=True,text=True)
    assert result.returncode==0 and '--run' in result.stdout


@pytest.mark.parametrize('model',study.LIBERO_MODELS)
def test_actual_closed_control_pins_and_changed_treatment_denied(model):
    old=study.read(lane.RUN_ROOT/study.CONTROLS[model]/'campaign.json')
    proof=cache_control.control_proof(lane.RUN_ROOT/study.CONTROLS[model],lane.RUN_ROOT/lane.RUNTIME_SOURCES[model],old)
    assert proof['completion']['accepted_episodes']==2000 and proof['non_r17_code_inherited']
    if model=='cosmos_libero':assert proof['recovery']['original_status']=='FAIL'
    changed=copy.deepcopy(old);changed['worker_vram_mib']+=1
    with pytest.raises(ValueError,match='worker_vram_mib'):
        cache_control.control_proof(lane.RUN_ROOT/study.CONTROLS[model],lane.RUN_ROOT/lane.RUNTIME_SOURCES[model],changed)


def test_lane_one_fleet_each_and_h100_order():
    dp=lane.jobs('4090',study.HERE/'yaml/STUDY.json','ct')
    h100=lane.jobs('h100',study.HERE/'yaml/STUDY.json','ct')
    assert [j['model'] for j in dp]==['dp_libero']
    assert [j['model'] for j in h100]==['fastwam_libero','cosmos_libero']
    for job in dp+h100:
        args=parse_args(job['argv']+['--prepare-only'])
        assert args.suites==list(study.SUITES) and not args.smoke and args.replicas==2
        assert args.runtime_from is None and args.cache_control_runtime.name==lane.RUNTIME_SOURCES[job['model']]


def test_control_rejects_summary_without_verified_native_execution(monkeypatch):
    control=lane.RUN_ROOT/study.CONTROLS['dp_libero'];old=study.read(control/'campaign.json')
    original=cache_control.read
    def changed(path):
        value=original(path)
        if Path(path).name=='SUMMARY.json':value['full_arm_execution_verified']=False
        return value
    monkeypatch.setattr(cache_control,'read',changed)
    with pytest.raises(ValueError,match='complete official'):
        cache_control.control_proof(control,control,old)


def test_control_overlay_freezes_every_non_r17_byte(tmp_path):
    def archive(path,files):
        with tarfile.open(path,'w') as t:
            for n,b in files.items():add_bytes(t,n,b)
    old=tmp_path/'old.tar';fresh=tmp_path/'fresh.tar';target=tmp_path/'cache.tar'
    archive(old,{'src/openpi/policy.py':b'control','exp/r17_integration/common/recipe.py':b'old'})
    data={'src/openpi/policy.py':b'UNRELATED WORKTREE CHANGE',
          'exp/r17_integration/common/recipe.py':b'reviewed cache admission'}
    data.update({n:b'frozen original Q3' for n in study.stall_runtime_files()});archive(fresh,data)
    result=cache_control.overlay_archive(old,fresh,target)
    with tarfile.open(target) as t:
        assert t.extractfile('src/openpi/policy.py').read()==b'control'
        assert t.extractfile('exp/r17_integration/common/recipe.py').read()==b'reviewed cache admission'
    assert result['non_r17_exact'] and 'src/openpi/policy.py' not in result['files']


def test_worker_overlay_keeps_baseline_simulator_and_client_bytes(tmp_path):
    root=tmp_path/'checkout';source=tmp_path/'control';source.mkdir();root.mkdir()
    oldroot=tmp_path/'oldroot';oldroot.mkdir()
    native='exp/r17_integration/conductor/worker_entry.py';ops='exp/r17_integration/ops/worker_lifetime.py'
    for name in [native,ops]:
        p=oldroot/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('original '+name)
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('CHANGED '+name)
    build_client_bundle(oldroot,[native,ops],source/'workers.tar')
    out=tmp_path/'out';out.mkdir()
    result=cache_control.worker_payload(dict(cache_control_runtime=str(source)),out,root,[native,ops])
    _,content=verified_members(out/'workers.tar','PAYLOAD.json')
    assert content[native]==('original '+native).encode() and content[ops]==('CHANGED '+ops).encode()
    assert native in result['manifest']['files']


def fake_failed(tmp_path):
    path=tmp_path/'failed';(path/'driver').mkdir(parents=True);(path/'assets').mkdir()
    extra=dict(bundle_identity='bundle',recipe_identity='recipe',commit_horizon=5,num_steps=100,
               miss_mode='never',corrector=False)
    tasks=[dict(task_uid='a',extra=extra),dict(task_uid='b',extra=extra),dict(task_uid='c',extra=extra)]
    write_json(path/'campaign.json',dict(cache_study='study',smoke=False,model='dp',run_name='old'))
    write_json(path/'RECEIPT.json',dict(status='FAIL',all_owned_closed=True,cleanup=[dict(closure_confirmed=True)]))
    write_json(path/'EPISODE_PLAN.json',dict(tasks=tasks))
    write_json(path/'assets/CACHE_STUDY.json',{})
    for name in ['runtime.tar','workers.tar','PREPARED.json']: (path/name).write_text('sealed')
    write_json(path/'driver/DRIVER_RECEIPT.json',dict(status='PARTIAL'))
    endings=[dict(task_uid=u,attempt=1,accepted=True,status='failed',success=False,error=None) for u in ['a','b']]
    attempts=[dict(r,n_steps=5) for r in endings]
    decision=dict(task_uid='a',attempt=1,accepted=True,decision_seq=0,cursor=0,source='library',stage_calls=[1,0,0],
                  actual_nfe=0,capture_enabled=False,resident_pid=123,resident_generation='gen',
                  corrector_enabled=False,miss_mode='never',bundle_identity='bundle',recipe_identity='recipe',commit_horizon=5)
    ack=dict(task_uid='a',attempt=1,accepted=True,_kind='execution_ack',decision_seq=0,executed_count=5,cursor_end=5,
             terminal=True,server_terminal_ack=dict(terminal=True,decision_seq=0,executed_count=5,
                 cursor_end=5,bundle_identity='bundle',request=dict(attempt=1,decision_seq=0)))
    for name,rows in [('journal.jsonl',endings),('attempts.jsonl',attempts),('per_step.jsonl',[decision,ack,dict(decision,task_uid='b')])]:
        (path/'driver'/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return path


def test_resume_keeps_legitimate_failure_drops_torn_and_missing(tmp_path):
    source=fake_failed(tmp_path);selection=cache_resume.select(source)
    assert selection['retained_task_uids']==['a'] and selection['missing_task_uids']==['b','c']
    assert not selection['success_selection']
    assert selection['dropped'][0]['capture_rows']==1
    campaign=dict(cache_resume_selection=selection)
    cache_resume.check(campaign)
    (source/'driver/per_step.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='source changed'):cache_resume.check(campaign)


@pytest.mark.parametrize('change',[dict(server_terminal_ack=False),dict(cursor_end=6),dict(executed_count=-1)])
def test_resume_refuses_incomplete_server_ack(tmp_path,change):
    source=fake_failed(tmp_path);path=source/'driver/per_step.jsonl'
    rows=cache_resume.lines(path);rows[1].update(change)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert cache_resume.select(source)['retained_task_uids']==[]


def test_resume_requires_confirmed_closure(tmp_path):
    source=fake_failed(tmp_path);write_json(source/'RECEIPT.json',dict(status='FAIL',all_owned_closed=False))
    with pytest.raises(ValueError,match='closed'):cache_resume.select(source)


def test_unchanged_single_L_reuses_existing_sealed_artifacts(tmp_path):
    source=study.HERE/'yaml/STUDY.json'
    assert lane.select_horizons(source,tmp_path/'unused',10,10)==source.resolve()
    assert not (tmp_path/'unused').exists()
    with pytest.raises(ValueError,match='native chunk'):lane.select_horizons(source,tmp_path/'bad',100,10)


@pytest.mark.parametrize('flag,value',[('--long-dp','5'),('--long-dp','12'),('--long-fast','15')])
def test_lane_cli_refuses_unregistered_L_before_any_preparation(tmp_path,flag,value):
    with pytest.raises(SystemExit) as error:
        lane.main(['--lane','4090','--directory',str(tmp_path/'never'),flag,value,'--prepare-only'])
    assert error.value.code==2 and not (tmp_path/'never').exists()


def test_new_L_is_one_parameter_for_all_long_groups(tmp_path,monkeypatch):
    captured={}
    monkeypatch.setattr(study,'fit_inputs',lambda path:captured.update(study.validate_manifest(path)))
    path=lane.select_horizons(study.HERE/'yaml/STUDY.json',tmp_path/'L15',15,10)
    assert captured['horizon_selection']['dp']==15
    for model,rows in captured['models'].items():
        for row in rows:
            expected=study.group_settings(model,row['group'],15 if model=='dp_libero' else 16 if model=='cosmos_libero' else 10)['commit']
            assert row['settings']['commit']==expected
            if model=='dp_libero' and 'h' in row['group']:assert 'C15' in row['fit']


def test_continuation_preparation_preserves_original_and_quarantines_partial(tmp_path,monkeypatch):
    import shutil
    from exp.r17_integration.ops import queue_retry
    source=fake_failed(tmp_path)
    original={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    target=tmp_path/'new'
    def cloned(snapshot,name,failure):
        assert snapshot==source/'.startup_inputs' and failure['cache_continuation']
        target.mkdir();shutil.copytree(source/'assets',target/'assets')
        for n in ('campaign.json','EPISODE_PLAN.json','runtime.tar','workers.tar'):shutil.copyfile(source/n,target/n)
        for n in ('PREFLIGHT.json','WORKER_DEPLOYMENT.json'):write_json(target/n,{})
        c=study.read(target/'campaign.json');c.update(run_name=name,runtime='frozen',runtime_identity='pin')
        write_json(target/'campaign.json',c)
        return target
    monkeypatch.setattr(queue_retry,'clone',cloned)
    assert cache_resume.prepare(source,'new')==target
    assert [t['task_uid'] for t in study.read(target/'EPISODE_PLAN.json')['tasks']]==['b','c']
    assert len(study.read(target/'FULL_EPISODE_PLAN.json')['tasks'])==3
    assert study.read(target/'campaign.json')['cache_plan_name']=='old'
    assert not study.read(target/'CACHE_RESUME_QUARANTINE.json')['library_or_summary_acceptance']
    assert {str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}==original
    assert 'CACHE_RESUME_QUARANTINE.json' in study.read(target/'PREPARED.json')['files']


def test_zero_progress_with_no_step_file_can_resume_without_fabricating(tmp_path):
    source=fake_failed(tmp_path)
    (source/'driver/per_step.jsonl').unlink()
    selection=cache_resume.select(source)
    assert not selection['retained_task_uids'] and len(selection['missing_task_uids'])==3
    cache_resume.check(dict(cache_resume_selection=selection))
    (source/'driver/per_step.jsonl').write_text('late orphan evidence')
    with pytest.raises(ValueError,match='source changed'):
        cache_resume.check(dict(cache_resume_selection=selection))


def test_cache_startup_retry_only_when_zero_acceptance(tmp_path):
    from exp.r17_integration.ops.queue_retry import zero_acceptance
    source=fake_failed(tmp_path)
    for n in ('journal.jsonl','attempts.jsonl','per_step.jsonl'):(source/'driver'/n).write_text('')
    assert zero_acceptance(source)['accepted_episodes']==0
    (source/'driver/attempts.jsonl').write_text(json.dumps(dict(accepted=True,error='EGL'))+'\n')
    with pytest.raises(ValueError,match='accepted'):zero_acceptance(source)


@pytest.mark.parametrize('model,c',[('dp_libero',15),('dp_libero',16),('fastwam_libero',20)])
def test_selected_L_requires_actual_matching_fitted_head(model,c,tmp_path,monkeypatch):
    from exp.r17_integration.tests.ops import test_cache_study as fixtures
    from exp.r17_integration.common.provider import binding_for
    from exp.r17_integration.search.cache_recipe import verified_fit
    from exp.r17_integration.search.store import Store
    provider=fixtures.fixture_model(model)
    if c%provider.model_spec.layout.replan_steps:
        original=fixtures.config_for
        def partial(provider,*,k,commit):
            config=original(provider,k=k,commit=provider.model_spec.layout.replan_steps)
            config.adapter_binding=binding_for(provider,'fake_mw.v1',num_steps=k,commit_horizon=commit,allow_partial_commit=True)
            return config
        monkeypatch.setattr(fixtures,'config_for',partial)
    config,fitted=fixtures.freeze(provider,tmp_path,c)
    assert fitted.commit==c
    binding=Store(config.recipe.store_path).manifest['binding']
    assert verified_fit(config,binding).commit==c
    stale=copy.deepcopy(config);stale.adapter_binding['commit_horizon']=10
    with pytest.raises(ValueError,match='commit'):verified_fit(stale,binding)


@pytest.mark.parametrize('cache,mode',[(True,'group-commit'),(True,'legacy'),(False,'group-commit'),(False,'legacy')])
def test_cache_orchestration_refresh_preserves_native_bytes_and_is_default_off(tmp_path,monkeypatch,cache,mode):
    from exp.r17_integration.ops import driver_release
    from exp.r17_integration.ops.immutable_release import archive_manifest,identity,sha
    archive=tmp_path/'runtime.tar'
    with tarfile.open(archive,'w') as t:
        add_bytes(t,'src/openpi/policies/native.py',b'control native implementation')
        for name in (*driver_release.OVERLAY,*driver_release.CACHE_OVERLAY):add_bytes(t,name,b'archived CPU orchestration')
    before=sha(archive);campaign=dict(runtime_identity=identity(archive_manifest(archive)),driver_mode=mode)
    if cache:campaign['cache_study']='sealed'
    installed={}
    def install(path,*args):installed.update(archive_manifest(path));return {}
    monkeypatch.setattr(driver_release,'install',install)
    driver_release.prepare(campaign,tmp_path,study.ROOT)
    document=study.read(tmp_path/'ORCHESTRATION.json')
    assert installed['files']['src/openpi/policies/native.py']==archive_manifest(archive)['files']['src/openpi/policies/native.py']
    assert sha(archive)==before and document['model_code_unchanged'] and document['worker_payload_unchanged']
    delta=set(document['overlay_files'])
    assert bool(delta & set(driver_release.CACHE_OVERLAY)) is cache
    if not cache and mode=='legacy':assert not delta
