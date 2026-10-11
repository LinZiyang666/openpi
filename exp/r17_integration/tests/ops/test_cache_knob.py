"""CPU planning and materialised B init routing; no campaign preparation."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import uuid

import numpy as np
import pytest
import yaml

from exp.r17_integration.ops import cache_knob as knob, cache_study as study
from exp.r17_integration.conductor.manifest import load_plan_config
from exp.r17_integration.common.recipe import bundle_identity, recipe_identity


@pytest.fixture(scope='module')
def dev_study(tmp_path_factory):
    directory=tmp_path_factory.mktemp('knob')
    gaps=directory/'gaps.jsonl'
    gaps.write_text(''.join(json.dumps(dict(model=model,suite=suite,size=size,
        levels={target:dict(gap=float(i+1)) for i,target in enumerate(list(knob.LEVELS.values())[:5])}))+'\n'
        for model in study.MODELS for suite in study.SUITES for size in knob.SIZES))
    return knob.build(directory/'dev',gaps=gaps),gaps


def fake_campaign(path,model,assets,*,rows=None):
    """Supply planner metadata and existing YAMLs without sealing/deploying."""
    manifest=study.validate_manifest(path)
    rows=copy.deepcopy(rows if rows is not None else manifest['models'][model])
    parent=study.read(Path(rows[0]['library'])/'READY.json')
    _,benchmark=study.profiles(model,parent)
    spec=replace(study.model_spec(parent['binding']['model']),allowed_benchmark_protocols=(benchmark.identity,))
    assets.mkdir()
    for row in rows:
        row['file']=str(path.parent/row['file'])
        config=load_plan_config(row['file'])
        row.update(bundle_id=study.size_label(row)+row['suite']+'_'+row['group'],
                   bundle_identity=bundle_identity(config),recipe_identity=recipe_identity(config))
    study.atomic_json(assets/'model_spec.json',asdict(spec))
    study.atomic_json(assets/'benchmark.json',benchmark.manifest())
    seal=dict(model=asdict(spec),benchmark_protocol=benchmark.identity,groups=rows,eval_pool=manifest['eval_pool'])
    seal['digest']=study.digest(seal)
    study.atomic_json(assets/'CACHE_STUDY.json',seal)
    return dict(model=model.removesuffix('_libero'),local_assets=str(assets),endpoint_host='127.0.0.1',
                public_port=23400,workers=2,run_name='knobcpu')


@pytest.mark.parametrize('model',study.MODELS)
def test_dev_plan_all_levels_unique_b_inits_and_official_protocol(dev_study,model,tmp_path):
    path,_=dev_study
    manifest=study.validate_manifest(path)
    assert len(manifest['models'][model])==72
    campaign=fake_campaign(path,model,tmp_path/'assets')
    benchmark,_,tasks,endpoints=study.plan(campaign)
    assert len(tasks)==len({t.task_uid for t in tasks})==21600
    assert benchmark.formal_indices==tuple(range(50)) and benchmark.dev_indices==()
    assert all(t.extra['benchmark_protocol']==benchmark.identity and endpoints[0].accepts(t) for t in tasks)
    assert {t.orig_init_state_idx for t in tasks}==set(range(30))
    assert all(t.extra['init_pool']=='B' and t.extra['partition']=='dev' and t.extra['serve_loeo'] for t in tasks)
    assert all(t.extra['logical_alias'] is None for t in tasks)
    assert {t.extra['library_size'] for t in tasks}==set(knob.SIZES)
    assert {t.extra['cache_group'].split('_')[1] for t in tasks}==set(knob.LEVELS)-{'pure'}
    assert all('n'+str(t.extra['library_size'])+'_'+t.extra['cache_group'] in t.task_uid for t in tasks)
    for row in manifest['models'][model]:
        assert row['settings']['commit']==(16 if model=='cosmos_libero' else 10)
        assert all(Path(row[k]).is_absolute() for k in ('library','fit','stall'))
        assert row['fit_digest']==study.read(Path(row['fit'])/'READY.json')['digest']
        assert row['stall_digest']==study.read(Path(row['stall'])/'READY.json')['digest']


def test_test_table_per_task_levels_including_pure_and_optional_indices(dev_study,tmp_path):
    dev,gaps=dev_study
    manifest=study.read(dev)
    table={}
    levels=list(knob.LEVELS)
    for suite in study.SUITES:
        row=next(r for r in manifest['models']['cosmos_libero'] if r['suite']==suite)
        names=[n for n in yaml.safe_load((dev.parent/row['file']).read_text())['recipe']['ordered_tasks'] if n.startswith(suite+'/')]
        table[suite]={str(size):{name:levels[(i+j)%len(levels)] for i,name in enumerate(names)} for j,size in enumerate(knob.SIZES)}
    table_path=tmp_path/'levels.json';study.atomic_json(table_path,table)
    path=knob.build(tmp_path/'test',gaps=gaps,model='cosmos_libero',table=table_path)
    campaign=fake_campaign(path,'cosmos_libero',tmp_path/'assets')
    _,_,tasks,_=study.plan(campaign)
    assert len(tasks)==len({t.task_uid for t in tasks})==6000
    assert all(t.extra['init_pool']=='A' and t.extra['partition']=='test' and not t.extra['serve_loeo'] for t in tasks)
    counts={}
    for task in tasks:
        key=(task.experiment,task.extra['library_size'],task.extra['task_name'])
        counts[key]=counts.get(key,0)+1
        level=table[key[0]][str(key[1])][key[2]]
        assert task.extra['cache_group']=='crg_'+level
        if level=='pure':assert task.extra['periodic_gap']==0.0
        if level=='koff':assert task.extra['periodic_gap'] is None
    assert len(counts)==120 and set(counts.values())=={50}
    row=study.read(path)['models']['cosmos_libero'][0]
    row['tasks']=row['tasks'][:1];row['indices']=[2,4,9]
    tiny=fake_campaign(path,'cosmos_libero',tmp_path/'tiny',rows=[row])
    assert [t.orig_init_state_idx for t in study.plan(tiny)[2]]==[2,4,9]


def test_incomplete_gaps_write_no_partial_study(tmp_path):
    gaps=tmp_path/'gaps.jsonl';gaps.write_text('')
    with pytest.raises(ValueError,match='missing 180 gaps'):
        knob.build(tmp_path/'dev',gaps=gaps)
    assert not (tmp_path/'dev').exists()


@pytest.mark.parametrize('model',study.MODELS)
def test_cpu_staging_and_seal_preserve_gap_loeo_tasks_indices_and_pool(dev_study,tmp_path,model):
    from exp.r17_integration.common.provider import AdapterProvider,register_provider
    path,_=dev_study
    manifest=study.read(path)
    base='crg' if model=='cosmos_libero' else 'chrg'
    rows=[r for r in manifest['models'][model] if r['library_size']==50 and r['group'] in (base+'_k30',base+'_koff')]
    for row in rows:
        row['file']=str(path.parent/row['file']);row['indices']=[29]
    manifest.update(models={model:rows},library_sizes=[50],groups=[base+'_k30',base+'_koff'])
    manifest.pop('digest');manifest['digest']=study.digest(manifest)
    source=tmp_path/'STUDY.json';study.atomic_json(source,manifest)
    parent=study.read(Path(rows[0]['library'])/'READY.json')
    _,benchmark=study.profiles(model,parent)
    spec=replace(study.model_spec(parent['binding']['model']),allowed_benchmark_protocols=(benchmark.identity,))
    provider=AdapterProvider('knob_seal_'+uuid.uuid4().hex,spec,lambda:None,lambda k:spec.default_sampler)
    register_provider(provider)
    assets=tmp_path/'assets';assets.mkdir()
    campaign=dict(model=model.removesuffix('_libero'),cache_study=str(source),assets=str(assets),local_assets=str(assets),
        server_node='local',suites=list(study.SUITES),run_name='cpu',endpoint_host='127.0.0.1',public_port=23400,workers=2)
    def forbid_remote(*args,**kwargs):raise AssertionError('CPU test attempted remote I/O')
    study.stage_inputs(campaign,tmp_path,SimpleNamespace(transfer=forbid_remote,remote=forbid_remote))
    assert campaign['eval_pool']=='B'
    assert set(campaign['bpool_transfer']['manifest']['alignment_digests'])==set(study.SUITES)
    study.seal(provider,campaign,benchmark)
    sealed=study.read(assets/'CACHE_STUDY.json')
    assert sealed['eval_pool']=='B'
    for row in sealed['groups']:
        config=load_plan_config(assets/row['file'])
        assert config.recipe.periodic_gap==(3.0 if row['group']==base+'_k30' else None)
        assert config.recipe.serve_loeo and row['indices']==[29]
    study.atomic_json(assets/'model_spec.json',asdict(spec))
    study.atomic_json(assets/'benchmark.json',benchmark.manifest())
    tasks=study.plan(campaign)[2]
    assert len(tasks)==40 and all(t.extra['init_pool']=='B' and t.orig_init_state_idx==29 for t in tasks)


@pytest.mark.parametrize('node',['timan107','timan1','local'])
@pytest.mark.parametrize('pool',['A','B'])
def test_worker_cli_routes_b_directory_through_every_wrapper(node,pool,monkeypatch):
    from exp.r17_integration.ops.formal_workers import worker_command
    from exp.r17_integration.ops import timan1_workers
    monkeypatch.setattr(timan1_workers,'route',lambda *args:None)
    campaign=dict(run_name='cpu',worker_node=node,worker_egl_device=0,endpoint='host:23400',
        driver_host='driver',driver_port=23419,domain='libero',worker_benchmark='benchmark.json',
        worker_max_episodes=4,cache_study='STUDY.json',eval_pool=pool,
        worker_init_states_dir='libero_spatial=/private/B/sp,libero_10=/private/B/l10')
    _,argv=worker_command(campaign,0,python='python')
    assert ('--init-states-dir' in argv)==(pool=='B')
    if pool=='B':assert argv[argv.index('--init-states-dir')+1]==campaign['worker_init_states_dir']


def test_official50_worker_setup_reads_installed_b_original_index(tmp_path,monkeypatch,request):
    """Run the actual init loader with a fake env and an existing B archive."""
    from exp.r17_integration.ops.library_init import install_init_archive
    from exp.r17_integration.ops.formal_eval import benchmark as official_benchmark
    from examples.libero.worker_entry import _build_episode_setup,parse_init_pools
    from openpi.conductor.task import EpisodeTask
    spec=official_benchmark(True)
    archive=sorted(Path('/data/r17_library').glob('dp_libero_libero_spatial/campaigns/teacher_*/bpool_inputs.tar'))[0]
    installed=install_init_archive(archive,tmp_path/'B',study.file_sha(archive))
    pools=installed['directories']
    assert set(pools)==set(study.SUITES)
    suites={}
    for suite in study.SUITES:
        name=next(n for n,_,s in spec.tasks if s==suite).split('/',1)[1]
        def forbid_a(*args):raise AssertionError('silently loaded A init states')
        suites[suite]=SimpleNamespace(get_task=lambda i,n=name:SimpleNamespace(name=n),get_task_init_states=forbid_a)
    # Import the existing native loader without importing a simulator.
    libero=ModuleType('libero');api=ModuleType('libero.libero');envs=ModuleType('libero.libero.envs')
    api.benchmark=SimpleNamespace(get_benchmark_dict=lambda:{s:(lambda s=s:suites[s]) for s in study.SUITES})
    api.get_libero_path=lambda *args:'';envs.OffScreenRenderEnv=object
    libero.libero=api
    for name,module in (('libero',libero),('libero.libero',api),('libero.libero.envs',envs)):
        monkeypatch.setitem(sys.modules,name,module)
    if 'examples.libero.main' not in sys.modules:
        def forget_native_import():
            sys.modules.pop('examples.libero.main',None)
            vars(sys.modules['examples.libero']).pop('main',None)
        request.addfinalizer(forget_native_import)
    from examples.libero import main as native
    monkeypatch.setattr(native,'_get_libero_env',lambda *args:(object(),'prompt'))
    pool_map=','.join(s+'='+p for s,p in pools.items())
    assert parse_init_pools(pool_map,'libero_spatial')==pools
    setup=_build_episode_setup(SimpleNamespace(task_suite_name='libero_spatial'),spec.seed,pool_map,'orig')
    for suite in study.SUITES:
        task=EpisodeTask('cpu','cpu','eval',suite,0,1,29,'host',23400,'bundle',extra={'init_pool':'B','partition':'dev'})
        _,initial,_,_=setup(task)
        actual_pool=native._load_init_states(suites[suite].get_task(0),suites[suite],0,pools[suite])
        np.testing.assert_array_equal(initial,actual_pool[29])
        assert not np.array_equal(initial,actual_pool[1])


@pytest.mark.parametrize('gap',[None,1.0,0.0])
@pytest.mark.parametrize('knob_call',[False,True])
def test_summary_accepts_periodic_policy_calls_only_when_configured(gap,knob_call,tmp_path):
    from exp.r17_integration.ops.cache_study_worker import extend_cost_row
    driver=tmp_path/'driver';driver.mkdir()
    extra=dict(cache_group='crg_pure',miss_mode='stall_only',corrector=True,num_steps=5,
               bundle_identity='bundle',recipe_identity='recipe',commit_horizon=16,periodic_gap=gap)
    study.atomic_json(tmp_path/'campaign.json',dict(cache_study=True,model='cosmos',replicas=1))
    study.atomic_json(tmp_path/'EPISODE_PLAN.json',dict(tasks=[dict(task_uid='cpu',experiment='libero_spatial',extra=extra)]))
    row=dict(task_uid='cpu',attempt=1,accepted=True,source='policy',decision_seq=0,cursor=0,
             stage_calls=[1,1,5],actual_nfe=5,bundle_identity='bundle',recipe_identity='recipe',commit_horizon=16,
             capture_enabled=False,stall_call=False,knob_call=knob_call,resident_pid=17,resident_generation='same')
    ack=dict(task_uid='cpu',attempt=1,accepted=True,_kind='execution_ack',decision_seq=0,
             executed_count=2,cursor_end=2,terminal=True,server_terminal_ack={'terminal':True})
    (driver/'journal.jsonl').write_text(json.dumps(dict(task_uid='cpu',attempt=1,status='done',accepted=True,success=True))+'\n')
    (driver/'per_step.jsonl').write_text(json.dumps(row)+'\n'+json.dumps(ack)+'\n')
    study.atomic_json(driver/'DRIVER_RECEIPT.json',dict(status='PASS',elapsed_seconds=1))
    result=study.aggregate(tmp_path)
    assert result['status']==('PASS' if gap is not None and knob_call else 'PARTIAL')
    assert result['alias_results']=={}
    extended=extend_cost_row(lambda *args:dict(base=True))
    assert extended({'__hit_meta__':{'knob_call':knob_call}},{'cache_study':True},0)['knob_call']==knob_call
    assert extended({'__hit_meta__':{'knob_call':knob_call}},{},0)=={'base':True}


def test_commands_preserve_lane_resources_and_print_placeholders():
    for model in study.MODELS:
        prepare,run=knob.commands('STUDY.json',model)
        profile=knob.cache_lane.PROFILES[model]
        assert '--server-node '+profile['node'] in prepare
        assert '--replicas 2' in prepare and '--max-batch '+str(profile['batch']) in prepare
        assert ('--mps' in prepare)==profile['mps']
        assert '--official-50' in prepare and '--seconds 518400' in prepare
        assert all('${'+name+'}' in prepare for name in ('RUN_NAME','WORKERS','WORKER_COUNTS','PORT','DRIVER_PORT'))
        assert '--prepared-run' in run and run.endswith('--run')
