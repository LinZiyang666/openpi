"""K1 VLA knob preparation, dispatch and selection rules; CPU stubs only."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from exp.r17_integration.ops import cache_knob as knob, cache_study as study, formal_eval
from exp.r17_integration.common.config import build_shared_storage, load_cache_config
from exp.r17_integration.common.contracts import SampleResult
from exp.r17_integration.common.provider import AdapterProvider, binding_for, register_provider
from exp.r17_integration.common.serving import make_connection_factory
from exp.r17_integration.common.fake_adapter import FakeAdapter
from exp.r17_integration.models.pi05.adapter import sampler_for_steps as pi_sampler
from exp.r17_integration.models.groot.manifest import sampler_for_steps as gr_sampler
from exp.r17_integration.tests.ops.test_libero_vla import assets, private_provider_registry, spec_for
from exp.r17_integration.tests.ops.test_cache_study import fixture_model, freeze
from exp.offline_search.rounds.r17.knob import common as c, descent, test as analysis
from exp.offline_search.rounds.r17.knob.test_knob import lane, name

MODELS=('pi05_libero','groot_libero')
SAMPLERS={'pi05_libero':pi_sampler,'groot_libero':gr_sampler}


@pytest.mark.parametrize('model',MODELS)
@pytest.mark.parametrize('pool,count',(('B',30),('A',50)))
def test_formal_cli_k1_reference_metadata_and_fixed_groot_startup(tmp_path,monkeypatch,model,pool,count):
    monkeypatch.setattr(formal_eval,'FORMAL_ROOT',tmp_path)
    seen=[]
    monkeypatch.setattr(formal_eval,'prepare',lambda campaign,directory:seen.append(dict(campaign)))
    def forbid(*args,**kwargs):raise AssertionError('CPU CLI test attempted a launch')
    monkeypatch.setattr(formal_eval,'run_campaign',forbid)
    monkeypatch.setattr(formal_eval,'open_tunnels',forbid)
    assert formal_eval.main(['--model',model,'--official-50','--suites',*study.SUITES,
        '--denoising-steps','1','--eval-pool',pool,'--init-count',str(count),
        '--run-name','cpu_reference','--prepare-only'])==0
    assert seen[0]['denoising_steps']==1 and seen[0]['init_count']==count
    assert seen[0].get('eval_pool','A')==pool
    if model=='groot_libero':assert seen[0]['live_k']==1


def source_study(tmp_path,model):
    """Metadata-only source fits: no renderer, weights or outcome reads."""
    b=formal_eval.benchmark()
    spec=spec_for(model,b)
    for suite in study.SUITES:
        library=tmp_path/'libraries'/(model+'_'+suite)/'bpool'
        library.mkdir(parents=True)
        study.atomic_json(library/'READY.json',dict(binding={'model':asdict(spec),'sampler':asdict(spec.default_sampler),
            'benchmark_protocol':b.identity},content_digest='CPU-parent'))
        for n in knob.SIZES:
            view=library/'subsets'/('size_'+str(n));view.mkdir(parents=True)
            study.atomic_json(view/'VIEW.json',dict(size=n,content_digest='CPU-selected-'+str(n)))
    dest=tmp_path/'source'
    study.generate(dest,library_root=tmp_path/'libraries',models=(model,),groups=('chrg',),sizes=knob.SIZES)
    gaps=tmp_path/'gaps.jsonl'
    gaps.write_text(''.join(json.dumps(dict(model=model,suite=s,size=n,
        levels={str(t):dict(gap=i+1.) for i,t in enumerate((.5,.4,.3,.2,.1))}))+'\n'
        for s in study.SUITES for n in knob.SIZES))
    return dest/'STUDY.json',gaps,b


@pytest.mark.parametrize('model',MODELS)
def test_dev_all_sizes_k1_full_system_and_test_partial(tmp_path,model):
    source,gaps,b=source_study(tmp_path,model)
    dev=knob.build(tmp_path/'dev',sources=[source],gaps=gaps,model=model,miss_steps='reduced')
    manifest=study.validate_manifest(dev);rows=manifest['models'][model]
    assert manifest['eval_pool']=='B' and len(rows)==74
    assert sum(len(r['indices'])*10 for r in rows)==22200
    pure=[r for r in rows if r['group'].endswith('_pure')]
    assert len(pure)==2 and all(r['library_size']==500 for r in pure)
    table={c.model_label(model):{s:{str(n):{task:c.LEVELS[i%7] for i,task in enumerate(b.ordered_tasks) if task.startswith(s+'/')}
                               for n in knob.SIZES} for s in study.SUITES}}
    path=tmp_path/'levels.json';study.atomic_json(path,table)
    test=knob.build(tmp_path/'test',sources=[source],gaps=gaps,model=model,table=path,miss_steps='reduced')
    test_rows=study.read(test)['models'][model]
    assert sum(50*len(r['tasks']) for r in test_rows)==6000
    for path,entries,loeo in ((dev,rows,True),(test,test_rows,False)):
        for row in entries:
            doc=yaml.safe_load((path.parent/row['file']).read_text())
            assert doc['adapter_binding']['num_steps']==1
            assert doc['adapter_binding']['sampler_identity']==SAMPLERS[model](1).identity
            assert doc['adapter_binding']['commit_horizon']==10
            assert doc['recipe']['corrector'] and doc['recipe']['miss_mode']=='stall_only'
            assert doc['recipe']['serve_loeo'] is loeo
            assert doc['recipe']['teacher_sampler_identity']!=SAMPLERS[model](1).identity
            if row['group'].endswith('_pure'):assert doc['recipe']['periodic_gap']==0
            if row['group'].endswith('_koff'):assert doc['recipe']['periodic_gap'] is None
    task=b.ordered_tasks[0]
    study.atomic_json(path,{c.model_label(model):{'libero_spatial':{'50':{task:'pure'}}}})
    retest=knob.build(tmp_path/'retest',sources=[source],gaps=gaps,model=model,table=path,partial=True,miss_steps='reduced')
    retest_rows=study.read(retest)['models'][model]
    assert len(retest_rows)==1 and retest_rows[0]['tasks']==[task]
    assert retest_rows[0]['group']=='chrg_pure'


@pytest.mark.parametrize('model',MODELS)
def test_pure_baseline_b30_a50_and_worker_b_inputs(tmp_path,model):
    from exp.r17_integration.ops.formal_driver import plan
    from exp.r17_integration.ops.formal_workers import worker_command
    from exp.r17_integration.ops.library_init import prepare_b_inputs
    campaign,p=assets(tmp_path/'assets',model)
    campaign.update(denoising_steps=1,suites=list(study.SUITES))
    spec=p.model_spec
    p=replace(p,sampler_for_steps=SAMPLERS[model])
    config=formal_eval.full_config(p,campaign)
    config.recipe.ordered_tasks=list(formal_eval.benchmark(True).ordered_tasks)
    config.recipe.task_table_identity=formal_eval.benchmark(True).task_table_identity
    (tmp_path/'assets/full.yaml').write_text(yaml.safe_dump(asdict(config)))
    study.atomic_json(tmp_path/'assets/execution_sampler.json',asdict(SAMPLERS[model](1)))
    campaign.update(eval_pool='B',init_count=30)
    _,_,tasks,endpoints=plan(campaign)
    assert len(tasks)==600 and {t.orig_init_state_idx for t in tasks}==set(range(30))
    assert all(t.extra['partition']=='dev' and t.extra['init_pool']=='B' and t.extra['num_steps']==1 for t in tasks)
    assert endpoints[0].allowed_k==(1,)
    prepare_b_inputs(campaign,tmp_path)
    assert len(campaign['bpool_transfer']['manifest']['files'])==20
    assert prepare_b_inputs(campaign,tmp_path)==campaign['bpool_transfer']
    campaign.update(endpoint='unused:23160',driver_host='unused',driver_port=23169,worker_node='timan107',worker_benchmark='benchmark.json',worker_egl_device=0,
        worker_init_states_dir='libero_spatial=/private/B/sp,libero_10=/private/B/l10')
    argv=worker_command(campaign,0,python='python')[1]
    assert argv[argv.index('--init-states-dir')+1]==campaign['worker_init_states_dir']
    campaign.pop('eval_pool');campaign.pop('init_count')
    _,_,tasks,_=plan(campaign)
    assert len(tasks)==1000 and {t.orig_init_state_idx for t in tasks}==set(range(50))


@pytest.mark.parametrize('model',MODELS)
@pytest.mark.parametrize('trigger',('stall','periodic','pure'))
def test_every_model_call_is_k1(model,trigger,tmp_path):
    provider=fixture_model(model)
    sampler=SAMPLERS[model]
    spec=replace(provider.model_spec,default_sampler=sampler(provider.model_spec.default_sampler.num_steps))
    received=[]
    class Stub(FakeAdapter):
        def sample(self,s2,k,rng,capture):
            received.append(k)
            assert k==sampler(1) and not capture.enabled
            return SampleResult(np.zeros_like(s2.data),1,k)
    provider=replace(provider,provider_id=provider.provider_id+'.k1',model_spec=spec,
        load_adapter=lambda:Stub(spec),sampler_for_steps=sampler)
    register_provider(provider)
    cfg,_=freeze(provider,tmp_path,10,gated=True)
    cfg.adapter_binding=binding_for(provider,spec.allowed_benchmark_protocols[0],num_steps=1,commit_horizon=10)
    cfg.recipe.periodic_gap=None if trigger=='stall' else 0. if trigger=='pure' else 1.
    policy=make_connection_factory(provider,cfg,build_shared_storage(cfg))()
    class Tracker:
        def __init__(self,*args):pass
        def observe(self,*args):pass
        def status(self):return dict(state='slow_confirmed' if trigger=='stall' else 'ok',W=2)
    policy.recipe.tracker_class=Tracker
    policy.on_episode_start(extra_metadata=dict(task_name='reach',task_id=0,original_init_idx=0))
    for seq in range(3):
        obs=dict(state=np.zeros(8,np.float32))
        if seq:obs['__execution__']=dict(decision_seq=seq-1,executed_count=5)
        hit=policy.infer(obs)['__hit_meta__']
        if hit['source']=='policy':
            assert hit['actual_nfe']==1 and hit['stage_calls']==(1,1,1)
            assert hit['sampler_identity']==sampler(1).identity
    assert received and (len(received)==3 if trigger=='pure' else True)


@pytest.mark.parametrize('model',MODELS)
def test_printed_vla_commands_use_formal_b1_topology(model,tmp_path):
    source,gaps,_=source_study(tmp_path,model)
    path=knob.build(tmp_path/'dev',sources=[source],gaps=gaps,model=model)
    prep,run=knob.commands(path,model,run_name='cpu_k1')
    tokens=shlex.split(prep);tokens=tokens[tokens.index('exp.r17_integration.ops.formal_eval')+1:]
    args=formal_eval.parse_args(tokens)
    p=knob.VLA_PROFILES[model]
    assert args.replicas==p['replicas'] and args.workers==p['workers'] and args.server_node==p['node']
    assert args.max_batch==1 and args.mps and args.driver_port==23169
    assert args.prepare_only and '--cache-control' not in prep and '--denoising-steps' not in prep
    if model=='pi05_libero':assert 'taskset -c 22-43,66-87' in prep and 'taskset -c 22-43,66-87' in run


@pytest.mark.parametrize('model',MODELS)
def test_real_fit_calibration_b_stage_k1_and_own_episode_exclusion(tmp_path,monkeypatch,model):
    from exp.r17_integration.debug.validate import model_binding
    from exp.r17_integration.search.teacher_library import build_teacher_parent, fit_nested_libraries
    from exp.r17_integration.tests.search.library_fixture import synthetic_capture
    from exp.r17_integration.common.recipe import make_recipe_factory
    from exp.r17_integration.common.query_view import QueryView
    from exp.r17_integration.common.contracts import RequestContext
    monkeypatch.setattr(knob,'SIZES',(500,))
    b=formal_eval.benchmark()
    spec=spec_for(model,b)
    spec=replace(spec,feature_spec=replace(spec.feature_spec,fields={'vision_0':3,'vision_1':3,'robot_state':8}))
    binding=model_binding(spec,b.identity)
    for suite in study.SUITES:
        source=tmp_path/('capture_'+suite)
        _,episodes,audit,names=synthetic_capture(source,suite=suite,binding_override=binding,capture_images=False)
        dest=tmp_path/'libraries'/(model+'_'+suite)/'bpool'
        parent=build_teacher_parent([source],dest,binding,episodes,audit,capture_images=False)
        fit_nested_libraries(parent,audit,task_names=names,sizes=(500,))
    source=tmp_path/'source'
    study.generate(source,library_root=tmp_path/'libraries',models=(model,),groups=('chrg',),sizes=(500,))
    study.fit_inputs(source/'STUDY.json',bundle=model)
    gaps=knob.calibrate(tmp_path/'gaps.jsonl',sources=[source/'STUDY.json'],model=model)
    gap_rows=[json.loads(row) for row in gaps.read_text().splitlines()]
    assert len(gap_rows)==2 and all(set(r['levels'])=={'0.5','0.4','0.3','0.2','0.1'} for r in gap_rows)
    dev=knob.build(tmp_path/'dev',sources=[source/'STUDY.json'],gaps=gaps,model=model)
    campaign=dict(model=model.removesuffix('_libero'),domain='libero',server_node='local',cache_study=str(dev),
        assets=str(tmp_path/'assets'),local_assets=str(tmp_path/'assets'),run_name='cpu',suites=list(study.SUITES))
    Path(campaign['assets']).mkdir()
    def forbid(*args,**kwargs):raise AssertionError('CPU stub attempted network')
    study.stage_inputs(campaign,tmp_path,SimpleNamespace(remote=forbid,transfer=forbid))
    assert campaign['eval_pool']=='B' and campaign['bpool_transfer']['manifest']['init_pool']=='B'
    assert all(r.get('num_steps')==1 for r in study.read(Path(campaign['assets'])/'cache_inputs/manifest.json')['groups'])
    serving=formal_eval.benchmark(True)
    live=replace(spec,allowed_benchmark_protocols=(serving.identity,),default_sampler=SAMPLERS[model](1))
    provider=AdapterProvider('cpu_live_'+model,live,lambda:None,SAMPLERS[model]);register_provider(provider)
    study.seal(provider,campaign,serving)
    sealed=study.read(Path(campaign['assets'])/'CACHE_STUDY.json')
    row=next(r for r in sealed['groups'] if r['suite']=='libero_spatial' and r['group']=='chrg_koff')
    cfg=load_cache_config(Path(campaign['assets'])/row['file'])
    assert cfg.adapter_binding['num_steps']==1 and cfg.recipe.serve_loeo
    recipe=make_recipe_factory(provider)(cfg,None)
    library=tmp_path/'libraries'/(model+'_libero_spatial')/'bpool'
    ready=study.read(library/'READY.json');first=ready['rows'][0]
    i=0
    keys={field:np.load(library/ready['arrays']['key_'+field]['file'])[i].copy() for field in live.feature_spec.fields}
    state=np.load(library/ready['arrays']['state']['file'])[i].copy()
    req=RequestContext('cpu',serving.identity,first['task_name'],first['orig_init_idx'],0,task_id=0)
    result=recipe.query(QueryView(req,keys,state))
    assert all(int(recipe.base.episode[j])!=first['library_episode_id'] for j in result.rows)
    cfg.recipe.serve_loeo=False
    other=make_recipe_factory(provider)(cfg,None)
    result=other.query(QueryView(req,keys,state))
    assert any(int(other.base.episode[j])==first['library_episode_id'] for j in result.rows)


@pytest.mark.parametrize('model',c.VLA_MODELS)
def test_descent_uses_k1_b_reference_and_stops_at_first_failure(tmp_path,model):
    baseline=tmp_path/'pure_b'
    lane(baseline,model,'B',[(t,'pure',i,i>=6) for t in range(10) for i in range(30)],control=True)
    entries=[(t,level,i,i>=6+(2 if level=='k40' else 0)) for t in range(10)
             for level in ('k50','k40') for i in range(30)]
    run=tmp_path/'dev';lane(run,model,'B',entries)
    result=descent.main(['--lane',str(run),'--reference-lane',str(baseline),'--out',str(tmp_path/'out')])
    task=result['cells'][0]['tasks'][name('libero_spatial',0)]
    assert task['levels']['pure']['successes']==24
    assert task['levels']['k50']['passes']['5'] and task['selected']=='k50'
    assert task['first_failure']=='k40' and not task['incomplete']


@pytest.mark.parametrize('model',c.VLA_MODELS)
def test_test_raises_only_failed_tasks_and_pure_stays(tmp_path,model):
    controls=tmp_path/'controls'
    lane(controls/c.REDUCED_CONTROL_NAMES[model][-1],model,'A',[(t,'pure',i,True) for t in range(10) for i in range(50)],control=True)
    first=tmp_path/'first'
    levels=['k30','k30','k30','pure']+['koff']*6
    deficits=[5,4,2,3]+[0]*6
    lane(first,model,'A',[(t,levels[t],i,i>=deficits[t]) for t in range(10) for i in range(50)])
    result=analysis.main(['--lane',str(first),'--controls','reduced','--control-root',str(controls),'--out',str(tmp_path/'r1')])
    raises=study.read(tmp_path/'r1/raise.json')[model]['libero_spatial']['50']
    # Task 3 already runs pure inference: it stays there instead of being retested.
    assert raises=={name('libero_spatial',0):'pure',name('libero_spatial',1):'k40'}
    second=tmp_path/'second'
    lane(second,model,'A',[(t,raises[name('libero_spatial',t)],i,i>=2) for t in (0,1) for i in range(50)])
    journal=second/'driver/journal.jsonl'
    journal.write_text(''.join(json.dumps(dict(json.loads(row),ts=3))+'\n' for row in journal.read_text().splitlines()))
    result=analysis.main(['--lane',str(first),'--lane',str(second),'--controls','reduced','--control-root',str(controls),'--out',str(tmp_path/'r2')])
    assert study.read(tmp_path/'r2/raise.json')=={}
    tasks=result['cells'][0]['tasks']
    assert all(row['within_tolerance'] for task,row in tasks.items() if task!=name('libero_spatial',3))
    assert tasks[name('libero_spatial',3)]['level']=='pure' and not tasks[name('libero_spatial',3)]['within_tolerance']
    assert result['cells'][0]['n']==500
