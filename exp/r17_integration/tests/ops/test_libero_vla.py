"""CPU-only formal/library/cache preparation for the two LIBERO VLA ports."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from exp.r17_integration.common.contracts import digest
from exp.r17_integration.common.provider import AdapterProvider, register_provider
from exp.r17_integration.common.recipe import bundle_identity, recipe_identity
from exp.r17_integration.ops import cache_study as study, formal_eval, library_campaign
from exp.r17_integration.ops.formal_driver import plan
from exp.r17_integration.models.pi05.adapter import model_spec, sampler_for_steps
from exp.r17_integration.models.groot.manifest import GrootManifest, digest as groot_digest
from exp.r17_integration.search.store import atomic_json

MODELS=('pi05_libero','groot_libero')


@pytest.fixture(autouse=True)
def private_provider_registry(monkeypatch):
    from exp.r17_integration.common import provider as registry
    for name in ('_providers','_provider_identities','_traces'):
        monkeypatch.setattr(registry,name,{})


def spec_for(model, benchmark):
    if model=='pi05_libero':
        return model_spec('CPU-weights','CPU-norm',benchmark='libero',benchmark_spec=benchmark)
    return GrootManifest(*(groot_digest(s) for s in ('weights','norm','code','transform','token')),
        benchmark='libero',benchmark_protocol=benchmark.identity).model_spec()


def assets(tmp_path, model, *, official=True):
    b=formal_eval.benchmark(official); spec=spec_for(model,b)
    provider=AdapterProvider('vla_'+model+'_'+tmp_path.name,spec,lambda:None,
                             sampler_for_steps if model=='pi05_libero' else lambda k:spec.default_sampler)
    register_provider(provider)
    config=formal_eval.full_config(provider,dict(domain='libero',model=model.removesuffix('_libero')))
    config.recipe.ordered_tasks=list(b.ordered_tasks);config.recipe.task_table_identity=b.task_table_identity
    tmp_path.mkdir(parents=True,exist_ok=True)
    atomic_json(tmp_path/'benchmark.json',b.manifest());atomic_json(tmp_path/'model_spec.json',asdict(spec))
    (tmp_path/'full.yaml').write_text(yaml.safe_dump(asdict(config)))
    return dict(domain='libero',model=model.removesuffix('_libero'),local_assets=str(tmp_path),assets=str(tmp_path),
        run_name='cpu',endpoint_host='unused',public_port=23160,workers=2,smoke=False,official_50=official),provider


@pytest.mark.parametrize('model',MODELS)
def test_official50_full_plan_and_suite_filter(tmp_path,model):
    c,p=assets(tmp_path,model)
    _,_,all_tasks,_=plan(c)
    assert len(all_tasks)==(2000 if model=='pi05_libero' else 1000)
    c['suites']=['libero_spatial','libero_10']
    _,_,tasks,_=plan(c)
    assert len(tasks)==len({t.task_uid for t in tasks})==1000
    assert {t.task_id for t in tasks}==set(range(10))|set(range(30,40))
    assert {t.orig_init_state_idx for t in tasks}==set(range(50))
    assert all(t.extra['num_steps']==(10 if model=='pi05_libero' else 4) for t in tasks)
    config=yaml.safe_load((tmp_path/'full.yaml').read_text())
    assert not config['enabled'] and not config['checkpoints'] and not config['recipe']['enabled']


@pytest.mark.parametrize('model',MODELS)
@pytest.mark.parametrize('suite',study.SUITES)
def test_full_b500_teacher_plan_and_cli(tmp_path,model,suite):
    c,p=assets(tmp_path,model,official=False)
    c.update(library=True,suite=suite,alignment=str(library_campaign.AUDIT_ROOT/(suite+'.json')))
    _,_,tasks,_=plan(c)
    assert len(tasks)==500 and {t.extra['library_episode_id'] for t in tasks}==set(range(500))
    assert all(t.extra['init_pool']=='B' and t.extra['split']=='library' for t in tasks)
    a=library_campaign.parse_args(['--model',model,'--suite',suite,'--replicas','5','--mps','--prepare-only',
        '--worker-egl-devices','0,1,2,3,4,5,6,7'])
    assert not a.capture_images and a.port==23160 and a.driver_port==23169
    assert a.worker_node=='timan107' and a.worker_max_episodes==4


@pytest.mark.parametrize('model',MODELS)
def test_all_sizes_eight_groups_c5_c10_and_native_miss(tmp_path,model):
    library_root=tmp_path/'library'
    sizes=(50,100,200,300,400,500)
    for suite in study.SUITES:
        library=library_root/(model+'_'+suite)/'bpool';library.mkdir(parents=True)
        b=formal_eval.benchmark(False);spec=spec_for(model,b)
        atomic_json(library/'READY.json',dict(binding={'model':asdict(spec),'sampler':asdict(spec.default_sampler),
            'benchmark_protocol':b.identity},content_digest='CPU-parent'))
        for n in sizes:
            view=library/'subsets'/('size_'+str(n));view.mkdir(parents=True)
            atomic_json(view/'VIEW.json',dict(content_digest='CPU-selected-'+str(n),size=n))
    output=tmp_path/'study'
    manifest=study.generate(output,models=(model,),library_root=library_root,sizes=sizes)
    rows=manifest['models'][model]
    assert len(rows)==96 and manifest['horizon_selection'][model.removesuffix('_libero')]==10
    assert len(study.validate_manifest(output/'STUDY.json')['models'][model])==96
    groups=[]
    # Build the same formal cache seal on CPU, retaining the generated recipes.
    serving=formal_eval.benchmark(True);spec=spec_for(model,serving)
    provider=AdapterProvider('teacher_template_'+model,spec,lambda:None,lambda k:spec.default_sampler)
    register_provider(provider)
    for row in rows:
        doc=yaml.safe_load((output/row['file']).read_text())
        assert doc['adapter_binding']['num_steps']==(10 if model=='pi05_libero' else 4)
        assert row['settings']['commit']==(10 if 'h' in row['group'] else 5)
        assert row['settings']['alias'] is None
        assert doc['recipe']['miss_mode']==('stall_only' if 'g' in row['group'] else 'never')
        assert doc['recipe']['corrector']==('r' in row['group']) and doc['recipe']['periodic_gap'] is None
        from exp.r17_integration.common.config import config_from_dict
        config=config_from_dict(doc)
        groups.append(dict(row,bundle_id=row['file'],bundle_identity=bundle_identity(config),recipe_identity=recipe_identity(config)))
    atomic_json(output/'model_spec.json',asdict(spec));atomic_json(output/'benchmark.json',serving.manifest())
    seal=dict(model=asdict(spec),benchmark_protocol=serving.identity,groups=groups)
    seal['digest']=digest(seal);atomic_json(output/'CACHE_STUDY.json',seal)
    campaign=dict(model=model.removesuffix('_libero'),domain='libero',local_assets=str(output),
                  endpoint_host='unused',public_port=23160,workers=2,run_name='cpu',smoke=False)
    # Plan one complete N cohort; all six cohorts' YAML semantics checked above.
    seal['groups']=[row for row in groups if row['library_size']==50]
    seal['digest']=digest({k:v for k,v in seal.items() if k!='digest'})
    atomic_json(output/'CACHE_STUDY.json',seal)
    _,_,tasks,_=study.plan(campaign)
    assert len(tasks)==len({t.task_uid for t in tasks})==8000
    assert {t.task_id for t in tasks}==set(range(10))|set(range(30,40))


@pytest.mark.parametrize('k',[1,2,3,4])
def test_groot_study_k_reaches_startup(tmp_path,monkeypatch,k):
    prepared=[]
    monkeypatch.setattr(formal_eval,'FORMAL_ROOT',tmp_path)
    monkeypatch.setattr(formal_eval,'prepare',lambda c,d:prepared.append(c) or {})
    (tmp_path/'arm.yaml').write_text(yaml.safe_dump({'adapter_binding':{'num_steps':k}}))
    (tmp_path/'pi.yaml').write_text(yaml.safe_dump({'adapter_binding':{'num_steps':10}}))
    path=tmp_path/'STUDY.json';atomic_json(path,{'models':{
        'pi05_libero':[{'file':'pi.yaml'}],'groot_libero':[{'file':'arm.yaml'}]}})
    assert formal_eval.main(['--model','groot_libero','--cache-study',str(path),'--official-50',
        '--run-name','cpu','--prepare-only'])==0
    assert prepared[0]['live_k']==k and prepared[0]['domain']=='libero'
    assert prepared[0]['suites']==list(study.SUITES)


def test_runtime_contains_libero_helpers_and_pi_norm(tmp_path):
    from exp.r17_integration.qualification.deploy_cpu import runtime_archive
    import tarfile
    root=Path(__file__).resolve().parents[4]
    archive=tmp_path/'runtime.tar';runtime_archive(root,archive)
    with tarfile.open(archive) as tar:
        names=set(tar.getnames())
    assert {'exp/libero_groot/policy_adapter.py','exp/libero_groot/libero_keys.py',
        'assets/pi05_libero/physical-intelligence/libero/norm_stats.json'}<=names


def test_h100_libero_vendor_pythonpath():
    c=dict(server_node='h100',model='groot',domain='libero')
    assert formal_eval.host_python(c)=='/home/exouser/gr00t_n15_venv/.venv/bin/python'
    assert '/home/exouser/gr00t_n15/examples/Libero' in formal_eval.server_env(c,'/runtime')['PYTHONPATH']


@pytest.mark.parametrize('model',MODELS)
def test_cpu_asset_seal_preserves_libero_protocol_and_load_descriptor(tmp_path,monkeypatch,model):
    from exp.r17_integration.qualification.assets import model_assets
    from exp.r17_integration.common import provider as registry
    monkeypatch.setattr(registry,'_providers',{})
    monkeypatch.setattr(registry,'_provider_identities',{})
    b=formal_eval.benchmark(True)
    atomic_json(tmp_path/'benchmark.json',b.manifest())
    checkpoint=tmp_path/'ckpt';checkpoint.mkdir()
    vendor=tmp_path/'vendor';eagle=vendor/'gr00t/model/backbone/eagle2_hg_model';eagle.mkdir(parents=True)
    (eagle/'tokenizer.json').write_text('{}')
    calls=[]
    if model=='pi05_libero':
        from exp.r17_integration.models.pi05 import adapter
        def manifest(path,**kw):
            calls.append(kw)
            return {'checkpoint_identity':'weights','norm_identity':'norm','code_digest':'code','tokenizer':{'sha256':'CPU-token'}}
        monkeypatch.setattr(adapter,'content_manifest',manifest)
    else:
        from exp.r17_integration.models.groot import manifest as gm
        def manifest(path,**kw):
            calls.append(kw)
            return GrootManifest(*(groot_digest(s) for s in ('w','n','c','t','token')),
                benchmark=kw['benchmark'],benchmark_protocol=kw['benchmark_protocol'],denoising_steps=kw['denoising_steps'])
        monkeypatch.setattr(gm,'checkpoint_manifest',manifest)
    spec,provider=model_assets(dict(model=model.removesuffix('_libero'),domain='libero',checkpoint=str(checkpoint),
        benchmark_spec=str(tmp_path/'benchmark.json'),vendor=str(vendor),live_k=4),tmp_path)
    desc=study.read(tmp_path/'load.json')
    assert spec.allowed_benchmark_protocols==(b.identity,)
    assert spec.feature_spec.fields=={'vision_0':32768,'vision_1':32768,'robot_state':8}
    assert desc['benchmark']=='libero' and all(call['benchmark']=='libero' for call in calls)
    assert provider.provider_id==('r17_pi05_libero_v1' if model=='pi05_libero' else 'r17_groot_libero_v1')
    if model=='groot_libero':
        assert set(desc['checkpoint'])==set(study.SUITES) and desc['benchmark_protocol']==b.identity
        assert desc['live_k']==4


@pytest.mark.parametrize('model',MODELS)
def test_image_free_teacher_parent_nested_fits_and_q3(tmp_path,model):
    from exp.r17_integration.debug.validate import model_binding
    from exp.r17_integration.debug.reader_profiles import CaptureReader
    from exp.r17_integration.search.teacher_library import build_teacher_parent, fit_nested_libraries
    from exp.r17_integration.tests.search.library_fixture import synthetic_capture
    b=formal_eval.benchmark()
    spec=spec_for(model,b)
    # Small image keys bound to an explicit CPU fixture; native actions/state/K retained.
    spec=replace(spec,feature_spec=replace(spec.feature_spec,fields={'vision_0':3,'vision_1':3,'robot_state':8}))
    binding=model_binding(spec,b.identity)
    library_root=tmp_path/'libraries'
    for suite in study.SUITES:
        source=tmp_path/('capture_'+suite)
        _,episodes,audit,names=synthetic_capture(source,suite=suite,binding_override=binding,capture_images=False)
        records=list(CaptureReader(source).decisions())
        assert all(set(values)=={'normalized_actions','state'} for _,values,_ in records)
        assert all(row['actual_nfe']==spec.default_sampler.num_steps for row,_,_ in records)
        dest=library_root/(model+'_'+suite)/'bpool'
        parent=build_teacher_parent([source],dest,binding,episodes,audit,capture_images=False)
        assert parent.manifest['metadata']['expected_episodes']==500
        products=fit_nested_libraries(parent,audit,task_names=names)
        assert set(products)=={'50','100','200','300','400','500'}
        previous=set()
        for n in (50,100,200,300,400,500):
            view=study.read(dest/'subsets'/('size_'+str(n))/'VIEW.json')
            ids=set(view['episode_ids']);assert len(ids)==n and previous<=ids;previous=ids
    output=tmp_path/'study'
    study.generate(output,library_root=library_root,models=(model,),sizes=(50,))
    study.fit_inputs(output/'STUDY.json',bundle=model)
    manifest=study.validate_manifest(output/'STUDY.json')
    assert not manifest['artifacts_pending']
    for row in manifest['models'][model]:
        fitted=study.read(Path(row['fit'])/'READY.json')
        assert fitted['receipt']['config']['commit_horizon']==row['settings']['commit']
        calibration=study.read(Path(row['stall'])/'READY.json')
        assert calibration['library_size']==50 and calibration['commit_horizon']==row['settings']['commit']
