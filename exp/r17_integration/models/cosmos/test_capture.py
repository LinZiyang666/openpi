"""CPU native-kernel substitutes exercise the real Cosmos/P6 serving seams."""
import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exp.r17_integration.common.contracts import CaptureSpec, Stage1, Stage2
from exp.r17_integration.models.cosmos.manifest import make_manifest, make_sampler
from exp.r17_integration.models.cosmos.protocols import PROPRIO_KEY, benchmark_profile
from exp.r17_integration.models.cosmos.staged import CosmosAdapter


@pytest.fixture
def resident(monkeypatch):
    from exp.r17_integration.common.benchmarks import register_benchmark
    from exp.r17_integration.common import provider
    from exp.r17_integration.models.cosmos import staged
    from exp.r17_integration.ops.formal_eval import benchmark
    profile = benchmark_profile(benchmark())
    register_benchmark(profile)
    # Each resident test registers the fixed Cosmos provider ID independently.
    monkeypatch.setattr(provider, '_providers', {k:v for k,v in provider._providers.items() if k != 'cosmos_libero'})
    manifest = make_manifest({}, {'release': 'cpu-capture-fixture'}, (profile.identity,))
    counts = dict(encode=0, condition=0, denoise=0)

    def seed_all(seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    monkeypatch.setitem(sys.modules, 'cosmos_policy.utils.utils', SimpleNamespace(set_seed_everywhere=seed_all))
    def noise(shape, dtype, device, seed):
        return torch.randn(shape, dtype=dtype, device=device, generator=torch.Generator().manual_seed(seed))
    monkeypatch.setitem(sys.modules, 'cosmos_policy._src.imaginaire.utils',
                        SimpleNamespace(misc=SimpleNamespace(arch_invariant_rand=noise)))
    monkeypatch.setattr(staged, 'make_data_batch', lambda native,*a:
                        {'video': torch.tensor([int(native['primary_image'][0,0,0])], dtype=torch.float32)})

    class Model:
        tensor_kwargs = {'device': 'cpu'}
        sde = SimpleNamespace(sigma_max=80, sigma_min=4)
        def _normalize_video_databatch_inplace(self, batch): pass
        def _augment_image_dim_inplace(self, batch): pass
        def encode(self, video):
            counts['encode'] += 1
            return torch.ones(1,16,9,28,28)*video[0]
        def get_x0_fn_from_batch(self, batch, *a, **kw):
            counts['condition'] += 1
            assert kw['skip_vae_encoding']
            def denoise(x, sigma):
                counts['denoise'] += 1
                return x*.5
            return denoise
        def sampler(self, denoise, noise, *, num_steps, **kwargs):
            for i in range(num_steps): noise = denoise(noise, i)
            return noise
    cu = SimpleNamespace(
        extract_action_chunk_from_latent_sequence=lambda generated,shape,indices:
            generated.reshape(1,-1)[:,:112].reshape(1,16,7),
        unnormalize_actions=lambda actions,stats:actions.astype(np.float64)*.123456789123+1)
    adapter = CosmosAdapter(Model(), manifest, None, {}, {'fixture prompt': torch.ones(1)}, cu)
    return adapter, profile, counts


def inputs(profile):
    image=np.arange(256*256*3,dtype=np.uint32).reshape(256,256,3).astype(np.uint8)
    return {'observation/image':image, 'observation/wrist_image':255-image,
            PROPRIO_KEY:np.arange(9,dtype=np.float64)*.01, 'prompt':'fixture prompt'}, dict(
            campaign='cosmos-cpu-teacher', benchmark_protocol=profile.identity,
            task_name=profile.ordered_tasks[0], original_init_idx=0, decision_seq=0,
            task_id=0, seed=195, explicit_seed=195, arm='full', attempt=1,
            bundle_generation='formal_full_v1', task_table_identity=profile.task_table_identity)


def rng_bytes():
    return random.getstate(), np.random.get_state()[1].tobytes(), torch.get_rng_state().numpy().tobytes()


def test_k2_real_serving_receipts_and_aggregate_expect_one_denoiser(resident, monkeypatch, tmp_path):
    from dataclasses import asdict
    import yaml
    from exp.r17_integration.common.config import build_shared_storage
    from exp.r17_integration.common.serving import make_connection_factory
    from exp.r17_integration.models.cosmos.provider import register_provider
    from exp.r17_integration.ops import formal_eval as launch
    from exp.r17_integration.ops.formal_driver import plan
    adapter, profile, counts = resident
    def upstream_k2(denoise, noise, *, num_steps, **kwargs):
        assert num_steps == 2
        return denoise(noise, torch.tensor([80.0]))
    monkeypatch.setattr(adapter.model, 'sampler', upstream_k2)
    provider = register_provider(adapter)
    campaign = dict(model='cosmos', domain='libero', denoising_steps=2, smoke=False,
        local_assets=str(tmp_path), run_name='cpu_k2', endpoint_host='unused', public_port=23240, workers=1)
    cfg = launch.full_config(provider, campaign)
    cfg.recipe.ordered_tasks=list(profile.ordered_tasks)
    cfg.recipe.task_table_identity=profile.task_table_identity
    sampler = make_sampler(2)
    def write(name, value):
        (tmp_path/name).write_text(json.dumps(value)+'\n')
    (tmp_path/'full.yaml').write_text(yaml.safe_dump(asdict(cfg)))
    write('benchmark.json', profile.manifest())
    write('model_spec.json', asdict(adapter.metadata()))
    write('execution_sampler.json', asdict(sampler))
    _, _, tasks, endpoints = plan(campaign)
    assert endpoints[0].allowed_k == (2,)
    assert {t.extra['num_steps'] for t in tasks} == {2}
    policy=make_connection_factory(provider,cfg,build_shared_storage(cfg))()
    task=tasks[0]
    policy.on_episode_start(extra_metadata=dict(task.extra,original_init_idx=0,task_id=0,
        attempt=1,seed=195,explicit_seed=195,ack_protocol='r17.execution.v1'))
    obs,_=inputs(profile)
    result=policy.infer(obs)
    row=dict(result['__hit_meta__'],task_uid=task.task_uid,accepted=True,attempt=1)
    assert result['actions'].shape == (16,7)
    assert row['actual_nfe']==1 and tuple(row['stage_calls'])==(1,1,1)
    assert row['sampler_identity']==sampler.identity and not row['capture_enabled']
    assert counts==dict(encode=1,condition=1,denoise=1)
    (tmp_path/'driver').mkdir()
    write('campaign.json',campaign)
    write('EPISODE_PLAN.json',dict(tasks=[asdict(task)],model=asdict(adapter.metadata()),benchmark_protocol=profile.identity))
    write('driver/DRIVER_RECEIPT.json',dict(status='PASS',elapsed_seconds=1))
    write('driver/journal.jsonl',dict(task_uid=task.task_uid,accepted=True,status='done',attempt=1,success=True))
    write('driver/per_step.jsonl',row)
    report=launch.aggregate(tmp_path)
    assert report['status']=='PASS' and report['arm_label']=='K2（含清理步）' and report['expected_nfe']==1
    for field,value in (('actual_nfe',2),('stage_calls',[1,1,2]),('sampler_identity',make_sampler(1).identity)):
        write('driver/per_step.jsonl',dict(row,**{field:value}))
        assert launch.aggregate(tmp_path)['status']=='PARTIAL'
    write('driver/per_step.jsonl',row)
    bad=asdict(sampler);bad['config']['expected_nfe']=2
    write('execution_sampler.json',bad)
    assert launch.aggregate(tmp_path)['status']=='PARTIAL'


@pytest.mark.parametrize('disabled_capture',(CaptureSpec(enabled=False),CaptureSpec(False,True,('latent',))))
def test_capture_off_default_is_byte_identical_and_never_calls_observer(resident, monkeypatch,disabled_capture):
    from exp.r17_integration.common.contracts import RequestContext
    from exp.r17_integration.common.serving import full_call
    adapter, profile, counts = resident
    obs, context = inputs(profile)
    monkeypatch.setattr(adapter, 'make_trace_adapter', lambda: pytest.fail('capture-off observer invoked'))
    request = RequestContext(**context)
    baseline, wire_a = full_call(adapter, obs, request)
    state_a = rng_bytes()
    explicit, wire_b = full_call(adapter, obs, request, capture=disabled_capture)
    assert wire_a.actions.dtype == np.float64
    assert wire_a.actions.tobytes() == wire_b.actions.tobytes()
    assert baseline.normalized_chunk.numpy().tobytes() == explicit.normalized_chunk.numpy().tobytes()
    assert baseline.capture == explicit.capture == {'capture_enabled':False,'pricing_eligible':True}
    assert not baseline.snapshots and rng_bytes() == state_a
    assert counts == dict(encode=2, condition=2, denoise=10)


def test_p6_library_capture_matches_plain_bytes_rng_keys_state_images_and_allowlist(resident, tmp_path):
    from exp.r17_integration.common.config import build_shared_storage
    from exp.r17_integration.common.qualification import config_for_probe
    from exp.r17_integration.common.serving import make_connection_factory
    from exp.r17_integration.debug.library_capture import LibraryCaptureFactory
    from exp.r17_integration.debug.reader_profiles import CaptureReader
    from exp.r17_integration.models.cosmos.provider import register_provider
    adapter, profile, counts = resident
    provider = register_provider(adapter)
    cfg = config_for_probe(provider,5,16)
    cfg.enabled=False; cfg.checkpoints={}; cfg.recipe.enabled=False
    cfg.recipe.ordered_tasks=list(profile.ordered_tasks)
    cfg.recipe.task_table_identity=profile.task_table_identity
    obs, context = inputs(profile)
    capture_factory = LibraryCaptureFactory(tmp_path/'capture')
    plain = make_connection_factory(provider,cfg,build_shared_storage(cfg))()
    captured = make_connection_factory(provider,cfg,build_shared_storage(cfg),
        capture_sink_factory=capture_factory,capture_spec=CaptureSpec(enabled=True,snapshots=False))()
    metadata=dict(context,task_uid='cpu:cosmos:0',orig_init_idx=0,split='library',init_pool='B',
        library_episode_id=0,suite_task_id=0,trial=0,task_data_hash='cpu-B-state',init_state_sha256='cpu-B-state',
        model_identity=adapter.metadata().checkpoint_identity,norm_identity=adapter.metadata().norm_identity,
        layout_identity=adapter.metadata().layout.identity)
    for core in (plain,captured): core.on_episode_start(task_id=0,episode_id=0,extra_metadata=metadata)
    expected=plain.infer(dict(obs,__request__=context))
    expected_rng=rng_bytes()
    captured.capture_sink.bind_observation(obs)
    actual=captured.infer(dict(obs,__request__=context))
    assert actual['actions'].tobytes()==expected['actions'].tobytes()
    assert rng_bytes()==expected_rng and counts==dict(encode=2,condition=2,denoise=10)
    assert actual['__hit_meta__']['capture_enabled'] is True
    assert actual['__hit_meta__']['pricing_eligible'] is False
    for core in (plain,captured):
        core.on_episode_end(__execution__=dict(decision_seq=0,executed_count=16,cursor=16,terminal=True),__success__=True)
    capture_factory.close()
    root=next((tmp_path/'capture').glob('binding_*'))
    row, arrays, keys = next(CaptureReader(root).decisions())
    assert set(arrays)=={'normalized_actions','state','image_0','image_1'}
    assert set(keys)==set(adapter.metadata().feature_spec.fields)=={'vision_0','vision_1','robot_state'}
    assert arrays['normalized_actions'].shape==(16,7) and arrays['normalized_actions'].dtype==np.float32
    assert arrays['state'].tobytes()==obs[PROPRIO_KEY].astype(np.float32).tobytes()
    for i, wire_key in enumerate(('observation/image','observation/wrist_image')):
        assert arrays['image_'+str(i)].dtype==np.uint8
        assert arrays['image_'+str(i)].tobytes()==obs[wire_key].tobytes()
    assert row['executed_count']==16 and row['terminal'] and row['actual_nfe']==5
    assert row['stage_calls']==[1,1,5] and row['episode_id']==row['library_episode_id']==0
    assert row['model_identity']==adapter.metadata().checkpoint_identity
    assert row['sampler_identity']==adapter.metadata().default_sampler.identity
    assert not set(row)&{'native_snapshots','native_capture_metadata','rng_receipt','diagnostics'}
    assert json.loads((root/'r17_capture.json').read_text())['status']=='closed'


@pytest.mark.parametrize('capture,k,seed',[(CaptureSpec(True,True),5,195),
    (CaptureSpec(True,False,('latent',)),5,195),(CaptureSpec(True,False),3,195),
    (CaptureSpec(True,False),2,195),(CaptureSpec(True,False),5,7)])
def test_teacher_profile_rejects_snapshots_extra_products_wrong_sampler_and_seed(resident,capture,k,seed):
    adapter,_,counts=resident
    stage=Stage2({'model_identity':adapter.metadata().identity},Stage1({}, {}, np.zeros(9)))
    with pytest.raises(ValueError,match='teacher'):
        adapter.sample(stage,make_sampler(k),SimpleNamespace(seed_for=lambda:seed),capture)
    assert counts==dict(encode=0,condition=0,denoise=0)
    adapter.metadata().capabilities.require('debug_capture')
    for capability in ('trace_twins','native_warm_resume','cp2'):
        with pytest.raises(ValueError): adapter.metadata().capabilities.require(capability)


def test_p6_teacher_b_state_guard_uses_the_worker_hash_namespace():
    from exp.r17_integration.benchmarks.spec import digest
    from exp.r17_integration.conductor.manifest import episode_tasks
    from exp.r17_integration.ops.formal_eval import benchmark
    from exp.r17_integration.ops.library_campaign import AUDIT_ROOT, teacher_tasks
    from openpi.conductor.task import ServerEndpoint
    profile=benchmark_profile(benchmark())
    tasks=episode_tasks(campaign='cpu-B-provenance',arm='full',bundle_id='default',
        endpoint=ServerEndpoint('unused',23360),model_identity='cpu',norm_identity='cpu',layout_identity='cpu',
        num_steps=5,spec=profile,partition='smoke',names=profile.ordered_tasks[:2],indices=[0])
    path=AUDIT_ROOT/'libero_spatial.json'
    rows=json.loads(path.read_text())['rows']
    actual=teacher_tasks(dict(alignment=str(path),suite='libero_spatial',smoke=True),profile,tasks)
    for task in actual:
        witness=next(r for r in rows if r['task_id']==task.task_id and r['orig_init_state_idx']==0)
        assert task.extra['task_data_hash']==digest(dict(witness['init_array'],task_name=task.extra['task_name']))
        assert task.extra['task_data_hash'].startswith('sha256:')
        assert task.extra['init_state_sha256']==witness['init_array']['state_sha256']
