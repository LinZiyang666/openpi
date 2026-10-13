"""CPU adapter tests; fake resident, no policy load or CUDA calls."""
from dataclasses import replace
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
from exp.r17_integration.common.contracts import CaptureSpec, RequestContext
from exp.r17_integration.common.serving import full_call
from exp.r17_integration.models.cosmos.metaworld import (
    CosmosMWAdapter, CosmosMWKeyBuilder, CosmosMWWirePolicy, inference_config,
    make_data_batch, make_manifest, native_observation,
)
from exp.r17_integration.models.cosmos.manifest import make_sampler


@pytest.fixture(autouse=True)
def isolated_provider_registry(monkeypatch):
    from exp.r17_integration.common import provider
    monkeypatch.setattr(provider, '_providers', {})


def observation():
    from exp.metaworld.tasks import PROMPTS
    return {'observation/image': np.arange(480 * 480 * 3, dtype=np.uint8).reshape(480, 480, 3),
            'observation/state': np.array([.1, .6, .2, .7], np.float32),
            'prompt': PROMPTS['assembly-v3']}


def request():
    return RequestContext('mw-cpu', FORMAL_SPEC.identity, 'assembly-v3', 0, 0,
                          arm='full', bundle_generation='cpu-v1', attempt=1, explicit_seed=7)


def utilities():
    def extract(latent, shape, indices):
        flat = latent[torch.arange(len(latent)), :, indices].flatten(1)
        count = flat.shape[1] // np.prod(shape)
        return flat[:, :count * np.prod(shape)].reshape(len(latent), count, *shape).mean(1)
    return SimpleNamespace(
        prepare_images_for_model=lambda images, cfg: np.stack([im[:224, :224] for im in images]),
        rescale_proprio=lambda values, stats, **kw: values.astype(np.float64),
        unnormalize_actions=lambda chunk, stats: chunk.astype(np.float64) * np.arange(1, 8) + np.arange(7),
        extract_action_chunk_from_latent_sequence=extract)


class FakeModel:
    tensor_kwargs = {'device': 'cpu'}
    config = SimpleNamespace(min_num_conditional_frames=3)
    sde = SimpleNamespace(sigma_min=4, sigma_max=80)

    def _normalize_video_databatch_inplace(self, batch):
        self.batch = batch

    def _augment_image_dim_inplace(self, batch):
        pass

    def encode(self, video):
        self.encodes = getattr(self, 'encodes', 0) + 1
        latent = torch.zeros(1, 16, 7, 28, 28)
        latent[:, :, 2] = float(video.float().mean())
        return latent

    def get_x0_fn_from_batch(self, batch, guidance, **kwargs):
        assert kwargs['skip_vae_encoding'] is True
        assert kwargs['previous_generated_latent'].shape == (1, 16, 7, 28, 28)
        return lambda x, sigma: x * .5

    def sampler(self, denoise, noise, num_steps, **kwargs):
        for _ in range(1 if num_steps == 2 else num_steps):
            noise = denoise(noise, torch.tensor(80.))
        # Distinct rotation values expose an incorrect projection/inverse order.
        output = torch.zeros_like(noise)
        output[:, :, 3] = .25
        return output


def adapter(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cosmos_policy.utils.utils', SimpleNamespace(set_seed_everywhere=lambda _: None))
    monkeypatch.setitem(sys.modules, 'cosmos_policy._src.imaginaire.utils', SimpleNamespace(
        misc=SimpleNamespace(arch_invariant_rand=lambda shape, dtype, device, seed: torch.ones(shape, dtype=dtype))))
    obs = observation()
    spec = SimpleNamespace(checkpoint='/unused')
    manifest = make_manifest({}, {'release': 'cpu-test'}, (FORMAL_SPEC.identity,))
    return CosmosMWAdapter(FakeModel(), manifest, inference_config(spec), {},
                           {obs['prompt']: torch.zeros(1, 512, 1024)}, utilities())


def test_wire_orientation_proprio_and_raw_state():
    obs = observation()
    native, prompt, state = native_observation(obs)
    assert native['primary_image'].tobytes() == obs['observation/image'].tobytes()
    np.testing.assert_allclose(native['proprio'], [.7, -.7, .1, .6, .2, 0, 0, 0, 1])
    state[0] = 99
    assert obs['observation/state'][0] != 99
    assert native['primary_image'].flags.c_contiguous


@pytest.mark.parametrize('key,value', [('observation/image', np.zeros((224,224,3), np.uint8)),
    ('observation/state', np.zeros(9)), ('observation/state', np.array([0, 1, 2, np.nan])), ('prompt', '')])
def test_invalid_wire(key, value):
    obs = observation(); obs[key] = value
    with pytest.raises(ValueError): native_observation(obs)


def test_batch_has_one_real_camera_and_three_conditioning_frames():
    native, _, _ = native_observation(observation())
    batch = make_data_batch(native, torch.zeros(1,512,1024), None, {}, FakeModel(), utilities(), torch)
    assert batch['video'].shape == (1,3,25,224,224)
    assert batch['num_conditional_frames'] == 3
    assert batch['proprio'].shape == (1,9) and batch['proprio'].dtype == torch.bfloat16
    assert batch['current_image_latent_idx'].item() == 2
    assert batch['action_latent_idx'].item() == 3
    assert all(v.item() == -1 for k,v in batch.items() if 'wrist' in k)
    video = batch['video']
    assert torch.equal(video[:,:,5:9], video[:,:,17:21])
    assert torch.count_nonzero(video[:,:,:5]) == 0


def test_one_camera_keys_ignore_future_and_proprio_padding():
    spec = make_manifest({}, {}, (FORMAL_SPEC.identity,)).spec.feature_spec
    builder = CosmosMWKeyBuilder(spec)
    latent = torch.zeros(1,16,7,28,28); latent[:,:,2] = 2
    state = np.arange(4,dtype=np.float32)
    fields = builder.fields_from_latent(latent,state)
    assert set(fields) == {'vision_0','robot_state'}
    assert fields['vision_0'].shape == (256,) and np.all(fields['vision_0'] == 2)
    latent[:,:,:2] = 100; latent[:,:,3:] = 100
    assert all(np.array_equal(v,builder.fields_from_latent(latent,state)[k]) for k,v in fields.items())


def test_inverse_then_project_without_scale_clip_or_gripper_flip():
    wire = CosmosMWWirePolicy({}, utilities())
    normalized = torch.zeros(16,7)
    normalized[:,0] = 3; normalized[:,6] = -2
    actions = wire.to_wire(normalized,None).actions
    assert actions.shape == (16,4) and actions.dtype == np.float64
    np.testing.assert_array_equal(actions[0], [3,1,2,-8])


@pytest.mark.parametrize('k,nfe', [(1,1),(2,1),(3,3),(5,5)])
def test_fake_full_call_and_capture_preserve_sampling(monkeypatch, k, nfe):
    a = adapter(monkeypatch)
    result, wire = full_call(a, observation(), request(), sampler=make_sampler(k))
    assert result.normalized_chunk.shape == (16,7)
    assert wire.actions.shape == (16,4) and result.actual_nfe == nfe
    assert a.model.encodes == 1
    if k == 5:
        captured, second = full_call(a, observation(), request(), capture=CaptureSpec(enabled=True))
        assert np.array_equal(second.actions, wire.actions)
        assert captured.capture['capture_enabled'] is True
        assert captured.rng_receipt['native_seed'] == 7
        assert a.make_trace_adapter().sample_fields(None,None,captured) == {}
    else:
        with pytest.raises(ValueError,match='default K5'):
            full_call(a, observation(), request(), sampler=make_sampler(k), capture=CaptureSpec(enabled=True))


def test_provider_clones_mw_and_uses_stock_protocol(monkeypatch):
    from exp.r17_integration.models.cosmos.provider import register_provider
    from exp.r17_integration.common.qualification import config_for_probe
    from exp.r17_integration.common.provider import validate_config
    a = adapter(monkeypatch)
    p = register_provider(a)
    clone = p.load_adapter()
    assert isinstance(clone,CosmosMWAdapter) and clone.model is a.model and clone.lock is a.lock
    config = config_for_probe(p,5,5)
    config.enabled = False
    config.checkpoints = {}
    validate_config(config)
    assert a.metadata().layout.active_actions == (0,1,2,6)
    assert a.metadata().layout.valid_state_indices == (0,1,2,3)
    assert a.metadata().feature_spec.real_cameras == ('corner2',)
    a.metadata().capabilities.require('debug_capture')
    with pytest.raises(ValueError): a.prepare(observation(),replace(request(),benchmark_protocol='libero'))
    obs = observation(); obs['prompt'] = 'unknown'
    with pytest.raises(ValueError,match='T5'): a.prepare(obs,request())
    config.recipe.enabled = True
    with pytest.raises(ValueError,match='unsupported'): p.config_validator(config)


def test_formal_selector_only_adds_mw_checkpoint():
    from exp.r17_integration.ops.formal_eval import DEFAULT_CHECKPOINTS, parse_args
    assert DEFAULT_CHECKPOINTS['cosmos_mw'] == '/data/mw_finetune/models/cosmos_mw'
    args = parse_args(['--model','cosmos_mw','--prepare-only','--server-node','local',
                       '--worker-node','timan107','--port','23240','--driver-port','23112'])
    assert args.model == 'cosmos_mw' and args.server_node == 'local'


def test_offline_resolver_keeps_policy_and_vae_checkpoints_distinct():
    from exp.r17_integration.models.cosmos.metaworld import offline_checkpoint_path
    policy = '/data/mw_finetune/models/cosmos_mw/consolidated.pt'
    vae = '/data/cosmos/hf_home/hub/snapshots/test/tokenizer/tokenizer.pth'
    def resolve(uri):
        return offline_checkpoint_path(uri, policy_path=policy, vae_path=vae)
    assert resolve(policy) == policy
    assert resolve(vae) == vae
    assert resolve('hf://nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth') == vae
    assert resolve('hf://nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt') == policy
    assert resolve('hf://nvidia/Cosmos-Policy-ALOHA-Predict2-2B/Cosmos-Policy-ALOHA-Predict2-2B.pt') == policy


def test_library_teacher_captures_one_wire_frame_and_native_chunk(monkeypatch, tmp_path):
    from dataclasses import asdict
    from exp.r17_integration.common.config import build_shared_storage
    from exp.r17_integration.common.qualification import config_for_probe
    from exp.r17_integration.common.serving import make_connection_factory
    from exp.r17_integration.debug.library_capture import LibraryCaptureFactory
    from exp.r17_integration.debug.reader_profiles import CaptureReader
    from exp.r17_integration.models.cosmos.provider import register_provider
    a = adapter(monkeypatch)
    provider = register_provider(a)
    cfg = config_for_probe(provider,5,5)
    cfg.enabled = False; cfg.checkpoints = {}; cfg.recipe.enabled = False
    cfg.recipe.ordered_tasks = list(FORMAL_SPEC.ordered_tasks)
    cfg.recipe.task_table_identity = FORMAL_SPEC.task_table_identity
    captures = LibraryCaptureFactory(tmp_path / 'capture')
    core = make_connection_factory(provider,cfg,build_shared_storage(cfg),
        capture_sink_factory=captures,capture_spec=CaptureSpec(enabled=True))()
    context = asdict(request())
    metadata = dict(context,task_uid='cpu:mw:0',orig_init_idx=0,split='library',init_pool='B',
        library_episode_id=0,suite_task_id=0,trial=0,task_data_hash='cpu-state',init_state_sha256='cpu-state',
        model_identity=a.metadata().checkpoint_identity,norm_identity=a.metadata().norm_identity,
        layout_identity=a.metadata().layout.identity)
    core.on_episode_start(task_id=0,episode_id=0,extra_metadata=metadata)
    obs = observation()
    core.capture_sink.bind_observation(obs)
    actual = core.infer(dict(obs,__request__=context))
    _, plain = full_call(a,obs,request())
    assert actual['actions'].shape == (5,4)
    assert actual['actions'].tobytes() == plain.actions[:5].tobytes()
    core.on_episode_end(__execution__=dict(decision_seq=0,executed_count=5,cursor=5,terminal=True),__success__=True)
    captures.close()
    root = next((tmp_path / 'capture').glob('binding_*'))
    row, arrays, keys = next(CaptureReader(root).decisions())
    assert set(arrays) == {'normalized_actions','state','image_0'}
    assert arrays['normalized_actions'].shape == (16,7)
    assert arrays['state'].shape == (4,)
    assert arrays['image_0'].tobytes() == obs['observation/image'].tobytes()
    assert set(keys) == {'vision_0','robot_state'}
    assert row['executed_count'] == 5 and row['actual_nfe'] == 5
