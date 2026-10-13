"""Community MetaWorld checkpoint on the stock R17 MT50 wire.

The release has no config or runner. Geometry follows the upstream one-camera
sequence; K5/chunk16/image augmentation are inherited defaults, not certified
training facts. See HANDBACK_MW_D.md before interpreting an MT50 score.
"""
from __future__ import annotations

import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np

from exp.r17_integration.common.contracts import (
    ActionStateLayout, Capabilities, Capability, FeatureSpec, ModelManifest,
    ModelSpec, PreparedObs, StateView, WireChunk, digest,
)
from .key_builder import CosmosKeyBuilder
from .manifest import make_sampler
from .staged import CosmosAdapter

RELEASE_FILES = ('.gitattributes', 'README.md', 'consolidated.pt',
                 'dataset_statistics.json', 't5_embeddings.pkl')
CHUNK_SIZE = 16


def is_metaworld(spec):
    from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
    return FORMAL_SPEC.identity in spec.benchmark_protocols


def native_observation(obs):
    image = np.asarray(obs['observation/image'])
    state = np.asarray(obs['observation/state'])
    if image.shape != (480, 480, 3) or image.dtype != np.uint8:
        raise ValueError('Cosmos MW requires the stock uint8 corner2 RGB480 frame')
    if state.shape != (4,) or state.dtype.kind != 'f' or not np.isfinite(state).all():
        raise ValueError('Cosmos MW requires finite state obs[:4] = xyz,gripper')
    prompt = obs.get('prompt')
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Cosmos MW requires the MT50 task prompt')
    # Released stats: opposed fingers, xyz, fixed identity xyzw quaternion.
    proprio = np.array([state[3], -state[3], *state[:3], 0, 0, 0, 1], dtype=np.float32)
    return {'primary_image': np.ascontiguousarray(image), 'proprio': proprio}, prompt, state.copy()


class CosmosMWWirePolicy:
    def __init__(self, stats, cu):
        self.stats, self.cu = stats, cu

    def prepare(self, obs, request):
        native, prompt, state = native_observation(obs)
        return PreparedObs({'native': native, 'prompt': prompt}, request, state.astype(np.float32))

    def extract_state_only(self, obs):
        return StateView(native_observation(obs)[2].astype(np.float32))

    def to_wire(self, normalized_chunk, prepared):
        chunk = (normalized_chunk.detach().float().cpu().numpy()
                 if hasattr(normalized_chunk, 'detach') else np.asarray(normalized_chunk))
        if chunk.shape != (CHUNK_SIZE, 7) or not np.isfinite(chunk).all():
            raise ValueError('Cosmos MW normalized chunk must be finite (16,7)')
        native = self.cu.unnormalize_actions(chunk[None], self.stats)[0]
        # Same MetaWorld command scale and gripper polarity; env clips to [-1,1].
        return WireChunk(np.ascontiguousarray(native[:, [0, 1, 2, 6]]))


class CosmosMWKeyBuilder(CosmosKeyBuilder):
    def fields_from_latent(self, latent, state):
        import torch
        if tuple(latent.shape) != (1, 16, 7, 28, 28):
            raise ValueError('Cosmos MW one-camera latent must be (1,16,7,28,28)')
        camera = torch.nn.functional.adaptive_avg_pool2d(latent[:, :, 2].float(), (4, 4))
        return {'vision_0': camera.flatten().detach().cpu().numpy().copy(),
                'robot_state': np.ascontiguousarray(state, dtype=np.float32)}


def make_data_batch(native, embedding, cfg, stats, model, cu, torch):
    image = cu.prepare_images_for_model([native['primary_image']], cfg)[0]
    blank = np.zeros_like(image)
    # One current observation, not a temporal history. Four raw frames/latent.
    latent_frames = [blank, blank, image, blank, blank, image, blank]
    frames = [blank[None]] + [np.repeat(frame[None], 4, axis=0) for frame in latent_frames[1:]]
    video = np.concatenate(frames).transpose(3, 0, 1, 2)[None]
    device = model.tensor_kwargs['device']
    proprio = cu.rescale_proprio(native['proprio'], stats, non_negative_only=False, scale_multiplier=1.0)
    batch = {'dataset_name': 'video_data', 'video': torch.from_numpy(video).to(device=device, dtype=torch.uint8),
             't5_text_embeddings': embedding.repeat(1, 1, 1).to(device=device, dtype=torch.bfloat16),
             'fps': torch.tensor([16], device=device, dtype=torch.bfloat16),
             'padding_mask': torch.zeros((1, 1, 224, 224), device=device, dtype=torch.bfloat16),
             'num_conditional_frames': 3,
             'proprio': torch.from_numpy(proprio).reshape(1, -1).to(device=device, dtype=torch.bfloat16)}
    indices = {'current_proprio': 1, 'current_image': 2, 'action': 3,
               'future_proprio': 4, 'future_image': 5, 'value': 6,
               'current_wrist_image': -1, 'current_wrist_image2': -1, 'current_image2': -1,
               'future_wrist_image': -1, 'future_wrist_image2': -1, 'future_image2': -1}
    batch.update({name + '_latent_idx': torch.tensor([value], device=device, dtype=torch.int64)
                  for name, value in indices.items()})
    return batch


class CosmosMWAdapter(CosmosAdapter):
    data_batch_builder = staticmethod(make_data_batch)
    action_latent_idx = 3
    teacher_seed = None  # Stock MT50 request RNG; capture never changes sampling.
    make_wire_policy = staticmethod(CosmosMWWirePolicy)
    make_key_builder = staticmethod(CosmosMWKeyBuilder)


def make_manifest(stats, digests, protocols):
    norm = digest({'stats': stats, 'proprio': '[g,-g,xyz,0,0,0,1]; native minmax',
                   'actions': 'native minmax inverse; select xyz,gripper; no scaling/clipping'})
    checkpoint = digest(digests)
    layout = ActionStateLayout(CHUNK_SIZE, 5, 4, tuple(range(4)), (0, 1, 2), 6,
        'MetaWorld: positive closes; unchanged native command', ('obs[:4]',),
        'MetaWorld_xyz_gripper', 'native MetaWorld command; environment clips [-1,1]',
        'MetaWorld world frame', 'cosmos.mw.minmax.xyz_gripper.v1',
        model_action_width=7, valid_action_indices=(0, 1, 2, 6))
    feature = FeatureSpec('Cosmos tokenizer.encode current camera latent frame', ('corner2',),
        {'vision_0': 256, 'robot_state': 4}, (7, 28, 28), 16,
        'per_camera_latent_channels.adaptive_avg_4x4.v1', 'reject', False, 'float32',
        tuple(range(4)), checkpoint, norm,
        digest({'wire': 'corner2 RGB480 rot180', 'jpeg': 95, 'resolution': 224,
                'crop_area': .9, 'state_t': 7, 'conditioning_frames': 3}))
    caps = {name: Capability('supported', 'R17 shared serving; CPU fake-model tests; GPU pending')
            for name in ('full', 'k_step', 'concurrent', 'bundle', 'episode', 'conductor', 'worker', 'analysis', 'debug_capture')}
    caps['cp1_retrieval'] = Capability('implemented_unverified', 'one-camera VAE keys; library quality not evaluated')
    for name in ('recipe', 'warmup_library', 'offline_replay', 'native_warm_resume', 'native_warm_reset',
                 'cp2', 'trace_twins', 'batch_stage1', 'batch_stage2', 'batch_stage3',
                 'split', 'shadow_routing', 'online_rit', 'flow_shoot'):
        caps[name] = Capability('unsupported', 'no MetaWorld implementation')
    return ModelManifest(ModelSpec('cosmos_mw', checkpoint, norm, layout, feature, make_sampler(),
        Capabilities(caps), tuple(protocols), digest({k: v for k, v in digests.items() if k.startswith('code:')}),
        'cosmos-policy/python3.10/torch2.7-cu128'), dict(digests))


def validate_config(config):
    if config.trace.enabled or config.warm_reset is not None or config.recipe.enabled:
        raise ValueError('Cosmos MW trace/warm reset/cache recipe is unsupported')


def cpu_manifest(spec):
    from .loader import asset_paths
    from .download import sha256_file
    from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
    from exp.r17_integration.common.benchmarks import register_benchmark
    register_benchmark(FORMAL_SPEC)
    paths = {k: v for k, v in asset_paths(spec).items() if not k.startswith('release:')}
    paths.update({'release:' + name: Path(spec.checkpoint) / name for name in RELEASE_FILES})
    digests = {k: sha256_file(v) for k, v in sorted(paths.items())}
    stats = json.loads((Path(spec.checkpoint) / 'dataset_statistics.json').read_text())
    return make_manifest(stats, digests, spec.benchmark_protocols or (FORMAL_SPEC.identity,))


def inference_config(spec):
    root = Path(spec.checkpoint)
    return SimpleNamespace(config='cosmos_predict2_2b_480p_libero__inference_only',
        config_file='cosmos_policy/config/config.py', ckpt_path=str(root / 'consolidated.pt'),
        dataset_stats_path=str(root / 'dataset_statistics.json'),
        t5_text_embeddings_path=str(root / 't5_embeddings.pkl'),
        suite='libero', use_wrist_image=False, num_wrist_images=0,
        use_third_person_image=True, num_third_person_images=1,
        use_proprio=True, normalize_proprio=True, unnormalize_actions=True,
        use_jpeg_compression=True, trained_with_image_aug=True, flip_images=False,
        use_variance_scale=False, chunk_size=CHUNK_SIZE, num_open_loop_steps=5,
        num_denoising_steps_action=5, seed=7, deterministic=True, randomize_seed=False)


def experiment_options(spec):
    from .loader import tokenizer_path
    return ['model.config.tokenizer.vae_pth=' + str(tokenizer_path(spec).absolute()),
            'model.config.state_t=7', 'model.config.min_num_conditional_frames=3',
            'model.config.max_num_conditional_frames=3', 'model.config.tokenizer.chunk_duration=25',
            'dataloader_train.dataset.use_wrist_images=False']


def offline_checkpoint_path(uri, *, policy_path, vae_path):
    # Only the eagerly registered policy defaults are replaced. Local paths,
    # particularly tokenizer.vae_pth, must retain their original meaning.
    paths = {
        'hf://nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt': policy_path,
        'hf://nvidia/Cosmos-Policy-ALOHA-Predict2-2B/Cosmos-Policy-ALOHA-Predict2-2B.pt': policy_path,
        'hf://nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth': vae_path,
    }
    return paths.get(str(uri), str(uri))


def load_adapter(spec, *, device='cuda:0'):
    from .loader import activate_vendor, tokenizer_path
    manifest = cpu_manifest(spec)
    activate_vendor(spec)
    import torch
    if torch.device(device).type != 'cuda':
        raise ValueError('Cosmos resident inference requires CUDA')
    torch.cuda.set_device(device)
    import cosmos_policy.experiments.robot.cosmos_utils as cu
    from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint
    from cosmos_policy.utils.utils import set_seed_everywhere
    from unittest.mock import patch
    cfg = inference_config(spec)
    set_seed_everywhere(cfg.seed)
    # Experiment registration resolves unrelated base/ALOHA HF checkpoints
    # eagerly. They are immediately overridden; keep this local load offline.
    vae_path = str(tokenizer_path(spec).absolute())
    resolve = lambda uri: offline_checkpoint_path(uri, policy_path=cfg.ckpt_path, vae_path=vae_path)
    with patch('cosmos_policy._src.imaginaire.utils.checkpoint_db.get_checkpoint_path', side_effect=resolve):
        model, _ = load_model_from_checkpoint(experiment_name=cfg.config, s3_checkpoint_dir=cfg.ckpt_path,
            config_file=cfg.config_file, load_ema_to_reg=False, to_device=device,
            experiment_opts=experiment_options(spec))
    model.eval().to(device)
    stats = cu.load_dataset_stats(cfg.dataset_stats_path)
    with Path(cfg.t5_text_embeddings_path).open('rb') as f:
        embeddings = pickle.load(f)
    return CosmosMWAdapter(model, manifest, cfg, stats, {k: v.cpu() for k, v in embeddings.items()}, cu)


def run_parity(adapter, *, steps=(5, 3, 1, 2)):
    """Coordinator-only native generator vs stages, with the reconstructed batch.

    Upstream get_action cannot return for a disabled wrist (unset wrist indices).
    Compare against its underlying generate_samples_from_batch instead. Real
    CPU frames separately check batch equality at get_action's generation boundary.
    This is not a comparison against the unavailable author's runner.
    """
    import torch
    from cosmos_policy.utils.utils import set_seed_everywhere
    from exp.metaworld.tasks import MT50_TASKS
    from exp.r17_integration.common.contracts import RequestContext
    from exp.r17_integration.common.serving import full_call
    image = np.arange(480 * 480 * 3, dtype=np.uint8).reshape(480, 480, 3)
    wire = {'observation/image': image, 'observation/state': np.array([0, .6, .2, .6], np.float32),
            'prompt': MT50_TASKS[0][1]}
    native, prompt, _ = native_observation(wire)
    request = RequestContext('cosmos-mw-parity', adapter.metadata().allowed_benchmark_protocols[0],
                             MT50_TASKS[0][0], 0, 0, explicit_seed=7)
    checks = []
    for k in steps:
        set_seed_everywhere(0)
        batch = make_data_batch(native, adapter.embeddings[prompt], adapter.cfg, adapter.stats,
                                adapter.model, adapter.cu, torch)
        with torch.inference_mode():
            generated = adapter.model.generate_samples_from_batch(batch, n_sample=1, num_steps=k,
                seed=7, is_negative_prompt=False, use_variance_scale=False)
            normalized = adapter.cu.extract_action_chunk_from_latent_sequence(generated, (CHUNK_SIZE,7),
                torch.tensor([3], device=generated.device, dtype=torch.int64)).float().cpu().numpy()
            reference = adapter.cu.unnormalize_actions(normalized, adapter.stats)[0][:, [0,1,2,6]]
        result, chunk = full_call(adapter, wire, request, sampler=make_sampler(k))
        if reference.dtype != chunk.actions.dtype or reference.tobytes() != chunk.actions.tobytes():
            raise AssertionError(f'Cosmos MW reconstructed native/staged parity failed at K{k}')
        checks.append({'k': k, 'actual_nfe': result.actual_nfe, 'bytes_exact': True,
                       'shape': list(chunk.actions.shape), 'dtype': str(chunk.actions.dtype)})
    return {'passed': True, 'status': 'PASS', 'checks': checks, 'tolerance': 0,
            'upstream': 'model.generate_samples_from_batch; reconstructed single-camera batch',
            'author_runner_parity': 'unavailable', 'model_identity': adapter.metadata().identity}
