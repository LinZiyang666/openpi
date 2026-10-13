"""Render four real MT50 observations, then check preprocessing with GPUs hidden.

Run --render-only in metaworld_sim; run the default in the Cosmos Python island.
Neither mode loads policy/VAE weights. Output stays beside the downloaded assets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def render(output):
    from exp.metaworld import env as kernel, tasks as tasks
    images, states, prompts, identities = [], [], [], []
    for name in ('assembly-v3', 'reach-v3'):
        for idx in (0, 1):
            env = kernel.make_env(name, idx, seed=7)
            try:
                state = kernel.reset_and_settle(env)
                wire = kernel.observation(env, state, tasks.PROMPTS[name])
                images.append(wire['observation/image'])
                states.append(wire['observation/state'])
                prompts.append(wire['prompt'])
                identities.append({'task': name, 'idx': idx, 'seed': 7})
            finally:
                env.close()
    np.savez_compressed(output / 'real_mt50_frames.npz', image=images, state=states, prompt=prompts)
    import importlib.metadata
    (output / 'real_mt50_frames.json').write_text(json.dumps({
        'episodes': identities, 'metaworld': importlib.metadata.version('metaworld'),
        'camera': 'corner2', 'camera_pos': list(tasks.CAMERA_POS), 'rotation': 180,
        'settle_steps': 15, 'policy_loaded': False}, indent=2) + '\n')


def check(output):
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Preprocessing check requires CUDA_VISIBLE_DEVICES empty')
    import pickle
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch
    from .loader import LoadSpec, activate_vendor
    from .metaworld import inference_config, native_observation, make_data_batch, CosmosMWKeyBuilder, make_manifest
    spec = LoadSpec(str(output))
    activate_vendor(spec)
    import cosmos_policy.experiments.robot.cosmos_utils as cu
    cfg = inference_config(spec)
    stats = cu.load_dataset_stats(cfg.dataset_stats_path)
    with open(cfg.t5_text_embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    class PreprocessingComplete(Exception):
        pass
    class FakeModel:
        tensor_kwargs = {'device': 'cpu'}
        config = SimpleNamespace(min_num_conditional_frames=3)
        def generate_samples_from_batch(self, batch, **kwargs):
            self.batch = batch
            # Upstream get_action later references unset wrist indices for
            # use_wrist_image=False. Stop at the preprocessed model input.
            raise PreprocessingComplete
    model = FakeModel()
    keys = CosmosMWKeyBuilder(make_manifest({}, {}, ('shape-check',)).spec.feature_spec)
    rows = []
    with np.load(output / 'real_mt50_frames.npz', allow_pickle=False) as frames:
        for i in range(len(frames['image'])):
            wire = {'observation/image': frames['image'][i], 'observation/state': frames['state'][i],
                    'prompt': str(frames['prompt'][i])}
            native, prompt, state = native_observation(wire)
            batch = make_data_batch(native, embeddings[prompt], cfg, stats, model, cu, torch)
            assert batch['video'].shape == (1,3,25,224,224)
            assert batch['proprio'].shape == (1,9) and torch.isfinite(batch['proprio']).all()
            # Upstream preprocessing uses .cuda(); redirect only its tensors to
            # CPU during this fake-model check. No CUDA initialization occurs.
            with patch.object(torch.Tensor, 'cuda', lambda self, *a, **kw: self):
                try:
                    cu.get_action(cfg, model, stats, dict(native, wrist_image=native['primary_image']),
                        embeddings[prompt].float().numpy(), seed=7, generate_future_state_and_value_in_parallel=False)
                except PreprocessingComplete:
                    pass
            assert batch.keys() == model.batch.keys()
            for name in batch:
                if isinstance(batch[name], torch.Tensor):
                    assert torch.equal(batch[name], model.batch[name]), name
                else:
                    assert batch[name] == model.batch[name], name
            fields = keys.fields_from_latent(torch.zeros(1,16,7,28,28), state)
            rows.append({'row': i, 'prompt': prompt, 'video': list(batch['video'].shape),
                         'proprio': list(batch['proprio'].shape),
                         'keys': {k: list(v.shape) for k,v in fields.items()},
                         'upstream_batch_equal': True})
    report = {'status': 'PASS', 'gpu_inference': False, 'cuda_initialized': torch.cuda.is_initialized(),
              'policy_loaded': False, 'vae_loaded': False, 'rows': rows}
    assert not report['cuda_initialized']
    (output / 'preprocessing_shape_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('/data/mw_finetune/models/cosmos_mw'))
    p.add_argument('--render-only', action='store_true')
    a = p.parse_args()
    render(a.checkpoint) if a.render_only else check(a.checkpoint)


if __name__ == '__main__':
    main()
