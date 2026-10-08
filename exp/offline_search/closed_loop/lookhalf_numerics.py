"""B-only GR00T exact-pixel diagnostic. Extra work is excluded from serving IR."""
from __future__ import annotations
import numpy as np
from .look_recording import as_numpy
from .lookhalf_cameras import GrootOverride


def deltas(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return dict(max_abs=float(np.max(np.abs(a-b))), rms=float(np.sqrt(np.mean((a-b)**2))),
                relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(a), 1e-12)))


def record(session, adapter):
    import torch
    from openpi.cache.groot.staged import GrootStage1Output
    from openpi.cache.groot.key_builder import _contiguous_runs
    from exp.offline_search.closed_loop.plugin import ep_seed
    runner = adapter.policy._runner
    override = GrootOverride()
    norm = session._camera_observation
    # fork_rng restores production RNG, even on exceptions. Both chunks use
    # the same explicit noise; no production policy result is substituted.
    device = next(runner._model.parameters()).device
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices), runner.session():
        inputs, action = override.prepare(runner, norm)
        eagle, pixels = runner._eagle, inputs['pixel_values']
        mask = inputs['input_ids'] == eagle.image_token_index
        runs = _contiguous_runs(mask[0])
        if [n for _, n in runs] != [256, 256]:
            raise ValueError('GR00T numerical check expects exact two-camera layout')
        positions = torch.nonzero(mask.reshape(-1)).reshape(-1)
        text = eagle.language_model.get_input_embeddings()(inputs['input_ids'])
        joint_tokens = eagle.extract_feature(pixels)
        separate_tokens = torch.cat([eagle.extract_feature(pixels[i:i+1]) for i in range(2)], 0)
        def prefix(tokens):
            emb = text.clone()
            emb.reshape(-1, emb.shape[-1])[positions] = tokens.reshape(-1, tokens.shape[-1])
            return GrootStage1Output(emb, inputs['attention_mask'], mask, action)
        joint, separate = prefix(joint_tokens), prefix(separate_tokens)
        s2j = runner.run_stage2_llm(joint)
        seed = ep_seed(session.ep.seed, f'r12-numerical:{session._dec["step"]}')
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = runner.sample_noise(s2j, generator=generator)
        chunkj = runner.run_stage3(s2j, noise=noise.clone()).action_pred
        s2s = runner.run_stage2_llm(separate)
        chunks = runner.run_stage3(s2s, noise=noise.clone()).action_pred
        out = dict(groot_encoder_pixels=as_numpy(pixels), groot_input_ids=as_numpy(inputs['input_ids'], np.int64),
                   groot_attention_mask=as_numpy(inputs['attention_mask'], np.int64),
                   groot_policy_noise=as_numpy(noise), groot_joint_policy=as_numpy(chunkj),
                   groot_separate_policy=as_numpy(chunks))
        for i in range(2):
            out[f'groot_joint_key_v{i}'] = as_numpy(session.kb._reduce_vision(joint_tokens[i].float())).reshape(-1)
            out[f'groot_single_key_v{i}'] = as_numpy(session.kb._reduce_vision(separate_tokens[i].float())).reshape(-1)
        info = dict(pixel_dtype=str(pixels.dtype), camera_order=['video.image', 'video.wrist_image'],
                    noise_seed=int(seed), exact_input=True, diagnostic_only=True,
                    key_deltas=[deltas(out[f'groot_joint_key_v{i}'], out[f'groot_single_key_v{i}']) for i in range(2)],
                    policy_delta=deltas(out['groot_joint_policy'], out['groot_separate_policy']))
    return out, info
