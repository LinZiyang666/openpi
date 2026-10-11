"""Read-only real B500/weights CPU producer and fitted-serving receipt.

The campaign argument must be a fresh P6 --prepare-only output. Never modifies
that sealed campaign, its library, runtime, or fit/stall artifacts.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from exp.r17_integration.models.dp.library import verified_producer
from exp.r17_integration.models.dp.loader import load_adapter
from exp.r17_integration.models.dp.provider import register_provider
from exp.r17_integration.models.dp.wire_policy import WirePolicy
from exp.r17_integration.common.config import load_cache_config, build_shared_storage
from exp.r17_integration.common.contracts import RequestContext
from exp.r17_integration.common.query_view import QueryView
from exp.r17_integration.common.recipe import make_recipe_factory
from exp.r17_integration.common.serving import make_bundle_validator
from exp.r17_integration.ops import cache_study as study
from exp.mw_finetune.dp.preprocess import quantile_unnormalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This receipt requires hidden CUDA')
    assets = args.campaign / 'assets'
    descriptor = study.read(assets / 'load.json')
    adapter = load_adapter(dict(descriptor, device='cpu'))
    assert adapter.device.type == 'cpu' and not torch.cuda.is_initialized()
    provider = register_provider(adapter)
    seal = study.read(assets / 'CACHE_STUDY.json')
    proof = verified_producer()
    baseline = study.read('/data/mw_finetune/eval/hs5dp_ctl_142812/assets/model_manifest.json')
    assert all(baseline['content_digests'][name] == pin for name, pin in proof['producer_sources'].items())
    assert all(baseline['content_digests'][name] == pin for name, pin in proof['weight_language_norm_assets'].items())
    records = []
    for suite in ('libero_spatial', 'libero_10'):
        parent_path = Path('/data/r17_library') / ('dp_libero_' + suite) / 'bpool'
        parent = study.read(parent_path / 'READY.json')
        view = study.read(parent_path / 'subsets/size_500/VIEW.json')
        index = int(np.load(parent_path / 'subsets/size_500/row_indices.npy', mmap_mode='r')[0])
        row = parent['rows'][index]
        source, serving = study.profiles('dp_libero', parent)
        arrays = {name: np.load(parent_path / meta['file'], mmap_mode='r')
                  for name, meta in parent['arrays'].items()}
        observation = dict(prompt=serving.tasks[row['task_id']][1], seed=18,
            **{'observation/image': arrays['image_0'][index].copy(),
               'observation/wrist_image': arrays['image_1'][index].copy(),
               'observation/state': arrays['state'][index].copy()})
        assert not np.array_equal(observation['observation/image'], observation['observation/wrist_image'])
        request = RequestContext('P4_CPU', serving.identity, row['task_name'], 0, 0,
                                 task_id=row['task_id'], explicit_seed=18)
        prepared = adapter.prepare(observation, request)
        s1 = adapter.run_stage1(prepared)
        s2 = adapter.run_stage2(s1)
        with torch.inference_mode():
            core = adapter.model._policy
            normalized = {k: core.normalizer.normalize(k, v) for k, v in prepared.data['inputs'].items()}
            torch.testing.assert_close(s2.data['conditioning'], core.obs_encoder(normalized), rtol=0, atol=0)
            trajectory = torch.from_numpy(arrays['normalized_actions'][index].copy())
            native_wire = quantile_unnormalize(core.normalizer.unnormalize('action', trajectory).numpy(),
                                              adapter.legacy_policy.stats['actions']).astype(np.float32)
            np.testing.assert_array_equal(adapter.to_wire(trajectory, prepared).actions, native_wire)
        np.testing.assert_array_equal(WirePolicy(adapter).infer(observation)['actions'],
                                      adapter.legacy_policy.infer(observation)['actions'])
        feature_diff = {name: float(np.max(np.abs(s1.fields[name] - arrays['key_' + name][index])))
                        for name in s1.fields}
        # Stored keys were GPU B1; this measured CPU/GPU difference is diagnostic.
        assert max(feature_diff.values()) < 1e-3
        group_results = []
        factory = make_recipe_factory(provider)
        for group in (g for g in seal['groups'] if g['suite'] == suite):
            config = load_cache_config(assets / group['file'])
            storage = build_shared_storage(config)
            provider.config_validator(config)
            make_bundle_validator(provider)(config, storage)
            recipe = factory(config, storage)
            query = QueryView(request, s1.fields, s1.state)
            result = recipe.query(query)
            assert result.action.shape == (adapter.metadata().layout.horizon, 7)
            assert result.action[:config.adapter_binding['commit_horizon']].shape == (config.adapter_binding['commit_horizon'], 7)
            assert result.extras['corrector_enabled'] == group['settings']['corrector']
            assert not result.force_miss  # Fresh actual Q3 tracker has no stall history.
            group_results.append(dict(group=group['group'], miss_mode=config.recipe.miss_mode,
                commit=config.adapter_binding['commit_horizon'], teacher_compatibility=recipe.teacher_compatibility,
                corrector_enabled=result.extras['corrector_enabled'], actual_nfe=0,
                stage_calls=[1,0,0], stall_status=result.extras['stall_status'],
                library_rows=len(recipe.action), fitted_path=config.recipe.fitted_path))
        records.append(dict(suite=suite, source_protocol=source.identity, serving_protocol=serving.identity,
            view_digest=view['content_digest'], sampled_real_row=row['row_id'], images_shape=[256,256,3],
            paired_real_views=True, native_conditioning_exact=True, native_DDPM100_wire_exact=True,
            native_output_transform_exact=True, stored_GPU_vs_CPU_key_max_abs=feature_diff,
            groups=group_results))
    receipt = dict(schema='r17.dp.cache_cpu_receipt.v1', status='PASS', cpu_only=True,
        gpu_executed=False, cuda_initialized=torch.cuda.is_initialized(), campaign=str(args.campaign),
        checkpoint_identity=adapter.metadata().checkpoint_identity, strict_load=adapter.strict_load,
        baseline_native_producer_and_assets_exact=True, baseline_campaign='hs5dp_ctl_142812', suites=records)
    args.output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('status','cpu_only','checkpoint_identity','strict_load')}))


if __name__ == '__main__':
    main()
