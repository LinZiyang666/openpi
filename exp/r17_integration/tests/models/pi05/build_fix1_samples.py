"""Generate reviewable round-1 CPU fixtures; no real checkpoint or renderer."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import yaml


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError("requires CUDA_VISIBLE_DEVICES=''")
    from exp.r17_integration.tests.models.pi05.test_pi05 import fixture_adapter
    from exp.r17_integration.tests.models.pi05.test_pi05_e2e import tiny_store
    from exp.r17_integration.models.pi05.provider import register_provider, config_document
    from exp.r17_integration.common.config import load_cache_config
    from exp.r17_integration.common.recipe import bundle_identity, recipe_identity, fit_identity
    from exp.r17_integration.search.store import Store
    from exp.r17_integration.tests.benchmarks.test_libero import spec, task, args, native_kernel, Env, TailClient
    from exp.r17_integration.benchmarks.libero_bridge import R17LiberoEpisodeRunner
    from exp.r17_integration.conductor.manifest import episode_tasks
    from openpi.conductor.task import ServerEndpoint
    import numpy as np
    output = Path(__file__).parent/'samples/fix_round1'
    output.mkdir(parents=True, exist_ok=True)
    adapter, _ = fixture_adapter()
    provider = register_provider(adapter, provider_id='p2_round1_samples')
    if (output/'sample_manifest.json').exists():
        Store(output/'teacher')
        for path in output.glob('*.yaml'):
            load_cache_config(path)
        print('existing round1 fixture store/config hashes validated')
        return
    store, recipe = tiny_store(adapter, output/'teacher')
    identities = {}
    for name, k, settings in [('plain1', 1, None), ('plain2', 2, None),
                              ('recipe10', 10, {**recipe, 'guard': False, 'periodic_gap': None}),
                              ('ir25_k2', 2, {**recipe, 'periodic_gap': None, 'target_ir': .25})]:
        path = output/(name+'.yaml')
        path.write_text(yaml.safe_dump(config_document(provider, k=k, evidence_dir=output/'miss', recipe=settings)))
        config = load_cache_config(path)
        identities[name] = {'bundle': bundle_identity(config), 'recipe': recipe_identity(config),
                            'fit': fit_identity(config) if config.recipe.enabled else None}
    s = spec()
    (output/'libero_fixture_protocol.json').write_text(json.dumps(s.manifest(), indent=2)+'\n')
    episodes = episode_tasks(campaign='round1_fixture', arm='full', bundle_id='fixture', endpoint=ServerEndpoint('fixture', 1),
                            model_identity='fixture', norm_identity='fixture', layout_identity='fixture', spec=s)
    (output/'libero_fixture_400.json').write_text(json.dumps([asdict(t) for t in episodes], indent=2)+'\n')
    c, t = TailClient(32), task(s)
    runner = R17LiberoEpisodeRunner(args(s), lambda _: (Env(), np.array([7., 8.]), s.tasks[t.task_id][1], 220),
                                  spec=s, client_factory=lambda _: c, run_episode_fn=native_kernel())
    result = runner.run(t, lambda *a: None)
    runner.close()
    (output/'libero_commit32_rows.json').write_text(json.dumps(result.per_step_rows, indent=2)+'\n')
    manifest = {'schema': 'r17.p2.fix1.samples.v1', 'synthetic_fixture': True, 'checkpoint_loaded': False,
                'renderer_used': False, 'quality_sr_claim': False, 'pricing_claim': False,
                'libero_task_names': 'synthetic fixture names; official maps must be exported before deployment',
                'libero_kernel': 'unchanged examples/libero/main.py::_run_episode; CPU environment fixture',
                'libero_actual_controls': result.n_steps, 'libero_ack_counts': [r['executed_count'] for r in runner.receipts],
                'model': asdict(adapter.metadata()), 'teacher_store_digest': store.manifest['content_digest'],
                'config_identities': identities}
    (output/'sample_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'synthetic_fixture': True, 'controls': result.n_steps, 'sample_directory': str(output)}))


if __name__ == '__main__':
    main()
