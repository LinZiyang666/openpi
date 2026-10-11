"""Emit sealed CPU table/config/service-ledger fixtures, explicitly not pricing."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import yaml


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError("requires CUDA_VISIBLE_DEVICES=''")
    from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
    from exp.r17_integration.benchmarks.execution import cost_row
    from exp.r17_integration.common.config import load_cache_config, build_shared_storage
    from exp.r17_integration.common.recipe import bundle_identity
    from exp.r17_integration.models.pi05.provider import register_provider, config_document, make_connection_factory
    from exp.r17_integration.tests.models.pi05.test_pi05 import fixture_adapter, observation, request
    from exp.r17_integration.tests.models.pi05.test_pi05_e2e import tiny_store
    from exp.r17_integration.tests.benchmarks.test_libero import spec as libero_spec
    from exp.r17_integration.search.store import Store
    output = Path(__file__).parent/'samples/fix_round2'
    output.mkdir(parents=True, exist_ok=True)
    adapter, _ = fixture_adapter()
    provider = register_provider(adapter, provider_id='p2_round2_samples')
    if (output/'sample_manifest.json').exists():
        Store(output/'teacher')
        for path in output.glob('*.yaml'):
            load_cache_config(path)
        print('existing round2 sealed fixture store/configs validated')
        return
    store, recipe = tiny_store(adapter, output/'teacher')
    identities = {}
    for name, k, settings in [('plain1', 1, None), ('plain2', 2, None), ('plain10', 10, None),
                              ('recipe10', 10, {**recipe, 'guard': False, 'periodic_gap': None}),
                              ('gap3_k2', 2, {**recipe, 'periodic_gap': 3})]:
        path = output/(name+'.yaml')
        path.write_text(yaml.safe_dump(config_document(provider, k=k, evidence_dir=output/'miss', recipe=settings)))
        config = load_cache_config(path)
        identities[name] = bundle_identity(config)
    config = load_cache_config(output/'plain2.yaml')
    core = make_connection_factory(provider, config, build_shared_storage(config))(None, 'default')
    req = replace(request(), benchmark_protocol=FORMAL_SPEC.identity, task_id=43, explicit_seed=123)
    row = cost_row(core.infer({**observation(), '__request__': asdict(req)}),
                   {'campaign': 'cpu_round2_fixture', 'task_uid': 'cpu_latency',
                    'task_table_identity': FORMAL_SPEC.task_table_identity}, 0)
    row.update(synthetic_fixture=True, pricing_claim=False, checkpoint_loaded=False, renderer_used=False)
    (output/'service_ledger.json').write_text(json.dumps(row, indent=2)+'\n')
    (output/'mw_protocol.json').write_text(json.dumps(FORMAL_SPEC.manifest(), indent=2)+'\n')
    (output/'libero_fixture_protocol.json').write_text(json.dumps(libero_spec().manifest(), indent=2)+'\n')
    manifest = {'schema': 'r17.p2.fix2.samples.v1', 'synthetic_fixture': True, 'checkpoint_loaded': False,
                'renderer_used': False, 'quality_sr_claim': False, 'pricing_claim': False,
                'libero_task_names': 'synthetic fixture names, not official task-map attestation',
                'task_table_identity': FORMAL_SPEC.task_table_identity,
                'teacher_store_digest': store.manifest['content_digest'], 'config_identities': identities,
                'timing': 'actual CPU fixture host wall; zero adapter shared-lock wait; GPU time unavailable',
                'periodic': 'gap3 is an explicit schedule; no target owner IR claim'}
    (output/'sample_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'synthetic_fixture': True, 'sample_directory': str(output), 'configs': len(identities)}))


if __name__ == '__main__':
    main()
