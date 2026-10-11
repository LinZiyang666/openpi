"""Native Cosmos cache-study admission and real B500 CPU serving regressions."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from exp.r17_integration.models.cosmos import library_admission as admission
from exp.r17_integration.models.cosmos.cache_probe import run_probe
from exp.r17_integration.models.cosmos.provider import configure_library_provider, validate_config
from exp.r17_integration.common.contracts import digest
from exp.r17_integration.common.config import load_cache_config
from exp.r17_integration.ops.cache_study import model_spec, GROUPS


@pytest.fixture(scope='module')
def release(tmp_path_factory):
    from exp.r17_integration.common import provider as registry
    from openpi.serving import websocket_policy_server as bundles
    root = tmp_path_factory.mktemp('cosmos_cache')/'preflight'
    with patch.object(registry, '_providers', {}), patch.object(registry, '_provider_identities', {}), \
            patch.object(bundles, '_bundles', {}), patch.object(bundles, '_current_bundle', None):
        report = run_probe(root)
        provider = registry.get_provider('cosmos_libero')
        config = load_cache_config(root/'libero_spatial_c.yaml')
        manifest = admission._manifests[provider.model_spec.identity]
        yield report, provider, config, manifest


@pytest.mark.parametrize('group', GROUPS)
def test_all_yaml_groups_key_only_hits_or_stall_only_native_k5(release, group):
    report, _, _, _ = release
    rows = [r for r in report['groups'] if r['group'] == group]
    assert len(rows) == 2 and report['resident_generations'] == 1
    for row in rows:
        gated = 'g' in group
        assert row['C'] == 16 and row['stage_totals'] == ([5,1,5] if gated else [5,0,0])
        assert row['fresh_wrappers_shared_resident'] and row['action_transform_byte_exact']
        assert sum(t['source'] == 'policy' for t in row['traces']) == int(gated)
        for trace in row['traces']:
            miss = trace['source'] == 'policy'
            assert trace['actual_nfe'] == (5 if miss else 0)
            assert trace['hit_type'] == ('MISS' if miss else 'FULL_HIT')
            assert tuple(trace['stage_calls']) == ((1,1,5) if miss else (1,0,0))
            assert trace['corrector_enabled'] == ('r' in group)
            if miss:
                assert trace['stall_call'] and trace['stall_status']['state'] == 'slow_confirmed'
        if gated:
            assert row['traces'][1]['stall_extra_look']
            assert row['traces'][3]['stall_cooldown'] and row['traces'][3]['source'] == 'library'


def test_cosmos_long_labels_are_same_c16_bundle_not_extra_horizon(release):
    report, _, _, _ = release
    rows = {(r['suite'],r['group']):r for r in report['groups']}
    for suite in ('libero_spatial','libero_10'):
        for alias, base in report['alias_pairs'].items():
            assert rows[suite,alias]['bundle_identity'] == rows[suite,base]['bundle_identity']


def test_provider_replacement_preserves_frozen_registry_identity_and_clone(release):
    _, provider, _, _ = release
    plain = replace(provider, config_validator=None, library_validator=None)
    before = dict(vars(plain))
    configured = configure_library_provider(plain)
    assert plain is not configured and vars(plain) == before
    assert configured.provider_id == plain.provider_id and configured.model_spec is plain.model_spec
    assert configured.load_adapter is plain.load_adapter
    a, b = configured.load_adapter(), configured.load_adapter()
    assert a is not b and a.model is b.model and a.lock is b.lock
    assert a.keys is not b.keys and a.wire is not b.wire


@pytest.mark.parametrize('field,value', [('miss_mode','legacy'),('guard',True),('periodic_gap',0),
    ('target_ir',0.5),('k',8),('kref',4),('early',False),('fitted_digest',None),('stall_path','ordinary')])
def test_unregistered_cache_assumptions_fail_before_model_load(release,field,value):
    _, _, config, _ = release
    bad = copy.deepcopy(config)
    setattr(bad.recipe,field,value)
    if field == 'stall_path': bad.recipe.stall_digest = 'digest'
    with pytest.raises(ValueError,match='sealed native K5/C16'):
        validate_config(bad)


@pytest.mark.parametrize('field,value',[('num_steps',2),('commit_horizon',32),('allow_partial_commit',True)])
def test_cache_cannot_change_teacher_native_cadence_or_sampler(release,field,value):
    _, _, config, _ = release
    bad = copy.deepcopy(config);bad.adapter_binding[field] = value
    with pytest.raises(ValueError,match='sealed native K5/C16'):validate_config(bad)


def test_default_off_config_remains_byte_identical(release):
    _, provider, _, _ = release
    from exp.r17_integration.common.qualification import config_for_probe
    config = config_for_probe(provider,5,16)
    before = json.dumps(asdict(config),sort_keys=True)
    validate_config(config)
    assert json.dumps(asdict(config),sort_keys=True) == before


def test_content_compatibility_records_both_exact_retrieval_identities(release):
    _, provider, config, _ = release
    from exp.r17_integration.models.cosmos.cache_teacher import DATA
    from exp.r17_integration.common.recipe import retrieval_model_identity
    binding = dict(model=DATA['teacher_manifest']['spec'],sampler=DATA['teacher_manifest']['spec']['default_sampler'],
                   benchmark_protocol=DATA['teacher_manifest']['spec']['allowed_benchmark_protocols'][0])
    receipt = provider.library_validator(config,binding)
    assert receipt['schema'] == 'r17.teacher_compatibility.v1' and receipt['status'] == 'PASS'
    assert receipt['teacher_identity'] == retrieval_model_identity(binding['model'])
    assert receipt['resident_identity'] == retrieval_model_identity(provider.model_spec)
    assert receipt['teacher_identity'] != receipt['resident_identity']
    assert receipt['evidence']['output_transform'] == 'cosmos.libero.official_minmax.v1'
    assert len(receipt['evidence']['source']) == 2


@pytest.mark.parametrize('pin', ['release:Cosmos-Policy-LIBERO-Predict2-2B.pt',
    'tokenizer:weights','release:libero_dataset_statistics.json','release:libero_t5_embeddings.pkl',
    'code:r17:key_builder.py','code:r17:wire_policy.py','code:r17:loader.py',
    'code:official:cosmos_policy/experiments/robot/cosmos_utils.py'])
def test_changed_weight_norm_prompt_native_producer_cannot_use_teacher(release,pin):
    _, _, config, manifest = release
    from exp.r17_integration.models.cosmos.cache_teacher import DATA
    value = copy.deepcopy(manifest);value['content_digests'][pin] = '0'*64
    spec = value['spec'];spec['checkpoint_identity'] = digest(value['content_digests'])
    spec['code_digest'] = digest({k:v for k,v in value['content_digests'].items() if k.startswith('code:')})
    spec['feature_spec']['policy_digest'] = spec['checkpoint_identity']
    binding = dict(model=DATA['teacher_manifest']['spec'],sampler=DATA['teacher_manifest']['spec']['default_sampler'],
                   benchmark_protocol=DATA['teacher_manifest']['spec']['allowed_benchmark_protocols'][0])
    with pytest.raises(ValueError,match='changed'):
        admission.compatibility(model_spec(spec),value,config,binding)


def test_no_blanket_teacher_or_code_exception(release):
    _, provider, config, manifest = release
    from exp.r17_integration.models.cosmos.cache_teacher import DATA
    binding = dict(model=copy.deepcopy(DATA['teacher_manifest']['spec']),
        sampler=DATA['teacher_manifest']['spec']['default_sampler'],
        benchmark_protocol=DATA['teacher_manifest']['spec']['allowed_benchmark_protocols'][0])
    binding['model']['norm_identity'] = 'other'
    with pytest.raises(ValueError,match='pinned B500'):provider.library_validator(config,binding)
    value = copy.deepcopy(manifest);value['content_digests']['code:r17:unknown_producer.py'] = '0'*64
    value['spec']['checkpoint_identity'] = digest(value['content_digests'])
    value['spec']['code_digest'] = digest({k:v for k,v in value['content_digests'].items() if k.startswith('code:')})
    value['spec']['feature_spec']['policy_digest'] = value['spec']['checkpoint_identity']
    binding['model'] = DATA['teacher_manifest']['spec']
    with pytest.raises(ValueError,match='unreviewed'):
        admission.compatibility(model_spec(value['spec']),value,config,binding)


def test_manifest_native_geometry_ast_is_not_exempted(release):
    source = Path(admission.__file__).with_name('manifest.py').read_text()
    changed = source.replace('ActionStateLayout(16, 16, 7','ActionStateLayout(32, 16, 7')
    assert admission._native_manifest_ast(source) != admission._native_manifest_ast(changed)


def test_python_version_ast_empty_fields_do_not_change_source_proof():
    import ast
    first = ast.parse('def f():\n return 1').body[0]
    before = admission._ast_dump(first)
    first.type_params = []
    assert admission._ast_dump(first) == before
