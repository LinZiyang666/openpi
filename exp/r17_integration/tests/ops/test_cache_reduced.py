"""Reduced cache bindings, staging, planning and MISS receipts on CPU."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from exp.r17_integration.common.config import build_shared_storage, load_cache_config
from exp.r17_integration.common.contracts import SampleResult
from exp.r17_integration.common.fake_adapter import FakeAdapter
from exp.r17_integration.common.provider import AdapterProvider, binding_for, register_provider, validate_binding, validate_config
from exp.r17_integration.common.serving import make_connection_factory
from exp.r17_integration.ops import cache_knob as knob, cache_study as study
from exp.r17_integration.tests.ops.test_cache_knob import fake_campaign
from exp.r17_integration.tests.ops.test_cache_study import fixture_model, freeze


def native_sampler(model, spec, k):
    if model == 'dp_libero':
        from exp.r17_integration.models.dp.sampler import SamplerProfile
        return SamplerProfile(num_steps=k).to_spec(spec.default_sampler.config['config'])
    if model == 'fastwam_libero':
        from exp.r17_integration.models.fastwam.manifest import make_sampler
        from exp.r17_integration.tests.models.fastwam.conftest import Scheduler
        return make_sampler(Scheduler(1000, 5), k)
    from exp.r17_integration.models.cosmos.manifest import make_sampler
    return make_sampler(k)


@pytest.fixture(scope='module')
def reduced_study(tmp_path_factory):
    directory = tmp_path_factory.mktemp('reduced_knob')
    gaps = study.HERE/'knob/gaps.jsonl'
    return knob.build(directory/'dev', gaps=gaps, miss_steps='reduced')


@pytest.mark.parametrize('model', tuple(k for k in knob.REDUCED if k not in knob.VLA_PROFILES))
def test_reduced_yaml_native_identity_teacher_and_complete_plan(reduced_study, model, tmp_path, monkeypatch):
    from exp.r17_integration.common import provider as registry
    monkeypatch.setattr(registry, '_providers', {})
    manifest = study.read(reduced_study)
    rows = manifest['models'][model]
    assert len(rows) == 74
    parent = study.read(Path(rows[0]['library'])/'READY.json')
    _, benchmark = study.profiles(model, parent)
    spec = replace(study.model_spec(parent['binding']['model']), allowed_benchmark_protocols=(benchmark.identity,))
    sampler_for_steps = lambda k: native_sampler(model, spec, k)
    provider = AdapterProvider('teacher_template_'+model, spec, lambda: None, sampler_for_steps)
    if model == 'dp_libero':
        from exp.r17_integration.models.dp.admission import admission_validators
        binding_check, config_check = admission_validators(spec, sampler_for_steps)
        provider = replace(provider, binding_validator=binding_check, config_validator=config_check)
    elif model == 'cosmos_libero':
        from exp.r17_integration.models.cosmos.provider import validate_config as cosmos_check
        provider = replace(provider, config_validator=cosmos_check)
    register_provider(provider)
    k, run = knob.REDUCED[model]
    pure_sampler = study.read(Path('/data/mw_finetune/eval')/run/'assets/execution_sampler.json')
    assert sampler_for_steps(k).identity == study.digest(pure_sampler)
    for row in rows:
        config = load_cache_config(reduced_study.parent/row['file'])
        validate_config(config)
        _, sampler = validate_binding(config.adapter_binding)
        assert sampler.num_steps == k and sampler.identity == study.digest(pure_sampler)
        assert config.recipe.teacher_sampler_identity == study.digest(parent['binding']['sampler'])
        if model == 'fastwam_libero':
            from exp.r17_integration.models.fastwam.ops.library_compatibility import library_validator
            assert library_validator(spec)(config, parent['binding'])['status'] == 'PASS'
    campaign = fake_campaign(reduced_study, model, tmp_path/'assets')
    _, _, tasks, endpoints = study.plan(campaign)
    assert len(tasks) == len({t.task_uid for t in tasks}) == 22200
    pure = [t for t in tasks if t.extra['cache_group'].endswith('_pure')]
    assert len(pure) == 600 and {t.extra['library_size'] for t in pure} == {500}
    assert {t.orig_init_state_idx for t in pure} == set(range(30))
    assert endpoints[0].allowed_k == (k,)
    assert all(t.extra['num_steps'] == k and endpoints[0].accepts(t) for t in tasks)
    assert all(t.extra.get('expected_nfe', k) == (1 if model == 'cosmos_libero' else k) for t in tasks)


@pytest.mark.parametrize('model', tuple(k for k in knob.REDUCED if k not in knob.VLA_PROFILES))
def test_reduced_cpu_stage_and_rebind_cannot_restore_default(reduced_study, model, tmp_path, monkeypatch):
    from exp.r17_integration.common import provider as registry
    monkeypatch.setattr(registry, '_providers', {})
    manifest = study.read(reduced_study)
    rows = [r for r in manifest['models'][model] if r['suite'] == 'libero_spatial' and
            (r['library_size'] == 50 and r['group'].endswith('_koff') or r['group'].endswith('_pure'))]
    for row in rows:
        row['file'] = str(reduced_study.parent/row['file'])
        row['indices'] = [29]
    manifest['models'] = {model: rows}
    manifest.pop('digest'); manifest['digest'] = study.digest(manifest)
    source = tmp_path/'STUDY.json'; study.atomic_json(source, manifest)
    parent = study.read(Path(rows[0]['library'])/'READY.json')
    _, benchmark = study.profiles(model, parent)
    spec = replace(study.model_spec(parent['binding']['model']), allowed_benchmark_protocols=(benchmark.identity,))
    provider = AdapterProvider('reduced_seal_'+model, spec, lambda: None, lambda k: native_sampler(model, spec, k))
    register_provider(provider)
    assets = tmp_path/'assets'; assets.mkdir()
    campaign = dict(model=model.removesuffix('_libero'), cache_study=str(source), assets=str(assets),
        local_assets=str(assets), server_node='local', suites=list(study.SUITES), run_name='cpu',
        endpoint_host='127.0.0.1', public_port=23400, workers=2)
    def forbid_remote(*args, **kwargs):
        raise AssertionError('CPU test attempted remote I/O')
    study.stage_inputs(campaign, tmp_path, SimpleNamespace(transfer=forbid_remote, remote=forbid_remote))
    k = knob.REDUCED[model][0]
    assert all(r['num_steps'] == k for r in study.read(assets/'cache_inputs/manifest.json')['groups'])
    study.seal(provider, campaign, benchmark)
    for row in study.read(assets/'CACHE_STUDY.json')['groups']:
        config = load_cache_config(assets/row['file'])
        assert config.adapter_binding['num_steps'] == k
        assert config.adapter_binding['sampler_identity'] == native_sampler(model, spec, k).identity
        assert config.recipe.teacher_sampler_identity == study.digest(parent['binding']['sampler'])
        assert config.recipe.serve_loeo and row['indices'] == [29]


def test_reduced_test_builder_keeps_frozen_task_levels_and_k(tmp_path):
    model = 'cosmos_libero'
    manifest = study.read(study.HERE/'yaml/STUDY.json')
    table = {}
    for suite in study.SUITES:
        row = next(r for r in manifest['models'][model] if r['suite'] == suite)
        names = [n for n in yaml.safe_load((study.HERE/'yaml'/row['file']).read_text())['recipe']['ordered_tasks']
                 if n.startswith(suite+'/')]
        table[suite] = {str(size): {name:list(knob.LEVELS)[i % len(knob.LEVELS)] for i,name in enumerate(names)}
                        for size in knob.SIZES}
    path = tmp_path/'levels.json'; study.atomic_json(path, table)
    test = knob.build(tmp_path/'test', gaps=study.HERE/'knob/gaps.jsonl', model=model, table=path, miss_steps='reduced')
    campaign = fake_campaign(test, model, tmp_path/'assets')
    _, _, tasks, endpoints = study.plan(campaign)
    assert len(tasks) == len({t.task_uid for t in tasks}) == 6000
    assert endpoints[0].allowed_k == (2,)
    for task in tasks:
        extra = task.extra
        level = table[task.experiment][str(extra['library_size'])][extra['task_name']]
        assert extra['cache_group'] == 'crg_'+level
        assert extra['num_steps'] == 2 and extra['expected_nfe'] == 1
        assert extra['init_pool'] == 'A' and not extra['serve_loeo']


@pytest.mark.parametrize('model', tuple(k for k in knob.REDUCED if k not in knob.VLA_PROFILES))
@pytest.mark.parametrize('trigger', ['stall', 'periodic', 'pure'])
def test_each_cache_model_call_uses_bound_reduced_sampler(model, trigger, tmp_path):
    provider = fixture_model(model)
    parent = study.read(Path('/data/r17_library')/(model+'_libero_spatial')/'bpool/READY.json')
    native_spec = study.model_spec(parent['binding']['model'])
    spec = replace(provider.model_spec, default_sampler=native_sampler(model, native_spec, native_spec.default_sampler.num_steps))
    k = knob.REDUCED[model][0]
    active = native_sampler(model, native_spec, k)
    received = []
    class Stub(FakeAdapter):
        def sample(self, stage2, sampler, rng, capture):
            received.append(sampler)
            assert sampler == active and not capture.enabled
            nfe = sampler.config.get('expected_nfe', sampler.num_steps)
            return SampleResult(np.zeros_like(stage2.data), nfe, sampler)
    provider = replace(provider, provider_id=provider.provider_id+'.native', model_spec=spec, load_adapter=lambda: Stub(spec),
                       sampler_for_steps=lambda steps: native_sampler(model, native_spec, steps))
    register_provider(provider)
    commit = 16 if model == 'cosmos_libero' else 10
    config, _ = freeze(provider, tmp_path, commit, gated=True)
    config.adapter_binding = binding_for(provider, spec.allowed_benchmark_protocols[0], num_steps=k, commit_horizon=commit)
    config.recipe.periodic_gap = None if trigger == 'stall' else 0.0 if trigger == 'pure' else 1.0
    policy = make_connection_factory(provider, config, build_shared_storage(config))()
    class Tracker:
        def __init__(self, *args): pass
        def observe(self, *args): pass
        def status(self): return dict(state='slow_confirmed' if trigger == 'stall' else 'ok', W=2)
    policy.recipe.tracker_class = Tracker
    policy.on_episode_start(extra_metadata=dict(task_name='reach', task_id=0, original_init_idx=0))
    observation = dict(state=np.zeros(len(spec.layout.valid_state_indices), np.float32))
    calls = []
    for seq in range(3):
        obs = dict(observation)
        if seq: obs['__execution__'] = dict(decision_seq=seq-1, executed_count=spec.layout.replan_steps)
        meta = policy.infer(obs)['__hit_meta__']
        if meta['source'] == 'policy':
            assert meta['sampler_identity'] == active.identity
            assert meta['stage_calls'] == (1, 1, 1 if model == 'cosmos_libero' else k)
            assert meta['actual_nfe'] == (1 if model == 'cosmos_libero' else k)
            assert meta['stall_call'] if trigger == 'stall' else meta['knob_call']
            calls.append(seq)
    assert received and (calls == [0, 1, 2] if trigger == 'pure' else True)


def test_reduced_commands_match_owner_topology_and_cosmos_nvme():
    expected = {'dp_libero': (4, 8, 'local=48'), 'fastwam_libero': (2, 16, 'timan107=45,timan1=40'),
                'cosmos_libero': (8, 1, 'timan107=45,timan1=40')}
    for model, (replicas, batch, fleet) in expected.items():
        prepare, run = knob.commands('STUDY.json', model, miss_steps='reduced')
        assert '--replicas '+str(replicas) in prepare and '--max-batch '+str(batch) in prepare
        assert '--worker-counts '+fleet in prepare and '--mps' in prepare and '--prepare-only' in prepare
        assert '--denoising-steps' not in prepare  # Cache MISS selection belongs to the YAML.
        assert 'knobdev' in prepare and '--port 234' in prepare
        if model == 'cosmos_libero':
            assert '--driver-mode legacy' in prepare and '--memory-budget-gib 76' in prepare
            assert '/tmp/r17_driver/knobdevcos_reduced/driver' in run and 'ln -s' in run
        assert 'knobtest' in knob.commands('STUDY.json', model, miss_steps='reduced', mode='test')[0]


def test_aggregate_cosmos_k2_accepts_one_nfe_and_rejects_full_step_receipt(tmp_path):
    driver = tmp_path/'driver'; driver.mkdir()
    extra = dict(cache_group='crg_pure', miss_mode='stall_only', corrector=True, num_steps=2, expected_nfe=1,
        bundle_identity='bundle', recipe_identity='recipe', commit_horizon=16, periodic_gap=0.0)
    study.atomic_json(tmp_path/'campaign.json', dict(cache_study=True, model='cosmos', replicas=1))
    study.atomic_json(tmp_path/'EPISODE_PLAN.json', dict(tasks=[dict(task_uid='cpu', experiment='libero_spatial', extra=extra)]))
    row = dict(task_uid='cpu', attempt=1, accepted=True, source='policy', decision_seq=0, cursor=0,
        stage_calls=[1,1,1], actual_nfe=1, bundle_identity='bundle', recipe_identity='recipe', commit_horizon=16,
        capture_enabled=False, stall_call=False, knob_call=True, resident_pid=17, resident_generation='same')
    ack = dict(task_uid='cpu', attempt=1, accepted=True, _kind='execution_ack', decision_seq=0,
        executed_count=2, cursor_end=2, terminal=True, server_terminal_ack={'terminal':True})
    (driver/'journal.jsonl').write_text(json.dumps(dict(task_uid='cpu', attempt=1, status='done', accepted=True, success=True))+'\n')
    study.atomic_json(driver/'DRIVER_RECEIPT.json', dict(status='PASS', elapsed_seconds=1))
    for nfe in (1, 2, 5):
        row.update(stage_calls=[1,1,nfe], actual_nfe=nfe)
        (driver/'per_step.jsonl').write_text(json.dumps(row)+'\n'+json.dumps(ack)+'\n')
        assert study.aggregate(tmp_path)['status'] == ('PASS' if nfe == 1 else 'PARTIAL')
