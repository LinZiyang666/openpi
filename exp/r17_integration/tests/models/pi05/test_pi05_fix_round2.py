"""Round-2 canonical recipe/config and latency/service ledger regressions."""
from dataclasses import asdict, replace

import pytest
import yaml

from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
from exp.r17_integration.common.config import load_cache_config, build_shared_storage
from exp.r17_integration.models.pi05.provider import register_provider, config_document, make_connection_factory
from exp.r17_integration.tests.models.pi05.test_pi05 import fixture_adapter, observation, request
from exp.r17_integration.tests.models.pi05.test_pi05_e2e import tiny_store


@pytest.mark.parametrize('k', [10, 2, 1])
def test_all_emitted_pi_configs_carry_canonical_table_even_when_recipe_disabled(tmp_path, k):
    adapter, _ = fixture_adapter()
    provider = register_provider(adapter, provider_id='p2_r2_plain_' + tmp_path.name)
    doc = config_document(provider, k=k, evidence_dir=tmp_path/'miss')
    assert doc['recipe']['task_table_identity'] == FORMAL_SPEC.task_table_identity
    assert tuple(doc['recipe']['ordered_tasks']) == FORMAL_SPEC.ordered_tasks
    path = tmp_path/'plain.yaml'
    path.write_text(yaml.safe_dump(doc))
    config = load_cache_config(path)
    assert not config.recipe.enabled
    core = make_connection_factory(provider, config, build_shared_storage(config))(None, 'default')
    assert core.recipe is None
    req = replace(request(), benchmark_protocol=FORMAL_SPEC.identity, task_id=43, explicit_seed=123)
    assert core.infer({**observation(), '__request__': asdict(req)})['__hit_meta__']['actual_nfe'] == k


@pytest.mark.parametrize('mutation', ['reordered', 'subset', 'wrong_digest'])
def test_noncanonical_pi_recipe_table_rejected_before_any_fit(tmp_path, monkeypatch, mutation):
    adapter, _ = fixture_adapter()
    provider = register_provider(adapter, provider_id='p2_r2_table_' + tmp_path.name)
    store, recipe = tiny_store(adapter, tmp_path/'teacher')
    changed = dict(recipe)
    if mutation == 'reordered':
        changed['ordered_tasks'] = sorted(recipe['ordered_tasks'])
    elif mutation == 'subset':
        changed['ordered_tasks'] = ['reach-v3', 'assembly-v3']
    else:
        changed['task_table_identity'] = 'incorrect_digest'
    with pytest.raises(ValueError, match='task table'):
        config_document(provider, evidence_dir=tmp_path/'miss', recipe=changed)
    # Raw hand-written YAML must fail as well, independent of our emitter.
    doc = config_document(provider, evidence_dir=tmp_path/'miss', recipe=recipe)
    doc['recipe'].update(changed)
    path = tmp_path/'bad.yaml'
    path.write_text(yaml.safe_dump(doc))
    from exp.r17_integration.search import fit
    monkeypatch.setattr(fit, 'fit_recipe', lambda *a, **k: pytest.fail('invalid table reached fitting'))
    with pytest.raises(ValueError, match='task_table|task table|ordered_tasks'):
        load_cache_config(path)


def test_canonical_partial_teacher_rows_work_but_query_id_name_mismatch_fails(tmp_path):
    adapter, model = fixture_adapter()
    provider = register_provider(adapter, provider_id='p2_r2_query_' + tmp_path.name)
    store, recipe = tiny_store(adapter, tmp_path/'teacher')
    doc = config_document(provider, evidence_dir=tmp_path/'miss', recipe={**recipe, 'guard': False, 'periodic_gap': None})
    path = tmp_path/'recipe.yaml'
    path.write_text(yaml.safe_dump(doc))
    config = load_cache_config(path)
    factory = make_connection_factory(provider, config, build_shared_storage(config))
    core = factory(None, 'default')
    req = replace(request(), benchmark_protocol=FORMAL_SPEC.identity, task_id=43, explicit_seed=123)
    result = core.infer({**observation(), '__request__': asdict(req)})
    assert result['__hit_meta__']['source'] == 'library' and result['__hit_meta__']['actual_nfe'] == 0
    before = list(model.counts)
    invalid = replace(req, task_name='assembly-v3', decision_seq=1)
    with pytest.raises(ValueError, match='task'):
        core.infer({**observation(), '__request__': asdict(invalid),
                    '__execution__': {'decision_seq': 0, 'executed_count': 5, 'terminal': False}})
    assert model.counts == before


def test_real_core_subtracts_contended_model_lock_wait_from_service_time(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    from exp.r17_integration.common.provider import AdapterProvider, register_provider as register
    from exp.r17_integration.models.pi05.adapter import sampler_for_steps
    from exp.r17_integration.benchmarks.execution import cost_row
    adapter, _ = fixture_adapter()
    lock, entered = threading.Lock(), threading.Event()
    class Contended:
        wait = 0.0
        def __getattr__(self, name):
            return getattr(adapter, name)
        def run_stage1(self, prepared):
            begin = time.perf_counter()
            entered.set()
            with lock:
                self.wait += (time.perf_counter() - begin) * 1000.
                return adapter.run_stage1(prepared)
        def drain_queue_wait_ms(self):
            wait, self.wait = self.wait, 0.0
            return wait
    provider = AdapterProvider('p2_r2_queue_' + tmp_path.name, adapter.metadata(), Contended, sampler_for_steps)
    register(provider)
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config_document(provider, k=2, evidence_dir=tmp_path/'miss')))
    config = load_cache_config(path)
    core = make_connection_factory(provider, config, build_shared_storage(config))(None, 'default')
    req = replace(request(), benchmark_protocol=FORMAL_SPEC.identity, task_id=43, explicit_seed=123)
    lock.acquire()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(core.infer, {**observation(), '__request__': asdict(req)})
        try:
            assert entered.wait(10)
            threading.Event().wait(.025)  # Deliberate CPU queue delay, not service work.
        finally:
            lock.release()
        out = future.result(timeout=10)
    row = cost_row(out, {'task_uid': 'queue_fixture'}, 0)
    assert row['queue_wait_ms'] >= 20.
    assert row['service_ms'] >= 0 and row['service_ms'] < row['cost_ms']
    assert row['service_ms'] == pytest.approx(row['cost_ms'] - row['queue_wait_ms'], abs=1e-5)
    assert row['stage_service_ms']['stage1'] == pytest.approx(
        row['stage_ms']['stage1'] - row['stage_queue_wait_ms']['stage1'], abs=1e-5)
    assert row['pricing_time_field'] == 'service_ms'
    assert row['gpu_service_ms'] is None


def test_pi_serve_emit_configs_is_cpu_only_and_carries_canonical_table(tmp_path, monkeypatch):
    from exp.r17_integration.models.pi05 import serve
    monkeypatch.setattr(serve, 'content_manifest', lambda path:
        {'checkpoint_identity': 'fixture_weight_digest', 'norm_identity': 'fixture_norm_digest', 'code_digest': 'fixture_code'})
    monkeypatch.setattr(serve, 'load_adapter', lambda *a, **kw: pytest.fail('config emission loaded a model'))
    serve.main(['--checkpoint', 'synthetic_metadata_only', '--emit-configs', str(tmp_path)])
    for k in (10, 2, 1):
        document = yaml.safe_load((tmp_path/f'k{k}.yaml').read_text())
        assert document['recipe']['task_table_identity'] == FORMAL_SPEC.task_table_identity
        assert tuple(document['recipe']['ordered_tasks']) == FORMAL_SPEC.ordered_tasks
        assert document['adapter_binding']['num_steps'] == k
