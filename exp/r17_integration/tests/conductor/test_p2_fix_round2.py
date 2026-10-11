"""P2 behavioral witnesses for the second review (CPU fixtures only)."""
from dataclasses import replace
import json
from types import SimpleNamespace as NS

import pytest

from openpi.conductor.driver import is_retriable_error
from openpi.conductor.protocol import result_to_wire, task_from_wire
from openpi.conductor.scheduler import StageState
from openpi.conductor.task import EpisodeResult, ServerEndpoint
from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
from exp.r17_integration.conductor.driver import MetaWorldConductorDriver
from exp.r17_integration.conductor.manifest import EndpointSpec, episode_tasks
from exp.r17_integration.conductor.strategy import MetaWorldStrategy, MetaWorldDescentStrategy, PerTaskDescent
from exp.r17_integration.tests.benchmarks.test_libero import spec as libero_spec


def descent(tmp_path, benchmark=FORMAL_SPEC, *, names=None, retries=1):
    names = names or [benchmark.tasks[0][0]]
    endpoint = EndpointSpec(ServerEndpoint('local', 1), 'm', 'n', 'l', benchmark.identity)
    controller = PerTaskDescent(names, ['k2', 'k1'], {n: {30: True, 31: True} for n in names}, n_dev=2)
    episodes = []
    for name in names:
        for level in controller.levels:
            group = episode_tasks(campaign='r2', arm=level, bundle_id=level, endpoint=endpoint.endpoint,
                model_identity='m', norm_identity='n', layout_identity='l', num_steps=2,
                partition='dev', names=[name], indices=[30, 31], spec=benchmark)
            episodes.extend(replace(t, extra={**t.extra, 'descent_level': level}) for t in group)
    strategy = MetaWorldDescentStrategy(episodes, endpoints=[endpoint], controller=controller)
    driver = MetaWorldConductorDriver(strategy, dispatch_path=tmp_path/'dispatch.jsonl',
        yaml_weights={t.yaml_id: 2 * len(names) for t in episodes}, servers=[endpoint.endpoint],
        journal_path=str(tmp_path/'journal.jsonl'), ctl_factory=lambda e: NS(close=lambda: None),
        scheduler_kwargs={'max_episode_retries': retries})
    return controller, strategy, driver


def finish(driver):
    sent = []
    for _ in range(40):
        driver.drive_stages_once()
        _, payload = driver.handle_pull('local:1')
        if 'task' in payload:
            t = task_from_wire(payload['task'])
            sent.append(t)
            driver.handle_result(result_to_wire(EpisodeResult(t.task_uid, True, 3, attempt=t.attempt)))
        if driver.scheduler.all_done():
            break
    assert driver.scheduler.all_done()
    assert not any(rt.dispatched for rt in driver.scheduler._rt.values())
    return sent


@pytest.mark.parametrize('error', ["ConfigValidationError('bad bundle')", 'ConfigValidation: wrong layout',
                                 "SidecarError('unavailable')", "RuntimeError('FatalEpisodeError: invalid')"])
@pytest.mark.parametrize('libero', [False, True], ids=['mw', 'libero'])
def test_every_nonretryable_error_stops_one_task_and_other_task_descends(tmp_path, error, libero):
    benchmark = libero_spec() if libero else FORMAL_SPEC
    names = [r[0] for r in benchmark.tasks[:2]]
    controller, strategy, driver = descent(tmp_path, benchmark, names=names)
    try:
        assert not is_retriable_error(error)
        driver.drive_stages_once()
        first = task_from_wire(driver.handle_pull('local:1')[1]['task'])
        in_flight = task_from_wire(driver.handle_pull('local:1')[1]['task'])
        failed_name = first.extra['task_name']
        driver.handle_result(result_to_wire(EpisodeResult(first.task_uid, False, 0, error, attempt=first.attempt)))
        assert failed_name in controller.stopped
        # Already-sent work is fenced and journaled, but cannot fill the failed cohort.
        driver.handle_result(result_to_wire(EpisodeResult(in_flight.task_uid, True, 3, attempt=in_flight.attempt)))
        sent = finish(driver)
        assert all(t.extra['task_name'] != failed_name for t in sent)
        assert len(sent) == 4
        receipt = strategy.freeze({'model': 'm', 'protocol': benchmark.identity})
        assert receipt['status'] == 'diagnostic_error'
        assert receipt['errors'][failed_name]['error'] == error
        assert not controller.rows[(failed_name, 'k2')]
        with pytest.raises(ValueError, match='terminal dev errors'):
            controller.authorize_test(receipt['freeze_hash'], receipt['identities'])
        records = [json.loads(line) for line in driver.journal.path.read_text().splitlines()]
        assert any(r['error'] == error and r['accepted'] for r in records if r.get('error'))
    finally:
        driver.close()
    # Live and resumed controllers must agree about the failed task and healthy task.
    resumed, resumed_strategy, resumed_driver = descent(tmp_path, benchmark, names=names)
    try:
        finish(resumed_driver)
        assert resumed.errors == controller.errors
        assert resumed.chosen == controller.chosen
        assert resumed_strategy.freeze(receipt['identities'])['status'] == 'diagnostic_error'
    finally:
        resumed_driver.close()


@pytest.mark.parametrize('boundary', ['ensure_bundle', 'on_dispatch'])
def test_dispatch_failure_exhaustion_is_journaled_and_freezable(tmp_path, boundary):
    controller, strategy, driver = descent(tmp_path)
    def fail(*args):
        raise ConnectionError('CPU fixture pre-send failure')
    setattr(strategy, boundary, fail)
    try:
        driver.drive_stages_once()
        # Repeated pulls may consume different pending original indices.
        for _ in range(8):
            try:
                driver.handle_pull('local:1')
            except ConnectionError:
                pass
            driver.drive_stages_once()
            if driver.scheduler.all_done():
                break
        assert driver.scheduler.all_done()
        assert not any(rt.dispatched for rt in driver.scheduler._rt.values())
        receipt = strategy.freeze({'model': 'm'})
        assert receipt['errors'] and receipt['status'] == 'diagnostic_error'
        rows = [json.loads(line) for line in driver.journal.path.read_text().splitlines()]
        assert len(rows) == 1 and rows[0]['accepted'] and rows[0]['attempt'] == 2
        assert 'retry budget exhausted' in rows[0]['error']
    finally:
        driver.close()


def test_transient_dispatch_callback_failure_requeues_exact_attempt(tmp_path):
    controller, strategy, driver = descent(tmp_path)
    original = strategy.on_dispatch
    count = 0
    def callback(task):
        nonlocal count
        count += 1
        if count == 1:
            raise ConnectionError('temporary CPU fixture failure')
        return original(task)
    strategy.on_dispatch = callback
    try:
        driver.drive_stages_once()
        with pytest.raises(ConnectionError):
            driver.handle_pull('local:1')
        assert not any(rt.dispatched for rt in driver.scheduler._rt.values())
        sent = finish(driver)
        assert any(t.attempt == 2 for t in sent)
        assert controller.errors == {}
        assert strategy.freeze({'model': 'm'})['status'] == 'qualified_dev'
    finally:
        driver.close()


def test_stage_with_undecided_previous_cohort_stops_without_sending_next_level(tmp_path):
    controller, strategy, driver = descent(tmp_path)
    try:
        # Simulate a completed scheduler stage whose controller cohort never arrived.
        first = next(s for s in driver._graph.stages.values() if s.setup['descent_level'] == 'k2')
        driver.scheduler._rt[first.stage_id].state = StageState.DONE
        finish(driver)
        receipt = strategy.freeze({'model': 'm'})
        assert receipt['status'] == 'diagnostic_error'
        assert 'unfinished' in next(iter(receipt['errors'].values()))['error']
        assert not (tmp_path/'dispatch.jsonl').exists()
    finally:
        driver.close()


def test_existing_stage_bundles_are_selected_without_reloading_artifacts(tmp_path):
    endpoint = EndpointSpec(ServerEndpoint('local', 1), 'm', 'n', 'l', FORMAL_SPEC.identity)
    tasks = [episode_tasks(campaign='r2', arm=arm, bundle_id=arm, endpoint=endpoint.endpoint,
        model_identity='m', norm_identity='n', layout_identity='l', names=['reach-v3'], indices=[0])[0]
        for arm in ('full', 'recipe')]
    registry, loads = set(), []
    class Control:
        def select_bundle(self, bundle):
            if bundle not in registry:
                raise KeyError('unknown bundle')
        def load_cache_config(self, path, *, yaml_id, bundle_id):
            loads.append(bundle_id)
            registry.add(bundle_id)
        def close(self):
            pass
    strategy = MetaWorldStrategy(tasks, endpoints=[endpoint], bundles={a: tmp_path/(a+'.yaml') for a in ('full', 'recipe')})
    driver = MetaWorldConductorDriver(strategy, dispatch_path=tmp_path/'dispatch.jsonl',
        yaml_weights={t.yaml_id: 1 for t in tasks}, servers=[endpoint.endpoint],
        journal_path=str(tmp_path/'journal.jsonl'), ctl_factory=lambda e: Control())
    try:
        assert loads == ['full', 'recipe']
        driver.drive_stages_once()
        assert loads == ['full', 'recipe']
        registry.remove('recipe')  # Partial registry loss should load only the missing bundle.
        strategy.on_stage_begin(next(iter(driver._graph.stages.values())), None, None)
        assert loads == ['full', 'recipe', 'recipe']
        registry.clear()  # Actual restart still restores all required bundles.
        strategy.on_stage_begin(next(iter(driver._graph.stages.values())), None, None)
        assert loads == ['full', 'recipe', 'recipe', 'full', 'recipe']
    finally:
        driver.close()


def test_runner_ledger_keeps_native_queue_wait_service_and_gpu_scope_separate():
    from exp.r17_integration.benchmarks.execution import cost_row
    native = {'cost_ms': 20., 'queue_wait_ms': 12., 'service_ms': 8.,
        'stage_ms': {'stage1': 17., 'prepare': 2., 'stage3': 1.},
        'stage_queue_wait_ms': {'stage1': 12.}, 'stage_service_ms': {'stage1': 5., 'prepare': 2., 'stage3': 1.},
        'gpu_ms': None, 'timing_kind': 'host_wall',
        'service_timing_scope': 'policy_host_wall_excluding_model_lock_queue_wait',
        'queue_wait_status': 'measured', 'pricing_time_field': 'service_ms',
        'capture_enabled': False, 'pricing_eligible': True}
    row = cost_row({'__hit_meta__': native, 'server_timing': {'infer_ms': 21.}}, {'task_uid': 'r2'}, 0)
    assert row['cost_ms'] == 20. and row['server_timing']['infer_ms'] == 21.
    for field in ('queue_wait_ms', 'service_ms', 'stage_queue_wait_ms', 'stage_service_ms', 'gpu_ms',
                  'queue_wait_status', 'pricing_time_field', 'service_timing_scope'):
        assert row[field] == native[field]
    # An older/uninstrumented endpoint supplies latency, not known service time.
    unknown = cost_row({'server_timing': {'infer_ms': 21.}}, {}, 0)
    assert unknown['cost_ms'] == 21.
    assert unknown['queue_wait_ms'] is None and unknown['service_ms'] is None
    assert unknown['pricing_eligible'] is not True


@pytest.mark.parametrize('marker', [{'allow_lossy_sampler': True}, {'sampler_study_label': 'plain-k1-study'},
                                  {'sampler_quality_status': 'lossy_study'}])
def test_lossy_plain_study_markers_never_enter_dev_descent(tmp_path, marker):
    controller, strategy, driver = descent(tmp_path)
    try:
        episodes = [replace(t, extra={**t.extra, **marker}) for t in strategy.episodes]
        with pytest.raises(ValueError, match='lossy sampler study'):
            MetaWorldDescentStrategy(episodes, endpoints=strategy.endpoints, controller=controller)
    finally:
        driver.close()


def test_opt_in_native_lock_view_keeps_exclusion_and_local_wait_accounting():
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from exp.r17_integration.benchmarks.execution import instrument_model_lock
    native = threading.RLock()
    first, second = [instrument_model_lock(NS(model_lock=native)) for _ in range(2)]
    entering = threading.Event()
    def operation():
        entering.set()
        with second.model_lock:
            return b'unchanged native output bytes'
    with ThreadPoolExecutor(max_workers=1) as pool:
        first.model_lock.acquire()
        try:
            future = pool.submit(operation)
            assert entering.wait(5)
            threading.Event().wait(.025)
            assert not future.done()
        finally:
            first.model_lock.release()
        assert future.result(timeout=5) == b'unchanged native output bytes'
    assert second.drain_queue_wait_ms() >= 20.
    assert second.drain_queue_wait_ms() == 0.
    assert instrument_model_lock(second) is second


def test_gr_default_yaml_emission_bridge_carries_table_and_preserves_options():
    from dataclasses import asdict
    from exp.r17_integration.common.fake_adapter import fake_provider, fake_spec
    from exp.r17_integration.models.groot.serve import default_config
    from exp.r17_integration.conductor.manifest import canonical_config_document
    # Actual P5 emitter, with explicitly synthetic CPU metadata instead of weights.
    model = replace(fake_spec(horizon=16), allowed_benchmark_protocols=(FORMAL_SPEC.identity,))
    provider = fake_provider('p2_gr_emitter_fixture', model)
    original = asdict(default_config(provider, commit_horizon=10))
    emitted = canonical_config_document(original, FORMAL_SPEC)
    # Round 3: the native GR emitter now carries the canonical table itself;
    # the bridge must be an idempotent no-op on it.
    assert tuple(original['recipe']['ordered_tasks']) == FORMAL_SPEC.ordered_tasks
    assert original['recipe']['task_table_identity'] == FORMAL_SPEC.task_table_identity
    assert emitted == original
    assert emitted['recipe']['task_table_identity'] == FORMAL_SPEC.task_table_identity
    assert tuple(emitted['recipe']['ordered_tasks']) == FORMAL_SPEC.ordered_tasks
    assert emitted['adapter_binding'] == original['adapter_binding']
    assert emitted['checkpoints'] == original['checkpoints']
    assert emitted['recipe']['enabled'] is False
