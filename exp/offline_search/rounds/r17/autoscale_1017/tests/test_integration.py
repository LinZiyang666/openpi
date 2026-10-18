"""Post-apply regression tests import the shared working tree directly."""
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import threading
import types

import pytest

import importlib
from exp.offline_search.rounds.r17.autoscale_1017.node import atomic, identity, read
from exp.offline_search.rounds.r17.autoscale_1017.policy import GIB


@pytest.fixture(scope='module')
def proposed():
    base = json.loads((Path(__file__).parents[1]/'PATCH_BASE.json').read_text())
    return {p:Path(p).read_text() for p in base}


def load(proposed, path, name, monkeypatch):
    return importlib.import_module(path.removesuffix('.py').replace('/', '.').removeprefix('src.'))


def test_applied_shared_modules_compile(proposed):
    for path, source in proposed.items():
        if path.endswith('.py'):compile(source, path, 'exec')
    from exp.offline_search.rounds.r17.filter_failed.worker_resilience import NODES
    assert 'timan108' not in NODES


def test_actual_r17_queue_identity_is_independent_of_worker_count_and_recycling(tmp_path):
    from exp.r17_integration.conductor.manifest import EndpointSpec, episode_tasks
    from exp.r17_integration.conductor.queue import CompatibleEpisodeQueue
    from exp.r17_integration.conductor.strategy import MetaWorldStrategy
    from exp.r17_integration.benchmarks.libero_bridge import LiberoBenchmarkSpec, SUITES, decision_seed as libero_seed
    from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC, decision_seed as mw_seed
    from openpi.conductor.task import ServerEndpoint
    from examples.libero.worker_entry import _init_state_index
    endpoint = ServerEndpoint('server', 1234)
    libero = LiberoBenchmarkSpec(tuple((s + '/task' + str(i), 'task ' + str(i), s) for s in SUITES for i in range(10)))
    for spec in (FORMAL_SPEC, libero):
        tasks = episode_tasks(campaign='same-science', arm='off', bundle_id='off', endpoint=endpoint,
                              model_identity='model', norm_identity='norm', layout_identity='layout', spec=spec,
                              partition='smoke', names=[r[0] for r in spec.tasks[:2]], indices=list(range(8)))
        original = json.dumps([asdict(t) for t in tasks], sort_keys=True)
        reference = None
        for label, counts in [('fixed1', [1]), ('fixed250', [250]), ('dynamic', [350, 250, 150, 44, 2, 6, 1])]:
            endpoints = [EndpointSpec(endpoint, 'model', 'norm', 'layout', spec.identity, capacity=32)]
            strategy = MetaWorldStrategy(tasks, endpoints=endpoints)
            yamls = sorted({t.yaml_id for t in tasks})
            graph = strategy.plan(yamls, dict.fromkeys(yamls, endpoint))
            queue = CompatibleEpisodeQueue(graph, endpoints=endpoints, dispatch_path=tmp_path / (spec.benchmark + label + '.jsonl'), eval_concurrency=32)
            for stage in queue.pending_setups():
                queue.mark_setup_running(stage.stage_id)
                queue.mark_setup_done(stage.stage_id)
            seen, turn = {}, 0
            while len(seen) < len(tasks):
                batch = [queue.next_task(endpoint.key) for _ in range(counts[turn % len(counts)])]
                # Reversed completions and regenerated worker labels must not
                # affect the planned UID, init-state index or explicit seed.
                for task in reversed([t for t in batch if t is not None]):
                    seed_fn = libero_seed if spec.benchmark == 'libero' else mw_seed
                    seen[task.task_uid] = (task.task_id, _init_state_index(task, 'orig'),
                        task.extra.get('seed', spec.seed), tuple(seed_fn(task.extra['task_name'], task.orig_init_state_idx, k, spec.seed) for k in range(5)))
                    assert queue.mark_result(task.task_uid, success=True, retriable=False, attempt=task.attempt)
                    assert not queue.mark_result(task.task_uid, success=True, retriable=False, attempt=task.attempt)
                turn += 1
                assert turn < 100
            if reference is None:reference = seen
            assert seen == reference
            assert json.dumps([asdict(t) for t in tasks], sort_keys=True) == original


def test_patched_worker_reports_before_recycle_and_does_not_mutate_task(tmp_path, proposed, monkeypatch):
    module = load(proposed, 'src/openpi/conductor/worker.py', 'autoscale_test_worker', monkeypatch)
    from openpi.conductor.task import EpisodeResult, EpisodeTask
    from openpi.conductor import protocol
    monkeypatch.setenv('OPENPI_WORKER_OBSERVER', 'exp.offline_search.rounds.r17.autoscale_1017.worker:Observer')
    monkeypatch.setenv('OPENPI_AUTOSCALE_MAX_EPISODES', '2')
    monkeypatch.setenv('OPENPI_AUTOSCALE_MAX_SECONDS', '1800')
    marker = tmp_path / 'RECYCLE.json'
    monkeypatch.setenv('R17_WORKER_RECYCLE_RECEIPT', str(marker))
    sent, recorded = [], []
    monkeypatch.setattr(protocol, 'send_message', lambda sock, kind, data: sent.append((kind, data)))
    class Runner:
        def run(self, task, report):
            recorded.append(asdict(task))
            return EpisodeResult(task.task_uid, True, 5)
        def close(self):pass
    loop = module.WorkerLoop('any-worker-generation', 'server:1', Runner(), connect=lambda: None)
    task = EpisodeTask('cell', 'yaml', 'eval', 'libero_spatial', 0, 7, 7, 'server', 1, 'off', attempt=3, extra=dict(seed=7))
    original = copy.deepcopy(asdict(task))
    loop._run_one(None, task)
    with pytest.raises(SystemExit) as exit:
        loop._run_one(None, task)
    assert exit.value.code == 75
    assert len(sent) == 2 and all(kind == protocol.MSG_REPORT_RESULT for kind, _ in sent)
    assert all(data['attempt'] == 3 for _, data in sent)
    assert all('_autoscale_episode' in data for _, data in sent)
    assert recorded == [original, original] and asdict(task) == original
    lifetime = load(proposed, 'exp/r17_integration/ops/worker_lifetime.py', 'autoscale_test_lifetime', monkeypatch)
    assert lifetime.planned_recycle(dict(pid=os.getpid(), start_ticks=identity(os.getpid()), recycle_receipt=str(marker)), 75)


def test_time_recycle_only_after_completed_episode(tmp_path, proposed, monkeypatch):
    from exp.offline_search.rounds.r17.autoscale_1017.worker import Observer
    clock = [0]
    monkeypatch.setattr('exp.offline_search.rounds.r17.autoscale_1017.worker.time.monotonic', lambda: clock[0])
    monkeypatch.setenv('OPENPI_AUTOSCALE_MAX_EPISODES', '32')
    monkeypatch.setenv('OPENPI_AUTOSCALE_MAX_SECONDS', '1800')
    marker = tmp_path / 'recycle.json'
    monkeypatch.setenv('R17_WORKER_RECYCLE_RECEIPT', str(marker))
    observer = Observer('worker', object())
    clock[0] = 2000
    observer.recycle()  # No episode has completed yet.
    observer.completed = 1
    with pytest.raises(SystemExit):observer.stop_requested()
    lifetime = load(proposed, 'exp/r17_integration/ops/worker_lifetime.py', 'autoscale_time_lifetime', monkeypatch)
    assert lifetime.planned_recycle(dict(pid=os.getpid(), start_ticks=identity(os.getpid()), recycle_receipt=str(marker)), 75)


def test_timing_wraps_server_wait_without_changing_returned_actions(monkeypatch):
    from exp.offline_search.rounds.r17.autoscale_1017.worker import Observer, TimedClient
    clock = [0.]
    monkeypatch.setattr('exp.offline_search.rounds.r17.autoscale_1017.worker.time.monotonic', lambda: clock[0])
    observer = Observer('worker', object())
    output = dict(actions=object(), server_timing=dict(infer_ms=250))
    class Client:
        def infer(self, value):
            clock[0] += 4
            return output
    client = TimedClient(Client(), observer)
    observer.begin(types.SimpleNamespace(task_uid='task', attempt=1, yaml_id='yaml'))
    clock[0] += 1
    assert client.infer({'unchanged': True}) is output
    sample = observer.result()
    assert sample['sim_render_s'] == 1
    assert sample['server_wait_s'] == 4
    assert sample['service_s'] == .25


def test_patched_driver_transport_drains_individual_at_next_pull(proposed, monkeypatch):
    module = load(proposed, 'src/openpi/conductor/driver.py', 'autoscale_test_driver', monkeypatch)
    from openpi.conductor import protocol
    # Use real transport handler with fake frames, no sockets/processes.
    frames = iter([(protocol.MSG_REPORT_RESULT, dict(task_uid='current', attempt=4)),
                   (protocol.MSG_PULL, dict(worker_id='newest', server_host='server:1', _autoscale=dict(retire=True)))])
    events = []
    monkeypatch.setattr(protocol, 'recv_message', lambda conn: next(frames))
    monkeypatch.setattr(protocol, 'send_message', lambda conn, kind, data: events.append(('reply', kind, data)))
    driver = object.__new__(module.ConductorDriver)
    driver._autoscaler = None
    driver._record_pull = lambda payload, peer: payload['worker_id']
    driver._record_result = lambda worker: None
    driver.handle_result = lambda payload: events.append(('result', payload['task_uid']))
    driver.handle_pull = lambda key: (_ for _ in ()).throw(AssertionError('retired worker must not pull another task'))
    driver._scheduler = types.SimpleNamespace(mark_result=lambda *a, **k: None)
    class Conn:
        def getpeername(self):return ('test', 0)
        def close(self):pass
        def settimeout(self, timeout):pass
    driver._handle_conn(Conn(), lambda: len(events) == 2)
    assert events == [('result', 'current'), ('reply', protocol.MSG_SHUTDOWN, {})]


def test_gpu_environment_creation_is_serialized_across_threads(tmp_path, monkeypatch):
    from exp.offline_search.rounds.r17.autoscale_1017.worker import creation_lock
    monkeypatch.setenv('OPENPI_AUTOSCALE_LEDGER', str(tmp_path))
    monkeypatch.setenv('OPENPI_AUTOSCALE_GPU_UUID', 'CPU-FAKE-UUID')
    first_entered, release_first, second_entered = (threading.Event() for _ in range(3))
    def first():
        with creation_lock():
            first_entered.set()
            assert release_first.wait(2)
    def second():
        assert first_entered.wait(2)
        with creation_lock():second_entered.set()
    one, two = threading.Thread(target=first), threading.Thread(target=second)
    one.start();two.start()
    assert first_entered.wait(2)
    assert not second_entered.wait(.03)
    release_first.set()
    one.join(2);two.join(2)
    assert second_entered.is_set() and not one.is_alive() and not two.is_alive()


def test_patched_driver_rates_count_only_actual_accepted_attempts(tmp_path, proposed, monkeypatch):
    module = load(proposed, 'src/openpi/conductor/driver.py', 'autoscale_accept_driver', monkeypatch)
    from openpi.conductor.task import EpisodeResult, EpisodeTask, ServerEndpoint, Stage, TaskGraph
    from openpi.conductor import protocol
    endpoint = ServerEndpoint('fake', 1)
    task = EpisodeTask('yaml__eval__0__0', 'yaml', 'eval', 'test', 0, 0, 0, 'fake', 1, 'off')
    class Strategy:
        def plan(self, yamls, assignments):
            graph = TaskGraph()
            graph.add_stage(Stage('stage', 'yaml', 'eval', endpoint, [task]))
            return graph
    driver = module.ConductorDriver(Strategy(), yaml_weights=dict(yaml=1), servers=[endpoint],
                                    journal_path=str(tmp_path / 'journal.jsonl'), ctl_factory=lambda s: None)
    driver.scheduler.mark_setup_running('stage')
    driver.scheduler.mark_setup_done('stage')
    _, assigned = driver.handle_pull(endpoint.key)
    received = []
    driver._autoscaler = types.SimpleNamespace(result=lambda payload, accepted: received.append(accepted))
    result = protocol.result_to_wire(EpisodeResult(task.task_uid, True, 5, attempt=assigned['task']['attempt']))
    driver.handle_result(result)
    driver.handle_result(result)
    assert received == [True, False]


def test_legacy_agent_memory_guard_covers_start_and_respawn(tmp_path, proposed, monkeypatch):
    module = load(proposed, 'src/openpi/conductor/agent.py', 'autoscale_test_agent', monkeypatch)
    monkeypatch.setenv('OPENPI_WORKER_GOVERNOR', 'exp.offline_search.rounds.r17.autoscale_1017.legacy')
    monkeypatch.setenv('OPENPI_WORKER_MEMORY_GUARD', '1')
    monkeypatch.setenv('OPENPI_AUTOSCALE_LEDGER', str(tmp_path / 'ledger'))
    monkeypatch.setenv('OPENPI_AUTOSCALE_RUN', str(tmp_path / 'run'))
    spawned = []
    class Handle:
        pid = os.getpid()
        code = None
        def poll(self):return self.code
        def wait(self, timeout):return self.code
        def terminate(self):raise AssertionError('No killing in autoscale actuation')
    def spawn(spec, host, port):
        spawned.append(spec)
        return Handle()
    from exp.offline_search.rounds.r17.autoscale_1017 import legacy, fleet
    monkeypatch.setattr(legacy.AgentGovernor, 'supervise', lambda self: None)
    class Slots:
        def __init__(self, c, r):self.assignments = {}
        def view(self):return None
        def acquire(self, i):
            self.assignments[i] = dict(device=0, uuid='fake');return True
        def release(self, i):self.assignments.pop(i, None)
    monkeypatch.setattr(fleet, 'Slots', Slots)
    agent = module.WorkerAgent([module.WorkerSpec('one', 'server:1', '0')], 'driver', 1, spawn_fn=spawn)
    atomic(agent._governor.local.control_path, dict(campaign=agent._governor.local.campaign['run_name'], time=__import__('time').time(), targets=dict(local=1)))
    g = agent._governor.local
    available, clock = [19 * GIB], [0.]
    g.clock = lambda: clock[0]
    g.reader = lambda: (available[0], {'full': dict(avg10=0)})
    atomic(g.history_path, [dict(peak=4 * GIB, complete=True)])
    agent.start()
    assert spawned == []
    available[0] = 80 * GIB
    clock[0] = 2
    agent.supervise_once()
    assert spawned == []
    clock[0] = 62
    agent.supervise_once()
    assert len(spawned) == 1
    assert spawned[0].env['OPENPI_WORKER_OBSERVER'].endswith(':Observer')
    agent._handles['one'].code = 75
    available[0] = 19 * GIB
    agent._governor.last = 0
    clock[0] = 64
    agent.supervise_once()
    assert len(spawned) == 1  # A planned recycle does not bypass the floor.


def test_r17_fleet_recycles_waits_below_floor_then_recovers(tmp_path, monkeypatch):
    from exp.offline_search.rounds.r17.autoscale_1017 import fleet
    from exp.offline_search.rounds.r17.autoscale_1017.node import LocalGovernor
    from exp.r17_integration.ops import worker_disk, worker_lifetime
    clock = [0.]
    monkeypatch.setattr(fleet.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(fleet.time, 'time', lambda: 1700000000 + clock[0])
    monkeypatch.setattr(fleet.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(worker_disk, 'admission', lambda *a: dict(admitted=True))
    monkeypatch.setattr(worker_lifetime, 'planned_recycle', lambda entry, code: code == 75)
    class Slots:
        def __init__(self, *a):pass
        def view(self):return None
        def acquire(self, i):return True
        def release(self, i):pass
    monkeypatch.setattr(fleet, 'Slots', Slots)
    campaign = dict(run_name='lane', domain='metaworld', worker_node='local', node_workers=1,
                    autoscale=dict(hold_s=6), seconds=80, worker_max_episodes=32)
    g = LocalGovernor(campaign, tmp_path / 'run', ledger=tmp_path / 'ledger', clock=lambda: clock[0],
                      reader=lambda: ((19 if 4 <= clock[0] < 16 else 70) * GIB, {'full': dict(avg10=0)}),
                      process_reader=lambda *a: (3 * GIB, 4 * GIB))
    atomic(g.history_path, [dict(peak=4 * GIB, complete=True)])
    atomic(g.control_path, dict(campaign='lane', time=1700000000, targets=dict(local=1)))
    children, logs, spawned = [], {}, []
    class Handle:
        pid = os.getpid()
        def __init__(self):self.born = clock[0]
        def poll(self):return 75 if clock[0] >= self.born + 4 else None
        def wait(self, timeout):return self.poll()
    def launch(index, generation):
        spawned.append(clock[0])
        logs[index] = types.SimpleNamespace(close=lambda: None)
        return Handle(), dict(pid=os.getpid(), start_ticks=identity(os.getpid()), worker_id='one', index=index, generation=generation)
    def draining():return len(spawned) >= 2
    report = dict(worker_failures=[], worker_recycles=[])
    fleet.run_fleet(campaign, g.run, report, children, logs, launch, lambda: False, draining, g)
    assert spawned[0] == 0 and spawned[1] >= 22
    assert report['status'] == 'PASS'
    assert len(report['worker_recycles']) == 2
    assert not report['worker_failures']


def test_trial_cli_and_metaworld_recycle_defaults(proposed, monkeypatch):
    from exp.r17_integration.ops import local_workers
    counts = load(proposed, 'exp/r17_integration/ops/local_workers.py', 'autoscale_cli_counts', monkeypatch).counts
    monkeypatch.setattr(local_workers, 'counts', counts)
    f = load(proposed, 'exp/r17_integration/ops/formal_eval.py', 'exp.r17_integration.ops.autoscale_test_eval', monkeypatch)
    for mode in ('fixed', 'dynamic'):
        args = f.parse_args(['--model', 'cosmos_libero', '--server-node', 'h100', '--worker-counts', 'timan1=150,timan107=100',
                             '--replicas', '8', '--official-50', '--suites', 'libero_spatial',
                             '--worker-autoscale', mode, '--semantic-campaign', 'as17_co_pair',
                             '--run-name', 'as17_co_' + mode, '--seconds', '21600', '--prepare-only'])
        assert args.worker_autoscale == mode and args.worker_counts == {'timan1': 150, 'timan107': 100}
        assert args.semantic_campaign == 'as17_co_pair'
    assert f.parse_args(['--model', 'dp_mw', '--prepare-only']).worker_max_episodes == 32
    with pytest.raises(ValueError, match='disabled'):
        f.remote('timan108', ['anything'])


def test_bundle_closure_contains_all_runtime_imports_without_server_edits():
    from exp.offline_search.rounds.r17.autoscale_1017.bundle import files
    paths = files()
    assert len(paths) == len(set(paths))
    assert all(Path(p).is_file() for p in paths)
    assert any(p.endswith('/bootstrap.py') for p in paths)
    assert any(p.endswith('/timan1_gpu2.py') for p in paths)
    assert not any(p.endswith('/serving.py') for p in paths)


def test_plan_comparison_and_closed_receipt_check_are_strict(tmp_path, monkeypatch):
    from exp.offline_search.rounds.r17.autoscale_1017 import compare
    a, b = tmp_path / 'fixed', tmp_path / 'dynamic'
    for root in (a, b):atomic(root / 'EPISODE_PLAN.json', dict(tasks=[dict(task_uid='same', orig_init_state_idx=7, extra=dict(seed=7))]))
    assert compare.same_plan(a, b) == 1
    atomic(b / 'EPISODE_PLAN.json', dict(tasks=[dict(task_uid='same', orig_init_state_idx=8, extra=dict(seed=7))]))
    with pytest.raises(ValueError):compare.same_plan(a, b)
    monkeypatch.setattr(compare, 'wait_io', lambda: None)
    monkeypatch.setattr(sys, 'argv', ['compare', str(a), '--closed'])
    atomic(a / 'RECEIPT.json', dict(status='PASS', all_owned_closed=False))
    with pytest.raises(ValueError):compare.main()
    atomic(a / 'RECEIPT.json', dict(status='PASS', all_owned_closed=True))
    compare.main()
