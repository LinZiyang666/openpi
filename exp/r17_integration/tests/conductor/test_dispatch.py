from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest

from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC as MT50_SPEC

# Synthetic configurable dev partition; production dev remains undecided.
FORMAL_SPEC = replace(MT50_SPEC, dev_seed=101, dev_indices=tuple(range(30, 50)))
from exp.r17_integration.conductor.manifest import EndpointSpec, episode_tasks
from exp.r17_integration.conductor.queue import CompatibleEpisodeQueue
from exp.r17_integration.conductor.strategy import MetaWorldStrategy, PerTaskDescent
from exp.r17_integration.conductor.driver import MetaWorldConductorDriver
from openpi.conductor.task import ServerEndpoint
from openpi.conductor.protocol import result_to_wire
from openpi.conductor.task import EpisodeResult


def setup(tmp_path, *, names=None, indices=(30, 31)):
    endpoints = [EndpointSpec(ServerEndpoint("local", p), "m", "n", "l", FORMAL_SPEC.identity) for p in (1, 2)]
    endpoints.append(replace(endpoints[0], endpoint=ServerEndpoint("other", 3), model_identity="wrong"))
    tasks = episode_tasks(campaign="test", arm="k2", bundle_id="k2_gen1", endpoint=endpoints[0].endpoint,
                          model_identity="m", norm_identity="n", layout_identity="l", num_steps=2,
                          spec=FORMAL_SPEC, partition="dev", names=names, indices=indices)
    s = MetaWorldStrategy(tasks, endpoints=endpoints)
    graph = s.plan([tasks[0].yaml_id], {tasks[0].yaml_id: endpoints[0].endpoint})
    q = CompatibleEpisodeQueue(graph, endpoints=endpoints, dispatch_path=tmp_path / "dispatch.jsonl")
    for stage in q.pending_setups():
        q.mark_setup_done(stage.stage_id)
    return endpoints, tasks, s, graph, q


def test_global_concurrent_pull_compatible_least_busy_exact_episode(tmp_path):
    endpoints, tasks, _, _, q = setup(tmp_path)
    with ThreadPoolExecutor(8) as pool:
        outputs = list(pool.map(lambda _: q.next_task("local:1"), range(100)))
    assert {r.task_uid for r in outputs} == {r.task_uid for r in tasks}
    assert len(outputs) == len(tasks) == 100
    assert {r.server.key for r in outputs} == {"local:1", "local:2"}
    assert q.busy() == {"local:1": 50, "local:2": 50, "other:3": 0}
    persisted = [json.loads(r) for r in (tmp_path / "dispatch.jsonl").read_text().splitlines()]
    assert len(persisted) == 100 and all(r["attempt"] == 1 for r in persisted)


def test_bundle_preload_only_targets_compatible_episode_endpoints(tmp_path):
    endpoints, tasks, strategy, _, _ = setup(tmp_path, names=["reach-v3"])
    # Same geometry/protocol but a different checkpoint may not receive this YAML.
    strategy.bundles = {"k2_gen1": tmp_path / "k2.yaml", "unused": tmp_path / "unreferenced.yaml"}
    requests = []
    strategy.preload(lambda endpoint: NS(load_cache_config=lambda path, **kwargs:
        requests.append((endpoint.key, path, kwargs))))
    assert [r[0] for r in requests] == ["local:1", "local:2"]
    assert all(r[2]["bundle_id"] == "k2_gen1" for r in requests)
    # A fixed live-K endpoint also must not receive an incompatible candidate bundle.
    strategy.endpoints[1] = replace(endpoints[1], allowed_k=(10,))
    requests.clear()
    strategy.preload(lambda endpoint: NS(load_cache_config=lambda path, **kwargs:
        requests.append((endpoint.key, path, kwargs))))
    assert [r[0] for r in requests] == ["local:1"]


def test_restart_attempt_persist_and_late_duplicate_fences(tmp_path):
    endpoints, tasks, _, graph, q = setup(tmp_path, names=["reach-v3"])
    a = q.next_task("local:1")
    assert q.mark_result(a.task_uid, success=False, retriable=True, attempt=a.attempt)
    q.next_task("local:1")  # the other episode
    retry = q.next_task("local:1")
    assert retry.task_uid == a.task_uid and retry.attempt == 2
    assert not q.mark_result(a.task_uid, success=True, retriable=False, attempt=1)
    assert q.mark_result(retry.task_uid, success=False, retriable=False, attempt=2)
    assert not q.mark_result(retry.task_uid, success=True, retriable=False, attempt=2)
    resumed = CompatibleEpisodeQueue(graph, endpoints=endpoints, dispatch_path=tmp_path / "dispatch.jsonl")
    for s in resumed.pending_setups():
        resumed.mark_setup_done(s.stage_id)
    assert resumed.next_task("local:1").attempt == 3


def test_dispatch_fsync_failure_never_returns_or_mutates_task(tmp_path, monkeypatch):
    _, tasks, _, _, q = setup(tmp_path, names=["reach-v3"])
    monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk fixture failure")))
    with pytest.raises(OSError):
        q.next_task("local:1")
    assert q.busy() == {"local:1": 0, "local:2": 0, "other:3": 0}
    assert q.progress()["episodes_done"] == 0


def test_manifest_partition_identity_duplicate_prompts_and_commit(tmp_path):
    endpoints, tasks, _, _, _ = setup(tmp_path, names=["push-v3", "push-back-v3"])
    assert len({t.task_uid for t in tasks}) == 4
    with pytest.raises(ValueError, match="leakage"):
        episode_tasks(campaign="c", arm="x", bundle_id="b", endpoint=endpoints[0].endpoint,
                      model_identity="m", norm_identity="n", layout_identity="l", spec=FORMAL_SPEC, partition="dev", indices=[0])
    with pytest.raises(ValueError, match="compatible"):
        MetaWorldStrategy([replace(tasks[0], extra={**tasks[0].extra, "num_steps": 4})], endpoints=endpoints)


def test_driver_reuses_framework_journal_rows_and_census(tmp_path):
    endpoints, tasks, strategy, _, _ = setup(tmp_path, names=["reach-v3"])
    calls = []
    class Ctl:
        def load_cache_config(self, path, yaml_id):
            calls.append((path, yaml_id))
        def close(self):
            pass
    d = MetaWorldConductorDriver(strategy, dispatch_path=tmp_path / "driver_dispatch.jsonl",
                                 yaml_weights={tasks[0].yaml_id: 2}, servers=[e.endpoint for e in endpoints[:2]],
                                 journal_path=str(tmp_path / "episodes.jsonl"), ctl_factory=lambda e: Ctl())
    d.drive_stages_once()
    _, payload = d.handle_pull("local:1")
    dispatched = payload["task"]
    d.handle_result(result_to_wire(EpisodeResult(dispatched["task_uid"], False, 2,
                   per_step_rows=[{"step_idx": 0}], attempt=dispatched["attempt"])))
    record = json.loads((tmp_path / "episodes.jsonl").read_text())
    assert record["accepted"] is True and record["success"] is False and record.get("error") is None
    assert record["status"] == "failed"
    assert d._per_step_rows[0]["accepted"] is True
    assert record["attempt"] == dispatched["attempt"]


def test_fifty_task_descent_complete_gate_first_fail_and_freeze():
    names = [t[0] for t in FORMAL_SPEC.tasks]
    refs = {n: {i: True for i in range(30, 50)} for n in names}
    state = PerTaskDescent(names, ["k2", "k1", "periodic"], refs)
    with pytest.raises(ValueError, match="finish"):
        state.freeze({"model": "m"})
    for task_index, name in enumerate(names):
        # k2 loses at most one /20 == 5pp, passes inclusively.
        for i in range(30, 50):
            state.expect(name, i, 1)
            assert not state.accept(name, "k2", i, attempt=1, success=True, error="transport")
            assert not state.accept(name, "k2", i, attempt=0, success=True)
            assert not state.accept(name, "k2", i, attempt=1, success=True, accepted=False)
            assert state.accept(name, "k2", i, attempt=1, success=i != 30)
            assert not state.accept(name, "k2", i, attempt=1, success=True)
        for i in range(30, 50):
            state.expect(name, i, 2)
            assert state.accept(name, "k1", i, attempt=2, success=i >= (32 if task_index % 2 else 31))
        if task_index % 2:
            assert name in state.stopped and state.chosen[name] == "k2"
        else:
            for i in range(30, 50):
                state.expect(name, i, 3)
                assert state.accept(name, "periodic", i, attempt=3, success=True)
            assert state.chosen[name] == "periodic"
    receipt = state.freeze({"model": "m", "norm": "n", "yaml": "y"})
    assert state.authorize_test(receipt["freeze_hash"], receipt["identities"]) == state.chosen
    with pytest.raises(ValueError):
        state.authorize_test(receipt["freeze_hash"], {"model": "changed"})
    receipt["choices"][names[0]] = "tampered"
    with pytest.raises(ValueError, match="tampered"):
        state.authorize_test(receipt["freeze_hash"], receipt["identities"])


def test_worker_client_only_import_fresh_process():
    result = subprocess.run([sys.executable, "-c", "import json; from exp.r17_integration.conductor.worker_entry import import_probe; print(json.dumps(import_probe()))"],
                            text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["client_only"] is True


def test_renderer_first_render_and_close_reuse_existing_lock(monkeypatch):
    import contextlib
    from exp.mw_finetune import rendering as R
    calls = []
    @contextlib.contextmanager
    def lock():
        calls.append("lock")
        yield
        calls.append("unlock")
    monkeypatch.setattr(R, "env_build_lock", lock)
    monkeypatch.setattr(R, "renderer_stamp", lambda: {"GL_VENDOR": "CPU fixture"})
    env = R.LockedEnv(NS(render=lambda: calls.append("render"), close=lambda: calls.append("close")))
    env.render()
    env.render()
    env.close()
    assert calls == ["lock", "render", "unlock", "render", "lock", "close", "unlock"]


def test_worker_probe_gate_before_dispatch_and_reconnect_identity(tmp_path):
    from exp.r17_integration.benchmarks.metaworld import MetaWorldBenchmarkAdapter
    from exp.r17_integration.conductor.manifest import validate_worker_probe
    endpoints, tasks, strategy, _, _ = setup(tmp_path, names=["reach-v3"])
    driver = MetaWorldConductorDriver(strategy, dispatch_path=tmp_path / "probe_dispatch.jsonl",
        yaml_weights={tasks[0].yaml_id: 2}, servers=[endpoints[0].endpoint],
        journal_path=str(tmp_path / "probe_journal.jsonl"), ctl_factory=lambda e: NS(close=lambda: None),
        worker_probe_validator=validate_worker_probe)
    probe = MetaWorldBenchmarkAdapter().worker_probe({"renderer": "egl", "GL_VENDOR": "NVIDIA fixture"},
                                                    {"task_data_identity": "fixture only"})
    probe["packages"] = {"metaworld": "3.0.0", "mujoco": "3.14.0", "gymnasium": "1.3.0"}
    payload = {"worker_id": "own_fixture", "server_host": "local:1", "probe": probe}
    wrong = {**probe, "replan_steps": 10}
    with pytest.raises(ValueError, match="protocol"):
        driver._record_pull({**payload, "probe": wrong}, ("127.0.0.1", 1))
    assert not (tmp_path / "probe_dispatch.jsonl").exists()
    assert not driver.worker_census
    driver._record_pull(payload, ("127.0.0.1", 1))
    driver._record_pull(payload, ("127.0.0.1", 2))
    changed = {**probe, "renderer": {"renderer": "egl", "GL_VENDOR": "NVIDIA changed fixture"}}
    with pytest.raises(ValueError, match="changed"):
        driver._record_pull({**payload, "probe": changed}, ("127.0.0.1", 2))
    assert len(driver.rejected_worker_probes) == 2
    assert driver.worker_census["own_fixture"]["pulls"] == 2
    driver.close()


def test_worker_attestation_failure_closes_debug_writer_without_renderer(tmp_path, monkeypatch):
    from dataclasses import asdict
    from exp.mw_finetune import rendering
    from exp.r17_integration.conductor import worker_entry
    from exp.r17_integration.models.pi05.adapter import model_spec
    from openpi_client import websocket_client_policy
    closed = []
    monkeypatch.setattr(rendering, "configure", lambda *a: None)
    monkeypatch.setattr(websocket_client_policy, "WebsocketClientPolicy", lambda *a: NS(
        get_server_metadata=lambda: {"r17_model_spec": asdict(model_spec("fixture_model", "fixture_norm"))},
        close=lambda: closed.append("metadata_client")))
    def reject(*args):
        raise ValueError("fixture renderer attestation rejected before any env creation")
    monkeypatch.setattr(worker_entry, "attest_worker", reject)
    directory = tmp_path / "failed_probe_capture"
    with pytest.raises(ValueError, match="attestation rejected"):
        worker_entry.main(["--worker-id", "p2_own_fixture", "--server-key", "unused:1",
            "--driver-host", "unused", "--driver-port", "1", "--debug-directory", str(directory)])
    assert closed == ["metadata_client"]
    # Writers are now lazy and selected by the actual task sampler. A failed
    # pre-dispatch probe creates no capture binding or empty teacher store.
    assert not directory.exists()


def test_real_workerloop_tcp_dispatch_attempt_result_and_census(tmp_path):
    import socket
    import threading
    import time
    from openpi.conductor.worker import WorkerLoop
    from exp.r17_integration.benchmarks.metaworld import MetaWorldBenchmarkAdapter
    from exp.r17_integration.tests.benchmarks.test_metaworld import Env, Client
    endpoints, tasks, strategy, _, _ = setup(tmp_path, names=["reach-v3"], indices=(30, 31, 32, 33))
    ctl = NS(close=lambda: None)
    driver = MetaWorldConductorDriver(strategy, dispatch_path=tmp_path / "tcp_dispatch.jsonl",
                 yaml_weights={tasks[0].yaml_id: 4}, servers=[e.endpoint for e in endpoints[:2]],
                 journal_path=str(tmp_path / "tcp_episodes.jsonl"), ctl_factory=lambda e: ctl)
    stop = threading.Event()
    thread = threading.Thread(target=lambda: driver.run(stop.is_set), daemon=True)
    workers = []
    thread.start()
    deadline = time.monotonic()+5
    while driver.port is None and time.monotonic() < deadline:
        stop.wait(.01)
    assert driver.port is not None
    try:
        for i in range(2):
            runner = MetaWorldBenchmarkAdapter(env_factory=lambda *a: Env(success=7)).make_episode_runner(
                       client_factory=lambda e: Client(), framework_context=True)
            loop = WorkerLoop(f"p2_fixture_{i}", "local:1", runner,
                       connect=lambda: socket.create_connection(("127.0.0.1", driver.port)),
                       probe={"client_only": True, "renderer": "CPU fixture, no GL"})
            worker = threading.Thread(target=lambda l=loop: l.run_forever(stop.is_set), daemon=True)
            workers.append(worker)
            worker.start()
        thread.join(timeout=10)
        assert not thread.is_alive()
        rows = [json.loads(r) for r in (tmp_path / "tcp_episodes.jsonl").read_text().splitlines()]
        assert len(rows) == 4 and all(r["accepted"] and r["attempt"] == 1 for r in rows)
        census = driver.worker_census
        assert set(census) == {"p2_fixture_0", "p2_fixture_1"}
        assert sum(r["results"] for r in census.values()) == 4
        assert all(r["probe"]["client_only"] for r in census.values())
        assert sum(r["executed_count"] for r in driver.per_step_rows if r.get("_kind") == "execution_ack") == 28
    finally:
        stop.set()
        thread.join(timeout=5)
        for w in workers:
            w.join(timeout=5)
        driver.close()


def test_client_payload_copied_files_imports_without_workspace(tmp_path):
    import shutil
    from exp.r17_integration.conductor.worker_entry import CLIENT_ONLY_FILES
    root = Path.cwd()
    for name in CLIENT_ONLY_FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    code = "import sys, json; sys.path[:0]=[sys.argv[1], sys.argv[1]+'/src']; from exp.r17_integration.conductor.worker_entry import import_probe; print(json.dumps(import_probe()))"
    result = subprocess.run([sys.executable, "-I", "-c", code, str(tmp_path)], cwd=tmp_path,
                            check=True, text=True, capture_output=True, timeout=15)
    assert json.loads(result.stdout)["client_only"]


def test_retry_exhaustion_has_terminal_error_receipt_and_resume_budget(tmp_path):
    endpoints, tasks, strategy, _, _ = setup(tmp_path, names=["reach-v3"], indices=(30,))
    driver = MetaWorldConductorDriver(strategy, dispatch_path=tmp_path / "retries.jsonl",
                 yaml_weights={tasks[0].yaml_id: 1}, servers=[endpoints[0].endpoint],
                 journal_path=str(tmp_path / "errors.jsonl"), ctl_factory=lambda e: NS(close=lambda: None),
                 scheduler_kwargs={"max_episode_retries": 1})
    driver.drive_stages_once()
    _, payload = driver.handle_pull("local:1")
    task1 = payload["task"]
    driver.handle_result(result_to_wire(EpisodeResult(task1["task_uid"], False, 0, "transport disconnected", attempt=1)))
    assert not (tmp_path / "errors.jsonl").exists()
    _, payload = driver.handle_pull("local:1")
    assert payload["task"]["attempt"] == 2
    driver.handle_result(result_to_wire(EpisodeResult(task1["task_uid"], False, 0, "transport disconnected", attempt=2)))
    row = json.loads((tmp_path / "errors.jsonl").read_text())
    assert row["accepted"] is True and row["attempt"] == 2
    assert "retry budget exhausted" in row["error"] and "transport disconnected" in row["error"]
    driver.drive_stages_once()
    assert driver.scheduler.all_done()
    driver.close()


def test_descent_strategy_driver_barriers_skip_after_first_fail_and_resume(tmp_path):
    from exp.r17_integration.conductor.strategy import MetaWorldDescentStrategy
    endpoints, _, _, _, _ = setup(tmp_path, names=["reach-v3"], indices=(30,))
    names = ["reach-v3", "push-v3"]
    refs = {name: {i: True for i in range(30, 50)} for name in names}
    episodes = []
    for level in ("k2", "k1", "periodic"):
        group = episode_tasks(campaign="descent", arm=level, bundle_id=level, endpoint=endpoints[0].endpoint,
                    model_identity="m", norm_identity="n", layout_identity="l", num_steps=1 if level == "k1" else 2,
                    spec=FORMAL_SPEC, partition="dev", names=names)
        episodes.extend(replace(t, extra={**t.extra, "descent_level": level}) for t in group)
    def make_driver():
        controller = PerTaskDescent(names, ["k2", "k1", "periodic"], refs)
        strategy = MetaWorldDescentStrategy(episodes, endpoints=endpoints, controller=controller)
        driver = MetaWorldConductorDriver(strategy, dispatch_path=tmp_path / "descent_dispatch.jsonl",
                      yaml_weights={e.yaml_id: 40 for e in episodes}, servers=[endpoints[0].endpoint],
                      journal_path=str(tmp_path / "descent_episodes.jsonl"), ctl_factory=lambda e: NS(close=lambda: None))
        return controller, strategy, driver
    controller, strategy, driver = make_driver()
    seen = []
    for _ in range(200):
        driver.drive_stages_once()
        _, p = driver.handle_pull("local:1")
        if "task" not in p:
            if driver.scheduler.all_done():
                break
            continue
        task = p["task"]
        extra = task["extra"]
        seen.append((extra["task_name"], extra["descent_level"]))
        success = not (extra["task_name"] == "reach-v3" and extra["descent_level"] == "k1" and task["orig_init_state_idx"] in (30, 31))
        payload = result_to_wire(EpisodeResult(task["task_uid"], success, 7, attempt=task["attempt"]))
        driver.handle_result(payload)
        driver.handle_result(payload)  # duplicate cannot re-enter cohort
    assert driver.scheduler.all_done()
    assert ("reach-v3", "periodic") not in seen
    assert controller.chosen == {"reach-v3": "k2", "push-v3": "periodic"}
    receipt = strategy.freeze({"model": "m", "protocol": FORMAL_SPEC.identity})
    assert controller.authorize_test(receipt["freeze_hash"], receipt["identities"]) == controller.chosen
    driver.close()
    resumed, resumed_strategy, resumed_driver = make_driver()
    assert resumed.chosen == controller.chosen and resumed.stopped == controller.stopped
    resumed_driver.close()
