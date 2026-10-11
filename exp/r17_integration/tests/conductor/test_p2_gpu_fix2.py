"""Q's first-probe-only WorkerLoop repro and strict TCP peer admission."""
import copy
import json
import socket
import threading
import time
from types import SimpleNamespace as NS

import pytest

from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC, MetaWorldBenchmarkAdapter
from exp.r17_integration.benchmarks.libero_bridge import LiberoBenchmarkAdapter
from exp.r17_integration.conductor.driver import BenchmarkConductorDriver
from exp.r17_integration.conductor.manifest import EndpointSpec, episode_tasks, validate_worker_probe
from exp.r17_integration.conductor.strategy import MetaWorldStrategy
from exp.r17_integration.tests.benchmarks.test_libero import spec as libero_spec
from openpi.conductor.protocol import result_to_wire
from openpi.conductor.task import EpisodeResult, ServerEndpoint
from openpi.conductor.worker import WorkerLoop


def make_driver(tmp_path, domain, *, validator=True):
    benchmark = FORMAL_SPEC if domain == "metaworld" else libero_spec()
    endpoint = ServerEndpoint("local", 1)
    episodes = episode_tasks(campaign="probe_cpu", arm="full", bundle_id="full", endpoint=endpoint,
        model_identity="m", norm_identity="n", layout_identity="l", spec=benchmark,
        names=["reach-v3" if domain == "metaworld" else benchmark.tasks[0][0]], indices=[0, 1])
    strategy = MetaWorldStrategy(episodes, endpoints=[EndpointSpec(endpoint, "m", "n", "l", benchmark.identity)])
    driver = BenchmarkConductorDriver(strategy, dispatch_path=tmp_path / "dispatch.jsonl",
        yaml_weights={episodes[0].yaml_id: 2}, servers=[endpoint],
        journal_path=str(tmp_path / "journal.jsonl"), ctl_factory=lambda e: NS(close=lambda: None),
        worker_probe_validator=(lambda p: validate_worker_probe(p, spec=benchmark)) if validator else None)
    if domain == "metaworld":
        probe = MetaWorldBenchmarkAdapter().worker_probe(
            {"renderer": "egl", "GL_VENDOR": "NVIDIA CPU fixture; no renderer instantiated"},
            {"task_data_identity": "CPU fixture"})
        probe["packages"] = {"metaworld": "3.0.0", "mujoco": "3.14.0", "gymnasium": "1.3.0"}
    else:
        probe = LiberoBenchmarkAdapter(benchmark).worker_probe({"libero_code_digest": "CPU fixture",
                                                               "task_data": {"identity": "CPU fixture"}})
    return driver, probe


@pytest.mark.parametrize("domain", ["metaworld", "libero"])
def test_q_repro_valid_first_then_probeless_second_and_terminal_pull(tmp_path, domain):
    driver, probe = make_driver(tmp_path, domain)
    peer = ("127.0.0.1", 1234)
    payload = {"worker_id": "cpu-worker", "server_host": "local:1"}
    try:
        assert driver._record_pull({**payload, "probe": probe}, peer) == "cpu-worker"
        assert driver._record_pull(payload, peer) == "cpu-worker"
        assert driver._record_pull(payload, peer) == "cpu-worker"
        assert driver.worker_census["cpu-worker"]["pulls"] == 3
        assert driver.worker_census["cpu-worker"]["probe"] == probe
        assert driver.worker_census["cpu-worker"]["peers"] == ["127.0.0.1:1234"]
        assert not driver.rejected_worker_probes
    finally:
        driver.close()


@pytest.mark.parametrize("domain", ["metaworld", "libero"])
@pytest.mark.parametrize("case", ["first_missing", "new_peer", "unknown_peer", "new_worker", "empty", "null",
                                  "modified_valid", "mismatching", "endpoint_missing", "endpoint_explicit"])
def test_probe_reuse_rejects_every_unattested_identity_and_explicit_change(tmp_path, domain, case):
    driver, probe = make_driver(tmp_path, domain)
    peer = ("127.0.0.1", 1234)
    base = {"worker_id": "cpu-worker", "server_host": "local:1"}
    accepted = case != "first_missing"
    if accepted:
        driver._record_pull({**base, "probe": probe}, peer)
    before = copy.deepcopy(driver.worker_census)
    payload = dict(base)
    if case == "new_peer": peer = ("127.0.0.1", 4321)
    if case == "unknown_peer": peer = None
    if case == "new_worker": payload["worker_id"] = "another-worker"
    if case == "empty": payload["probe"] = {}
    if case == "null": payload["probe"] = None
    if case == "modified_valid": payload["probe"] = {**probe, "extra_attestation": "changed"}
    if case == "mismatching": payload["probe"] = {**probe, "benchmark_protocol": "wrong"}
    if case.startswith("endpoint_"):
        payload["server_host"] = "other:2"
        if case == "endpoint_explicit": payload["probe"] = probe
    try:
        with pytest.raises(ValueError):
            driver._record_pull(payload, peer)
        assert driver.worker_census == before
        assert not (tmp_path / "dispatch.jsonl").exists()
        assert len(driver.rejected_worker_probes) == 1
        row = driver.rejected_worker_probes[0]
        assert row["pull_payload"] == payload
        assert row["probe_present"] == ("probe" in payload)
        assert row["probe"] == payload.get("probe")
        previous = before.get(payload["worker_id"], {}).get("probe")
        assert row["accepted_probe"] == previous
        assert row["peer"] == (list(peer) if isinstance(peer, tuple) else peer)
        assert row["reason"]
        # Rejection evidence and accepted attestation own their nested data.
        expected = copy.deepcopy(row)
        payload["server_host"] = "mutated after rejection"
        if isinstance(payload.get("probe"), dict): payload["probe"]["later"] = True
        assert row == expected
    finally:
        driver.close()


@pytest.mark.parametrize("domain", ["metaworld", "libero"])
def test_reconnect_requires_full_identical_probe_and_retains_immutable_attestation(tmp_path, domain):
    driver, probe = make_driver(tmp_path, domain)
    original = copy.deepcopy(probe)
    payload = {"worker_id": "cpu-worker", "server_host": "local:1", "probe": probe}
    try:
        driver._record_pull(payload, ("127.0.0.1", 1234))
        probe["later"] = "caller mutation"
        assert driver.worker_census["cpu-worker"]["probe"] == original
        payload["probe"] = original
        driver._record_pull(payload, ("127.0.0.1", 4321))
        driver._record_pull({k: v for k, v in payload.items() if k != "probe"}, ("127.0.0.1", 4321))
        assert driver.worker_census["cpu-worker"]["pulls"] == 3
        assert driver.worker_census["cpu-worker"]["peers"] == ["127.0.0.1:1234", "127.0.0.1:4321"]
    finally:
        driver.close()


@pytest.mark.parametrize("domain", ["metaworld", "libero"])
def test_actual_stock_workerloop_one_tcp_connection_two_episodes_then_shutdown(tmp_path, domain):
    driver, probe = make_driver(tmp_path, domain)
    driver.drive_stages_once()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0)); listener.listen(1)
    worker_socket = socket.create_connection(listener.getsockname(), timeout=3)
    connection, _ = listener.accept(); listener.close()
    connection.settimeout(3)
    errors = []
    stop = threading.Event()
    def serve():
        try: driver._handle_conn(connection, stop.is_set)
        except BaseException as exc: errors.append(exc)
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    loop = WorkerLoop("cpu-worker", "local:1", NS(), connect=lambda: None, probe=probe)
    try:
        for _ in range(2):
            task = loop._pull(worker_socket)
            assert task.task_uid and task.attempt == 1
            loop._report_result(worker_socket, EpisodeResult(task.task_uid, True, 2, attempt=task.attempt))
        # Results have no transport ack. Complete the real stock stage only
        # after its handler has accepted both reports, then pull shutdown.
        deadline = time.monotonic() + 3
        while driver.worker_census["cpu-worker"]["results"] < 2 and time.monotonic() < deadline:
            stop.wait(.001)
        assert driver.worker_census["cpu-worker"]["results"] == 2
        driver.drive_stages_once()
        assert driver.scheduler.all_done()
        assert loop._pull(worker_socket) is None
        census = driver.worker_census["cpu-worker"]
        assert census["pulls"] == 3 and census["results"] == 2
        assert len(census["peers"]) == 1 and census["probe"] == probe
        assert not driver.rejected_worker_probes
        rows = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
        assert len(rows) == 2 and all(row["accepted"] and row["attempt"] == 1 for row in rows)
    finally:
        stop.set(); worker_socket.close(); thread.join(timeout=4); driver.close()
    assert not thread.is_alive() and not errors


def test_disabled_probe_gate_census_is_byte_identical_to_stock(tmp_path):
    from openpi.conductor.driver import ConductorDriver
    driver, _ = make_driver(tmp_path, "metaworld", validator=False)
    stock = ConductorDriver(driver._strategy, yaml_weights={key: 2 for key in driver._assignment},
                            servers=list(driver._assignment.values()), journal_path=str(tmp_path / "stock_journal.jsonl"),
                            ctl_factory=lambda e: NS(close=lambda: None))
    from unittest.mock import patch
    try:
        with patch("openpi.conductor.driver.time.time", return_value=1234):
            for payload, peer in [({"worker_id": "legacy", "server_host": "local:1"}, ("127.0.0.1", 1)),
                                  ({"worker_id": "legacy", "server_host": "other:2", "probe": {}}, ("127.0.0.1", 2)),
                                  ({"worker_id": "legacy", "server_host": "local:1", "probe": {"legacy": True}}, None)]:
                assert driver._record_pull(payload, peer) == stock._record_pull(payload, peer)
        assert json.dumps(driver.worker_census, sort_keys=True).encode() == json.dumps(stock.worker_census, sort_keys=True).encode()
        assert not driver.rejected_worker_probes
    finally:
        driver.close()
