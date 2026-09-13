"""The driver's worker census and the probe a worker attaches to its first pull.

A driver may sit on another machine than its fleet, so the only record of who
served a run is what the driver saw on its pull socket: every worker id, the
server it was bound to, where it dialled from, and the environment probe it
introduced itself with. The probe rides on the first pull of each connection,
so a reconnect re-introduces the worker.
"""

from __future__ import annotations

import socket
import threading
import time

from openpi.conductor import task as T  # noqa: N812
from openpi.conductor.driver import ConductorDriver
from openpi.conductor.worker import WorkerLoop

from .conftest import FakeCtl
from .conftest import FakeEpisodeRunner
from .conftest import WarmupEvalFakeStrategy

S1 = T.ServerEndpoint("127.0.0.1", 8001)

PROBE = {"hostname": "box", "torch": "2.0", "per_task_digests": {"libero_spatial": {"t": "x"}}}


def _driver(tmp_path):
    driver = ConductorDriver(
        WarmupEvalFakeStrategy(n_warmup=1, n_eval=4),
        yaml_weights={"y": 5},
        servers=[S1],
        journal_path=str(tmp_path / "j.jsonl"),
        ctl_factory=FakeCtl,
    )
    thread = threading.Thread(target=driver.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5.0
    while driver.port is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert driver.port is not None
    return driver, thread


def test_census_records_every_worker_with_its_probe_and_peer(tmp_path):
    driver, thread = _driver(tmp_path)

    def connect():
        return socket.create_connection(("127.0.0.1", driver.port))

    stop = threading.Event()
    loops = [
        WorkerLoop(f"box-{i}", S1.key, FakeEpisodeRunner(), connect=connect,
                   max_backoff_s=0.5, probe={**PROBE, "hostname": f"box{i}"})
        for i in range(2)
    ]
    threads = [threading.Thread(target=w.run_forever, kwargs={"stop": stop.is_set}, daemon=True)
               for w in loops]
    for t in threads:
        t.start()
    thread.join(timeout=15.0)
    stop.set()
    assert not thread.is_alive()

    census = driver.worker_census
    assert set(census) == {"box-0", "box-1"}
    for i in range(2):
        entry = census[f"box-{i}"]
        assert entry["server_key"] == S1.key
        assert entry["probe"] == {**PROBE, "hostname": f"box{i}"}
        assert entry["pulls"] >= entry["results"] >= 0
        assert entry["peers"] and all(p.startswith("127.0.0.1:") for p in entry["peers"])
        assert "probe_conflict" not in entry
    assert sum(e["results"] for e in census.values()) == 5


def test_worker_without_probe_is_still_counted_and_conflicts_are_flagged(tmp_path):
    driver, thread = _driver(tmp_path)
    driver._record_pull({"worker_id": "w0", "server_host": S1.key}, ("10.0.0.5", 4000))
    driver._record_pull({"worker_id": "w0", "server_host": S1.key, "probe": {"a": 1}}, ("10.0.0.5", 4001))
    driver._record_pull({"worker_id": "w0", "server_host": S1.key, "probe": {"a": 2}}, None)
    driver._record_pull({"server_host": S1.key}, None)  # anonymous pull: not a worker
    driver._record_result("w0")
    driver._record_result(None)
    census = driver.worker_census
    assert set(census) == {"w0"}
    entry = census["w0"]
    assert entry["pulls"] == 3 and entry["results"] == 1
    assert entry["probe"] == {"a": 1} and entry["probe_conflict"] is True
    assert entry["peers"] == ["10.0.0.5:4000", "10.0.0.5:4001"]
    thread.join(timeout=0.1)


def test_probe_rides_on_the_first_pull_of_every_connection():
    sent = []

    class Sock:
        def close(self):
            pass

    loop = WorkerLoop("w", S1.key, FakeEpisodeRunner(), connect=Sock, probe={"p": 1})
    from openpi.conductor import protocol as proto

    def fake_send(sock, msg_type, payload):
        sent.append(payload)

    def fake_recv(sock):
        return proto.MSG_ASSIGN, {"none": True, "backoff_ms": 1}

    proto_send, proto_recv = proto.send_message, proto.recv_message
    proto.send_message, proto.recv_message = fake_send, fake_recv
    try:
        loop._pull(Sock())
        loop._pull(Sock())
        loop._probe_sent = False  # what run_forever does on a fresh connection
        loop._pull(Sock())
    finally:
        proto.send_message, proto.recv_message = proto_send, proto_recv
    assert [("probe" in p) for p in sent] == [True, False, True]
    assert all(p["worker_id"] == "w" and p["server_host"] == S1.key for p in sent)
