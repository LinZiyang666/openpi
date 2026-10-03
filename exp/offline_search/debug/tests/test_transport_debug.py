"""Real loopback sockets, durability, overlap ACK loss, replay and journal fence."""
import hashlib
import io
import json
from pathlib import Path
import socket
import threading

import pytest

from exp.offline_search.debug.transport.collect import certify_tree, replay_spills, verify_arm
from exp.offline_search.debug.transport.dispatch_fence import DispatchFence
from exp.offline_search.debug.transport.protocol import filename
from exp.offline_search.debug.transport.receiver import Receiver, Store
from exp.offline_search.debug.transport.sink import StreamSink
from exp.offline_search.debug.tests.test_client_capture import make_episode


def run_root(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "arms.json").write_text('[{"arm":"A"}]')
    journal = root / "runs/A/client/journal.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_text(json.dumps(dict(task_uid="A:eval:0:0", attempt=1, accepted=True, status="failed", success=False)) + "\n")
    return root


class LocalReceiver:
    def __init__(self, root, **kw):
        try:
            self.server = Receiver(("127.0.0.1", 0), Store(root), "8" * 64, **kw)
        except PermissionError:
            pytest.skip("sandbox denies AF_INET sockets; run on coordinator for loopback admission")
        assert not 23100 <= self.server.server_address[1] <= 23199
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return "127.0.0.1:" + str(self.server.server_address[1])

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(3)
        assert not self.thread.is_alive()


@pytest.mark.parametrize("drop", [False, True])
def test_offset_ack_receipts_and_lost_ack(tmp_path, drop):
    root = run_root(tmp_path)
    with LocalReceiver(root, drop_once=drop) as address:
        def factory(directory, uid, attempt):
            return StreamSink(address, "8" * 64, root.name, "A", uid, attempt, directory, timeout=.2, close_seconds=3)
        client, _, _, _ = make_episode(tmp_path / "local", n=12, sink_factory=factory)
        assert not client.errors and not client.sink.spilled
    assert not (tmp_path / "local").exists()
    receipt = root / "runs/A/debug/receipts" / client.identity["episode_key"] / "complete.json"
    record = json.loads(receipt.read_text())
    assert record["status"] == "complete" and record["dispatch_gen"] == 3
    assert record["files"]["episode.json"]["bytes"] > 0
    verified = verify_arm(root, "A", 1, require_server=False)
    assert verified["accepted_attempts"] == 1 and verified["decisions"] == 3
    with pytest.raises(ValueError, match="unverified/missing"):
        verify_arm(root, "A", 1)


def test_bounded_spill_replay_is_idempotent(tmp_path, monkeypatch):
    root = run_root(tmp_path)
    def unavailable(self, event):
        raise OSError("injected offline receiver")
    monkeypatch.setattr(StreamSink, "_send", unavailable)
    def factory(directory, uid, attempt):
        return StreamSink("127.0.0.1:24099", "8" * 64, root.name, "A", uid, attempt, directory,
                          queue_bytes=1024, fail_seconds=.01, timeout=.02, close_seconds=.1)
    client, _, _, _ = make_episode(tmp_path / "spill", n=7, sink_factory=factory)
    assert not client.errors and client.sink.spilled
    assert client.sink.stats["queue_high_water"] <= 1024
    path = tmp_path / "spill" / client.identity["episode_key"] / "stream.osdebugspill"
    assert path.is_file() and not path.with_name(path.name + ".part").exists()
    assert replay_spills(root, "A", tmp_path / "spill") == 1
    assert replay_spills(root, "A", tmp_path / "spill") == 1
    assert verify_arm(root, "A", 1, False)["verified"]
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        replay_spills(root, "A", tmp_path / "spill")


def test_wire_protocol_retry_without_network(tmp_path, monkeypatch):
    """Use the production pack/receive and sender, with AF_INET replaced only."""
    from exp.offline_search.debug.transport.protocol import pack, receive
    root = run_root(tmp_path)
    store = Store(root)
    fault = [True]

    class WireSocket:
        def settimeout(self, timeout):
            pass

        def sendall(self, raw):
            h, body = receive(io.BytesIO(raw).read)
            assert h.pop("token") == "8" * 64
            result = store.apply(h, body)
            self.response = io.BytesIO(pack(result))
            if h["op"] == "data" and fault[0]:
                fault[0] = False
                self.response = io.BytesIO(b"")  # durable ACK lost

        def recv(self, n):
            return self.response.read(n)

        def close(self):
            pass

    monkeypatch.setattr("exp.offline_search.debug.transport.sink.socket.create_connection", lambda *a, **kw: WireSocket())
    def factory(directory, uid, attempt):
        return StreamSink("127.0.0.1:24099", "8" * 64, root.name, "A", uid, attempt, directory, timeout=.1, close_seconds=3)
    client, _, _, _ = make_episode(tmp_path / "local", n=7, sink_factory=factory)
    assert not client.errors and not client.sink.spilled and client.sink.stats["reconnects"] >= 2
    assert verify_arm(root, "A", 1, False)["verified"]


def test_file_mode_certification_checksum_and_journal(tmp_path):
    root = run_root(tmp_path)
    client, _, _, _ = make_episode(root / "runs/A/debug/client", n=5)
    certify_tree(root, "A")
    assert verify_arm(root, "A", 1, False)["verified"]
    with pytest.raises(ValueError, match="count mismatch"):
        verify_arm(root, "A", 2, False)
    ep = root / "runs/A/debug/client" / client.identity["episode_key"] / "episode.json"
    ep.write_bytes(ep.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        verify_arm(root, "A", 1, False)


def test_paths_offsets_duplicates_and_attempt_fence(tmp_path):
    root = run_root(tmp_path)
    store = Store(root)
    from exp.offline_search.debug.transport.protocol import attempt_key
    h = dict(run=root.name, arm="A", uid="A:eval:0:0", attempt=1, key=attempt_key("A:eval:0:0", 1))
    for name in ("../events.jsonl", "controls.jsonl", "controls_0.npz", "step_000000.npz", "snap_-1.npz", "a/episode.json"):
        with pytest.raises(ValueError):
            filename(name)
    data = dict(h, op="data", file="events.jsonl", offset=0)
    assert store.apply(data, b"abc")["offset"] == 3
    assert store.apply(data, b"abc")["offset"] == 3
    with pytest.raises(ValueError, match="conflicting"):
        store.apply(data, b"bad")
    with pytest.raises(ValueError, match="gap"):
        store.apply(dict(data, offset=4), b"x")
    fence = DispatchFence(tmp_path / "fence")
    assert fence.reserve(h["uid"], 1) == 1
    assert DispatchFence(tmp_path / "fence").reserve(h["uid"], 1) == 2
    assert DispatchFence(tmp_path / "fence").reserve(h["uid"], 5) == 5
    legacy = DispatchFence(tmp_path / "legacy", root / "runs/A/client/journal.jsonl")
    assert legacy.reserve(h["uid"], 1) == 100001


def test_server_join_and_orphans(tmp_path):
    root = run_root(tmp_path)
    client, _, _, _ = make_episode(root / "runs/A/debug/client", n=5)
    certify_tree(root, "A")
    ep = root / "runs/A/debug/client" / client.identity["episode_key"]
    decisions = [r for r in map(json.loads, (ep / "events.jsonl").read_text().splitlines()) if r["ev"] == "decision"]
    server = root / "runs/A/debug/server_24090"
    server.mkdir()
    rows = [dict(decision_id=r["decision_id"], **client.identity) for r in decisions]
    (server / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert verify_arm(root, "A", 1)["server_join_verified"]
    with (server / "decisions.jsonl").open("a") as f:
        f.write(json.dumps(dict(client.identity, decision_id="orphan")) + "\n")
    with pytest.raises(ValueError, match="orphan"):
        verify_arm(root, "A", 1)


def test_certified_spill_cleanup_preserves_new_data_and_fence(tmp_path):
    from exp.offline_search.debug.transport.cleanup import cleanup
    from exp.offline_search.debug.transport.receiver import digest
    key = "f" * 24 + "_a1"
    root = tmp_path / "spill"
    directory = root / key
    directory.mkdir(parents=True)
    file = directory / "stream.osdebugspill"
    file.write_bytes(b"archived")
    newer = directory / "stream.osdebugspill.json"
    newer.write_bytes(b"new bytes")
    proof = dict(files=[dict(path=key + "/stream.osdebugspill", bytes=file.stat().st_size, sha256=digest(file)),
                        dict(path=key + "/stream.osdebugspill.json", bytes=3, sha256="0" * 64)])
    with pytest.raises(ValueError, match="changed"):
        cleanup(root, proof)
    assert file.exists()  # all hashes checked before the first removal
    proof["files"].pop()
    assert cleanup(root, proof)["removed"] == 1
    assert newer.read_bytes() == b"new bytes" and directory.exists()
    with pytest.raises(ValueError, match="allowlist"):
        cleanup(root, dict(files=[dict(path=".osdebug_dispatch/state.json", bytes=0, sha256="0" * 64)]))
