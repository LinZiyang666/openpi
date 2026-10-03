import hashlib
import json
import os
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from exp.offline_search.debug import schema
from exp.offline_search.debug.server.writer import BlockWriter, pack_arrays
from exp.offline_search.debug.server.replay_tape import compare_outputs, record_tape, replay_tape
from exp.offline_search.debug.tests.test_server_support import no_git, server_json, server_rows


@pytest.fixture
def scratch(tmp_path):
    return tmp_path


def item(i):
    return dict(decision_id="fixture:0:{}".format(i), rawkeys_sampled=False), {
        "state_wire": np.asarray([i, 1 / 3], np.float64),
        "img_third": np.full((3, 4, 3), i % 256, np.uint8),
        "served_wire": np.full((10, 7), i, np.float32)}


def test_identity_sampling_and_lossless_atomic_npz(scratch):
    uid = "arm:eval:4:19"
    ep = schema.episode_key(uid, 3)
    assert ep == hashlib.sha256(uid.encode()).hexdigest()[:24] + "_a3"
    assert schema.decision_id(ep, 8, 7) == ep + ":8:7"
    for seq in range(200):
        h = int(hashlib.sha256("campaign|{}|{}".format(uid, seq).encode()).hexdigest(), 16)
        assert schema.rawkeys_sampled("campaign", uid, seq) == (h % 16 == 0)
        assert schema.snapshot_sampled("campaign", uid, seq) == (seq == 0 or h % 16 == 1)
        assert schema.draws_sampled("campaign", uid, seq) == (h % 32 == 2)
    arrays = {"ints": np.array([2 ** 63 + 5], np.uint64), "state": np.array([np.nan, 1 / 3], ">f8"),
              "id": np.array(["α"], dtype="<U80")}
    path = scratch / "block.npz"
    schema.write_npz_block(path, arrays)
    data = schema.read_npz_block(path)
    assert data["ints"].tobytes() == arrays["ints"].tobytes()
    np.testing.assert_array_equal(data["state"], arrays["state"])
    assert data["state"].dtype.byteorder != ">"
    assert not path.with_name("block.npz.part").exists()
    with pytest.raises(ValueError):
        schema.write_npz_block(scratch / "bad.npz", {"bad": np.array([{}], object)})


def test_blocks_rawkeys_and_drain(scratch):
    writer = BlockWriter(scratch, queue_bytes=4096)
    for i in range(41):
        rec, arrays = item(i)
        writer.put(rec, arrays, {"raw_key_third": np.arange(6, dtype=np.float32)} if i % 7 == 0 else None,
                   prompt="same prompt" if i % 2 else "another prompt")
    writer.close()
    records = server_rows(scratch)
    assert [r["decision_id"] for r in records] == ["fixture:0:{}".format(i) for i in range(41)]
    blocks = list((scratch / "blocks").glob("*.npz"))
    assert sum(len(schema.read_npz_block(p)["decision_id"]) for p in blocks) == 41
    for p in blocks:
        data = schema.read_npz_block(p)
        assert len(data["decision_id"]) <= 16
        for key, value in data.items():
            assert not value.dtype.hasobject, key
        for i, did in enumerate(data["decision_id"]):
            index = int(did.rsplit(":", 1)[1])
            assert data["state_wire"][i].tobytes() == item(index)[1]["state_wire"].tobytes()
            assert data["served_wire"][i].tobytes() == item(index)[1]["served_wire"].tobytes()
    assert sum(len(schema.read_npz_block(p)["decision_id"]) for p in (scratch / "rawkeys").glob("*.npz")) == 6
    stats = server_json(scratch, "writer_stats")
    assert stats["accepted"] == stats["written"] == 41 and stats["drained"] and not stats["errors"]
    assert stats["queue_high_water_bytes"] <= 4096


def test_backpressure_waits_and_no_loss(scratch):
    gate, entered = threading.Event(), threading.Event()
    original = BlockWriter._publish
    def slow(self, records, stream):
        entered.set()
        assert gate.wait(5)
        return original(self, records, stream)
    with patch.object(BlockWriter, "_publish", slow):
        writer = BlockWriter(scratch, queue_bytes=500, flush_seconds=.01)
        writer.put(*item(0))
        assert entered.wait(2)
        writer.put(*item(1))
        done = threading.Event()
        producer = threading.Thread(target=lambda: (writer.put(*item(2)), done.set()))
        producer.start()
        assert not done.wait(.05)
        gate.set()
        producer.join(5)
        assert done.is_set()
        writer.flush()
        writer.close()
    assert writer.stats["written"] == 3 and writer.stats["queue_wait_ms"] >= 40


def test_explicit_io_failure_and_closed_writer(scratch):
    writer = BlockWriter(scratch)
    with patch("exp.offline_search.debug.server.writer.write_npz_block", side_effect=OSError("disk full")):
        writer.put(*item(0))
        writer.flush()
    writer.close()
    assert "disk full" in writer.error
    row, = server_rows(scratch)
    assert row["status"] == "error" and "disk full" in row["writer_error"]
    assert not writer.stats["drained"]
    with pytest.raises(RuntimeError):
        writer.put(*item(1))


def test_ragged_diag_shapes_split_and_uint64_missing_is_lossless(scratch):
    writer = BlockWriter(scratch)
    for i in range(3):
        rec, arrays = item(i)
        arrays["diag_rows"] = np.arange(i + 1, dtype=np.int64)
        writer.put(rec, arrays)
    writer.close()
    assert writer.stats["blocks"] == 3
    rows = [dict(record=dict(decision_id=str(i)), arrays={}, rawkeys={}) for i in range(2)]
    rows[0]["arrays"]["diag_large"] = np.array([2 ** 63 + 5], np.uint64)
    data = pack_arrays(rows)
    assert data["diag_large"][0, 0] == 2 ** 63 + 5
    np.testing.assert_array_equal(data["diag_large_available"], [True, False])


def test_coordinator_tape_comparison_checks_exact_dtype_shape_bytes(scratch):
    left, right = scratch / "off", scratch / "on"
    data = {"decision_id": np.array(["d0", "d1"], dtype="<U80"), "actions": np.zeros((2, 10, 7), np.float32)}
    schema.write_npz_block(left / "responses_0000.npz", data)
    schema.write_npz_block(right / "responses_0000.npz", data)
    assert compare_outputs(left, right)["decisions"] == 2
    data["actions"][1, 2, 3] = np.nextafter(np.float32(0), np.float32(1))
    schema.write_npz_block(right / "responses_0000.npz", data)
    with pytest.raises(ValueError, match="action bytes differ"):
        compare_outputs(left, right)


def test_unusual_prompt_and_invalid_identity_are_not_truncated():
    did, prompt = "invalid:" + "x" * 100, "task " + "λ" * 600
    data = pack_arrays([dict(record=dict(decision_id=did, status="error"), arrays={}, rawkeys={}, prompt=prompt)])
    assert data["decision_id"][0] == did and data["prompts"][0] == prompt


def test_coordinator_wire_tape_record_and_replay(scratch, monkeypatch):
    from exp.offline_search.harness import store
    images = np.arange(3 * 4 * 5 * 3, dtype=np.uint8).reshape(3, 4, 5, 3)
    states = np.arange(24, dtype=np.float64).reshape(3, 8) / 7
    episode = dict(start=0, end=3, task_id=4, init=8, task="move object")
    query = types.SimpleNamespace(tok_index=np.arange(3), episodes=[episode], raw_state=states,
                                  tok=lambda name: images if name == "img0" else images[:, ::-1])
    monkeypatch.setattr(store, "QueryCell", lambda root, cell: query)
    tape = scratch / "tape"
    assert record_tape(scratch, "fake_cache", tape) == dict(episodes=1, decisions=3)
    clients = []
    class Client:
        def __init__(self, host, port):
            clients.append(self)
            self.requests, self.ended = [], False
        def select_bundle(self, bundle):
            assert bundle == "default"
        def episode_start(self, **kwargs):
            assert kwargs["extra_metadata"]["task_id"] == 4
        def infer(self, obs):
            seq = len(self.requests)
            np.testing.assert_array_equal(obs["observation/image"], images[seq])
            np.testing.assert_array_equal(obs["observation/wrist_image"], images[seq, ::-1])
            assert obs["observation/state"].tobytes() == states[seq].tobytes()
            assert obs["__extra__"] == dict(decision_id=seq, executed_steps=5)
            self.requests.append(obs)
            return {"actions": np.full((10, 7), seq, np.float32),
                    "__debug__": dict(decision_id=obs["__debug__"]["decision_id"], status="available")}
        def episode_end(self, success):
            assert not success
            self.ended = True
    module = types.ModuleType("openpi_client.websocket_client_policy")
    module.WebsocketClientPolicy = Client
    monkeypatch.setitem(sys.modules, "openpi_client.websocket_client_policy", module)
    first = replay_tape(tape, scratch / "off", expected_echo=False)
    second = replay_tape(tape, scratch / "on", expected_echo=True)
    assert first == second and first["decisions"] == 3
    assert all(client.ended for client in clients)
    assert compare_outputs(scratch / "off", scratch / "on")["PASS"]
