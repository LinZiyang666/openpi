"""Drive the production websocket dispatcher with queued frames, no listener."""
import asyncio
import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from openpi_client import msgpack_numpy

from openpi.serving import websocket_policy_server as wps
from openpi.serving.batching_coordinator import BatchingCoordinator, Stage3MissPayload
from openpi.serving.split_runtime import DecisionPolicy
from openpi.serving.split_service import RemoteCoordinator, StageService
from tests.serving.decouple.stubs import PiModel
from tests.serving.decouple.test_stages import Harness
from tests.serving.test_websocket_policy_server import _FakeConnection


def test_yaml_hot_switch_pins_existing_connection_and_updates_new_bind(monkeypatch, tmp_path):
    import openpi.cache.config as cc
    group = json.loads(Path("/home/weiland/trace_runs/os_closed_loop/r16_conductor_dev_pi05_k1_v1/group.json").read_text())
    config = yaml.safe_load(Path(group["entries"][0]["row"]["yaml"]).read_text())
    files = []
    for k in (1, 2):
        config["miss"]["num_steps"] = k
        config["miss"]["evidence_dir"] = str(tmp_path / f"evidence{k}")
        path = tmp_path / f"k{k}.yaml"
        path.write_text(yaml.safe_dump(config))
        files.append(str(path))
    monkeypatch.setattr(cc, "build_shared_storage", lambda config: object())
    monkeypatch.setattr(wps, "_bundles", {})
    monkeypatch.setattr(wps, "_current_bundle", None)
    monkeypatch.setattr(wps, "_bundle_version", 0)
    model = PiModel()
    core = BatchingCoordinator(model, max_batch_size=1, max_wait_ms=0)
    core.start()
    harness = Harness(StageService(model, coordinator=core))
    remote = RemoteCoordinator(harness.client)
    class Policy:
        def __init__(self, k):
            self.k = k
        def infer(self, obs):
            first = remote.submit_to_stage(1, "b", {"state": obs["state"]})
            second = remote.submit_to_stage(2, "b", first)
            action = remote.submit_to_stage(3, "b", Stage3MissPayload(second, torch.zeros(10, 32), self.k))
            return {"actions": action.action_chunk[0].numpy(), "k": self.k}
    def factory(base, bundle_id):
        bundle = wps.get_current_cache_bundle(bundle_id) or wps.get_current_cache_bundle()
        return DecisionPolicy(Policy(bundle.cache_config.miss.num_steps), harness.client)
    server = wps.WebsocketPolicyServer(None, concurrent=True, connection_policy_factory=factory)
    packer = msgpack_numpy.Packer()
    obs = {"state": np.ones(32, np.float32)}
    first = _FakeConnection([packer.pack(x) for x in (
        {"__ctrl__": "load_cache_config", "yaml_path": files[0], "bundle_id": "b"},
        {"__ctrl__": "select_bundle", "bundle_id": "b"}, obs,
        {"__ctrl__": "load_cache_config", "yaml_path": files[1], "bundle_id": "b"}, obs,
        {"__ctrl__": "select_bundle", "bundle_id": "default"},
        {"__ctrl__": "select_bundle", "bundle_id": "b"}, obs,
    )])
    second = _FakeConnection([packer.pack(x) for x in ({"__ctrl__": "select_bundle", "bundle_id": "b"}, obs)])
    try:
        asyncio.run(server._handler(first))
        asyncio.run(server._handler(second))
        outputs = [msgpack_numpy.unpackb(x) for x in first.sent if isinstance(x, bytes)]
        actions = [x for x in outputs if "actions" in x]
        assert [x["k"] for x in actions] == [1, 1, 2]
        assert np.array_equal(actions[0]["actions"], actions[1]["actions"])
        assert np.array_equal(actions[2]["actions"], actions[0]["actions"] * 2)
        other = [msgpack_numpy.unpackb(x) for x in second.sent if isinstance(x, bytes)]
        assert [x["k"] for x in other if "actions" in x] == [2]
    finally:
        harness.close()
        core.stop()
