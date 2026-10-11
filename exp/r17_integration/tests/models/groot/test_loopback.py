"""The actual shared WebSocket control path with a CPU GR00T-MW backend."""
import asyncio
from dataclasses import asdict
import socket

import numpy as np
import yaml

from exp.r17_integration.common.serving import make_bundle_validator
from exp.r17_integration.models.groot.serve import make_connection_factory
from exp.r17_integration.common.config import build_shared_storage
from openpi.serving import websocket_policy_server as ws
from .test_shared_factory import config, published


def test_gr_mw_two_clients_hot_switch_and_bad_yaml_atomic(published, obs, monkeypatch):
    from openpi_client import msgpack_numpy
    import websockets.asyncio.client
    cfg = config(published, commit=10)
    factory = make_connection_factory(published, cfg, build_shared_storage(cfg))
    created = []

    def tracked(base, bundle_id):
        policy = factory(base, bundle_id)
        created.append((bundle_id, policy))
        return policy

    monkeypatch.setattr(ws, "_bundles", {})
    monkeypatch.setattr(ws, "_current_bundle", None)
    monkeypatch.setattr(ws, "_bundle_version", 0)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    async def run():
        ready = asyncio.Event()
        from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
        server = WebsocketPolicyServer(object(), host="127.0.0.1", port=port, concurrent=True,
                metadata={"model": "groot-MW-CPU-stub"}, connection_policy_factory=tracked,
                bundle_validator=make_bundle_validator(published), ready_callback=ready.set)
        task = asyncio.create_task(server.run(stop_on_request=True))
        await asyncio.wait_for(ready.wait(), 5)
        packer = msgpack_numpy.Packer()

        async def call(client, payload):
            await client.send(packer.pack(payload))
            reply = await asyncio.wait_for(client.recv(), 5)
            assert isinstance(reply, bytes), reply
            return msgpack_numpy.unpackb(reply)

        try:
            async with websockets.asyncio.client.connect(f"ws://127.0.0.1:{port}", compression=None) as first, websockets.asyncio.client.connect(f"ws://127.0.0.1:{port}", compression=None) as second:
                await first.recv()
                await second.recv()
                for bundle_id, commit in (("full-g1", 10), ("tail-g2", 15)):
                    ack = await call(first, {"__ctrl__": "load_cache_config", "bundle_id": bundle_id,
                                            "yaml_content": yaml.safe_dump(asdict(config(published, commit=commit)))})
                    assert ack["__ack__"] == "load_cache_config", ack
                assert (await call(first, {"__ctrl__": "select_bundle", "bundle_id": "full-g1"}))["__ack__"] == "select_bundle"
                assert (await call(second, {"__ctrl__": "select_bundle", "bundle_id": "tail-g2"}))["__ack__"] == "select_bundle"
                count = len(created)
                await call(first, {"__ctrl__": "select_bundle", "bundle_id": "full-g1"})
                assert len(created) == count
                episode = {"__ctrl__": "episode_start", "__task__": "reach-v3", "__extra__": {
                    "task_name": "reach-v3", "original_init_idx": 0, "ack_protocol": "r17.execution.v1"}}
                await asyncio.gather(call(first, episode), call(second, episode))
                a, b = await asyncio.gather(call(first, obs), call(second, obs))
                assert a["__hit_meta__"]["actual_nfe"] == b["__hit_meta__"]["actual_nfe"] == 4
                np.testing.assert_array_equal(a["actions"], b["actions"][:10])
                old_version = ws._bundle_version
                bad = asdict(config(published))
                bad["adapter_binding"]["norm_identity"] = "LIBERO-wrong"
                ack = await call(first, {"__ctrl__": "load_cache_config", "bundle_id": "full-g1", "yaml_content": yaml.safe_dump(bad)})
                assert ack["__ack__"] == "error" and ws._bundle_version == old_version
                tail = await call(first, dict(obs, __execution__={"decision_seq": 0, "executed_count": 3, "cursor": 3}))
                assert tail["__hit_meta__"]["stage_calls"] == [0, 0, 0]
                np.testing.assert_array_equal(tail["actions"], a["actions"][3:])
                await call(first, {"__ctrl__": "episode_end", "success": False,
                                   "__execution__": {"decision_seq": 1, "executed_count": 2}})
                ack = await call(first, {"__ctrl__": "select_bundle", "bundle_id": "tail-g2"})
                assert ack["__ack__"] == "select_bundle"
                assert created[-1][1].adapter.runner is not created[-2][1].adapter.runner
                assert (await call(first, {"__ctrl__": "select_bundle", "bundle_id": "missing"}))["__ack__"] == "error"
            for _ in range(50):
                if all(policy._chunk is None for _, policy in created):
                    break
                await asyncio.sleep(.01)
            assert all(policy._chunk is None for _, policy in created)
        finally:
            server.request_stop()
            await asyncio.wait_for(task, 5)

    asyncio.run(run())
