"""Actual stock WS + P1 staged policy + P6 writer, CPU fake model only."""
import asyncio
from dataclasses import asdict
import uuid

import numpy as np
import yaml

from exp.r17_integration.common.fake_adapter import fake_provider
from exp.r17_integration.common.provider import binding_for
from exp.r17_integration.common.serving import make_bundle_validator, make_connection_factory
from exp.r17_integration.debug.capture_bridge import CaptureBridge, CaptureFactory, DecisionSink
from exp.r17_integration.debug.reader_profiles import CaptureReader
from exp.r17_integration.debug.validate import model_binding
from exp.r17_integration.ops.serve_registry import PortReservation
from exp.r17_integration.tests.debug.fixtures import record


def config(provider, *, k=4, commit=10):
    from exp.r17_integration.common.config import (BackendConfig, CacheConfig, CheckpointConfig,
                                    GateConfig, JudgeConfig, KeyBuilderConfig, KeysConfig,
                                    SearchStrategyConfig, WritePolicyConfig)
    dims = dict(provider.model_spec.feature_spec.fields)
    keys = KeysConfig()
    for name, value in vars(keys).items():
        value.enabled = name in dims
    return CacheConfig(enabled=True, diagnostic_context=True, keys=keys, key_builder=KeyBuilderConfig(type="adapter_native"),
                       backend=BackendConfig(type="in_memory", vector_dims=dims),
                       write_policy=WritePolicyConfig(type="never"),
                       checkpoints={"cp1": CheckpointConfig(gate=GateConfig(type="always_search"),
                            judge=JudgeConfig(type="always_hit"), search_strategy=SearchStrategyConfig(type="weighted_rrf_knn", top_k=1))},
                       adapter_binding=binding_for(provider, "fake_mw.v1", num_steps=k, commit_horizon=commit))


def test_real_websocket_two_clients_debug_parity_and_capture(tmp_path, monkeypatch):
    import websockets.asyncio.client
    from exp.r17_integration.common.config import build_shared_storage
    from openpi.serving import websocket_policy_server as ws
    from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
    from openpi_client import msgpack_numpy
    provider = fake_provider("p6.fixture." + uuid.uuid4().hex)
    cfg = config(provider)
    storage = build_shared_storage(cfg)
    bridge = CaptureBridge(tmp_path / "capture", enabled=True, binding=model_binding(provider.model_spec, "fake_mw.v1"))
    sink = DecisionSink(bridge, lambda request: dict(record(), benchmark_protocol=request.benchmark_protocol,
                        task_uid=f"p6:{request.task_name}:{request.original_init_idx}"))
    plain = make_connection_factory(provider, cfg, storage)
    debug = make_connection_factory(provider, cfg, storage, capture_sink=sink)
    policies = []
    def factory(base, bundle_id):
        policy = (debug if bundle_id == "debug.v1" else plain)(base, bundle_id)
        policies.append(policy)
        return policy
    monkeypatch.setattr(ws, "_bundles", {})
    monkeypatch.setattr(ws, "_current_bundle", None)
    monkeypatch.setattr(ws, "_bundle_version", 0)
    reservation = PortReservation()
    port = reservation.port
    # Stock WS has no inherited-FD option; release immediately before its bind.
    reservation.close()
    async def run():
        ready = asyncio.Event()
        server = WebsocketPolicyServer(object(), host="127.0.0.1", port=port, concurrent=True,
            connection_policy_factory=factory, metadata={"model": "p6_cpu_fake"}, ready_callback=ready.set,
            bundle_validator=make_bundle_validator(provider))
        task = asyncio.create_task(server.run(stop_on_request=True))
        await asyncio.wait_for(ready.wait(), 5)
        packer = msgpack_numpy.Packer()
        async def call(client, value):
            await client.send(packer.pack(value))
            response = await asyncio.wait_for(client.recv(), 5)
            assert isinstance(response, bytes), response
            return msgpack_numpy.unpackb(response)
        try:
            async with websockets.asyncio.client.connect(f"ws://127.0.0.1:{port}", compression=None) as a, \
                       websockets.asyncio.client.connect(f"ws://127.0.0.1:{port}", compression=None) as b:
                await a.recv(); await b.recv()
                for name in ("plain.v1", "debug.v1"):
                    ack = await call(a, {"__ctrl__": "load_cache_config", "bundle_id": name,
                                         "yaml_content": yaml.safe_dump(asdict(cfg))})
                    assert ack["__ack__"] == "load_cache_config", ack
                await call(a, {"__ctrl__": "select_bundle", "bundle_id": "plain.v1"})
                await call(b, {"__ctrl__": "select_bundle", "bundle_id": "debug.v1"})
                obs = {"state": np.arange(4, dtype=np.float32), "__request__": dict(
                    campaign="cpu-fixture", benchmark_protocol="fake_mw.v1", task_name="reach-v3", original_init_idx=10,
                    decision_seq=0, arm="full", attempt=1, bundle_generation="generation1")}
                direct = plain().infer(obs)
                ra, rb = await asyncio.gather(call(a, obs), call(b, obs))
                assert direct["actions"].tobytes() == ra["actions"].tobytes() == rb["actions"].tobytes()
                stable = ("source", "stage_calls", "actual_nfe", "cursor", "commit_horizon", "sampler_identity")
                assert {k:ra["__hit_meta__"][k] for k in stable} == {k:rb["__hit_meta__"][k] for k in stable}
                ack = {"decision_seq": 0, "executed_count": 3, "cursor": 3}
                tail_obs = dict(obs, __execution__=ack)
                tail_obs["__request__"] = dict(obs["__request__"], decision_seq=1)
                ta, tb = await asyncio.gather(call(a, tail_obs), call(b, tail_obs))
                assert ta["actions"].tobytes() == tb["actions"].tobytes()
                assert ta["__hit_meta__"]["stage_calls"] == [0, 0, 0]
                assert policies[0].adapter.counts == policies[1].adapter.counts
                for client in (a,b):
                    response=await call(client,{"__ctrl__":"episode_end","__success__":True,
                        "__execution__":{"decision_seq":1,"executed_count":2,"cursor":5,"terminal":True}})
                    assert response["__ack__"]=="episode_end"
        finally:
            server.request_stop()
            await asyncio.wait_for(task, 5)
    try:
        asyncio.run(run())
    finally:
        bridge.close()
    captures = list(CaptureReader(tmp_path / "capture").decisions())
    assert len(captures) == 2 and captures[1][0]["actual_nfe"] == 0
    assert len(captures[0][0]["native_snapshots"]) == 3
    assert captures[0][0]["executed_count"] == 3 and captures[0][0]["cursor_end"] == 3
    assert captures[1][0]["execution_status"] == "acknowledged"
    assert captures[1][0]["executed_count"]==2 and captures[1][0]["terminal"] is True


def test_same_bundle_concurrent_capture_and_execution_are_isolated(tmp_path):
    from exp.r17_integration.common.config import build_shared_storage
    provider = fake_provider("p6.concurrent." + uuid.uuid4().hex)
    cfg = config(provider)
    storage = build_shared_storage(cfg)
    resolver = lambda request: dict(record(init=request.original_init_idx), benchmark_protocol=request.benchmark_protocol,
                                    task_uid=f"p6:reach:{request.original_init_idx}")
    factory = CaptureFactory(provider, cfg, storage, directory=tmp_path / "capture", enabled=True, episode_metadata=resolver)
    plain = make_connection_factory(provider, cfg, storage)
    a, b = factory(), factory()
    def obs(init, seq=0, ack=None):
        result = {"state": np.arange(4, dtype=np.float32), "__request__": dict(campaign="cpu-fixture",
            benchmark_protocol="fake_mw.v1", task_name="reach-v3", original_init_idx=init, decision_seq=seq,
            arm="full", attempt=1, bundle_generation="generation1")}
        if ack is not None:
            result["__execution__"] = ack
        return result
    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(2) as pool:
            ra, rb = list(pool.map(lambda pair: pair[0].infer(obs(pair[1])), [(a, 10), (b, 11)]))
        assert ra["actions"].tobytes() == plain().infer(obs(10))["actions"].tobytes()
        assert rb["actions"].tobytes() == plain().infer(obs(11))["actions"].tobytes()
        ack = {"decision_seq": 0, "executed_count": 3}
        with ThreadPoolExecutor(2) as pool:
            tails = list(pool.map(lambda pair: pair[0].infer(obs(pair[1], 1, ack)), [(a, 10), (b, 11)]))
        assert all(r["__hit_meta__"]["stage_calls"] == (0, 0, 0) for r in tails)
        assert a.capture_sink is not b.capture_sink and len(factory.bridges) == 1
    finally:
        factory.close()
    rows = list(CaptureReader(next((tmp_path / "capture").iterdir())).decisions())
    initial = [r for r, arrays, keys in rows if r["decision_seq"] == 0]
    assert len(initial) == 2 and all(r["executed_count"] == 3 for r in initial)
