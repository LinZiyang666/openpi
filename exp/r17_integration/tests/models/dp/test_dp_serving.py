import asyncio
from dataclasses import asdict
import json
import socket

import numpy as np
import pytest
import torch
import yaml

from exp.r17_integration.common.provider import artifact_binding
from exp.r17_integration.common.serving import make_bundle_validator
from exp.r17_integration.models.dp.provider import make_connection_factory, register_provider
from exp.r17_integration.models.dp.serve import plain_config
from exp.r17_integration.models.dp.wire_policy import WirePolicy
from exp.r17_integration.common.config import build_shared_storage
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID
from openpi.serving.batching_core import BatchingCore
from openpi.serving import websocket_policy_server as registry_ws
from exp.r17_integration.common import websocket_policy_server as ws
from exp.r17_integration.tests.models.dp.conftest import observation, episode_metadata


@pytest.fixture
def registered(make_adapter, monkeypatch):
    from exp.r17_integration.common import provider as registry
    monkeypatch.setattr(registry, "_providers", {})
    monkeypatch.setattr(registry, "_traces", {})
    adapter = make_adapter()
    return adapter, register_provider(adapter)


def add_entry(adapter, storage, config, req_context):
    s1 = adapter.run_stage1(adapter.prepare(observation(), req_context))
    entry = CacheEntry("p4_library", CheckpointID.CP1,
                       {k: torch.from_numpy(v) for k, v in adapter.retrieval_fields(s1).items()},
                       CachePayload(torch.full((16, 4), 0.2), task_key="reach"), step_idx=0)
    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    backend = InMemoryBackend(dict(config.backend.vector_dims))
    backend.insert(entry)
    backend.artifact_meta = {"r17_binding": artifact_binding(config)}
    backend.freeze()
    storage._backend = backend
    return entry


@pytest.mark.parametrize("commit", [10, 15])
def test_stock_hit_miss_actual_cursor_tail_debug_no_extra_calls(registered, req_context, commit):
    adapter, provider = registered
    config = plain_config(provider, num_steps=2, commit_horizon=commit)
    storage = build_shared_storage(config)
    captures = []
    factory = make_connection_factory(adapter, provider=provider, default_config=config,
                                      default_storage=storage, capture_sink=captures.append)
    p = factory()
    p.on_task_begin()
    p.on_episode_start(task_id=0, extra_metadata=episode_metadata())
    obs = {**observation(), "__debug__": {"trace": "do not preprocess"}, "__request__": {"explicit_seed": 12}}
    out = p.infer(obs)
    assert out["__hit_meta__"]["stage_calls"] == (1, 1, 2) and out["actions"].shape == (commit, 4)
    assert len(captures[0].native_snapshots) == 1
    native_capture = adapter.make_trace_adapter().capture(captures[0])
    assert "snapshot_rng_state" in native_capture["arrays"]
    for seq in range(1, commit//5):
        tail = p.infer(dict(obs, __execution__={"decision_seq": seq-1, "executed_count": 5}))
        assert tail["__hit_meta__"]["stage_calls"] == (0, 0, 0)
        assert tail["__hit_meta__"]["cursor"] == seq*5
        assert tail["actions"].tobytes() == out["actions"][seq*5:].tobytes()
        assert not captures[-1].native_snapshots
    p.on_episode_end(execution={"decision_seq": commit//5-1, "executed_count": 5})
    assert p._chunk is None and not p.orchestrator.key_builder.cached_data
    hitcfg = plain_config(provider, num_steps=2, commit_horizon=commit, retrieval=True)
    hitstore = build_shared_storage(hitcfg)
    entry = add_entry(adapter, hitstore, hitcfg, req_context)
    hit = make_connection_factory(adapter, provider=provider, default_config=hitcfg, default_storage=hitstore)()
    hit.on_episode_start(extra_metadata=episode_metadata())
    answer = hit.infer(obs)
    assert answer["__hit_meta__"]["stage_calls"] == (1, 0, 0)
    expected = adapter.to_wire(entry.payload.action_chunk, None).actions[:commit]
    assert answer["actions"].tobytes() == expected.tobytes()


def test_two_connections_debug_on_off_byte_parity_with_real_batch_core(registered):
    adapter, provider = registered
    cfg = plain_config(provider, num_steps=2, commit_horizon=10)
    storage = build_shared_storage(cfg)
    with BatchingCore(adapter.make_stage_batcher(), device="cpu", max_batch_size=1, max_wait_ms=5) as core:
        records = []
        a = make_connection_factory(adapter, provider=provider, default_config=cfg, default_storage=storage,
                                    stage_core=core)()
        b = make_connection_factory(adapter, provider=provider, default_config=cfg, default_storage=storage,
                                    stage_core=core, capture_sink=records.append)()
        a.on_episode_start(extra_metadata=episode_metadata())
        b.on_episode_start(extra_metadata=episode_metadata())
        obs = dict(observation(), __request__={"explicit_seed": 23})
        # Sequential same bucket-size calls give exact capture parity.
        assert a.infer(obs)["actions"].tobytes() == b.infer(obs)["actions"].tobytes()
        assert a.adapter.adapter is not b.adapter.adapter
        assert a.adapter.model is b.adapter.model
        assert a.orchestrator is not b.orchestrator
        before = b._cursor
        tail = a.infer(dict(obs, __execution__={"decision_seq": 0, "executed_count": 3}))
        assert tail["__hit_meta__"]["stage_calls"] == (0, 0, 0) and b._cursor == before
        a.on_episode_end(execution={"decision_seq": 1, "executed_count": 2})
        assert b._chunk is not None


def test_real_WS_K100_2_1_yaml_switch_reject_and_episode_isolation(registered, monkeypatch, tmp_path):
    import websockets.asyncio.client
    from openpi_client import msgpack_numpy
    adapter, provider = registered
    cfg = plain_config(provider, num_steps=100, commit_horizon=15)
    storage = build_shared_storage(cfg)
    records, created = [], []
    factory = make_connection_factory(adapter, provider=provider, default_config=cfg,
                                      default_storage=storage, capture_sink=records.append)
    def track(base, bid):
        p = factory(base, bid)
        created.append(p)
        return p
    monkeypatch.setattr(registry_ws, "_bundles", {})
    monkeypatch.setattr(registry_ws, "_current_bundle", None)
    monkeypatch.setattr(registry_ws, "_bundle_version", 0)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    async def run():
        ready = asyncio.Event()
        base = WirePolicy(adapter)
        server = ws.WebsocketPolicyServer(base, host="127.0.0.1", port=port, concurrent=True,
                    connection_policy_factory=track, metadata=base.get_metadata(), ready_callback=ready.set,
                    bundle_validator=make_bundle_validator(provider))
        task = asyncio.create_task(server.run(stop_on_request=True))
        packer = msgpack_numpy.Packer()
        async def call(c, payload):
            await c.send(packer.pack(payload))
            value = await asyncio.wait_for(c.recv(), 15)
            assert isinstance(value, bytes), value
            return msgpack_numpy.unpackb(value)
        await asyncio.wait_for(ready.wait(), 5)
        try:
            async with websockets.asyncio.client.connect(f"ws://127.0.0.1:{port}", compression=None) as c1, \
                       websockets.asyncio.client.connect(f"ws://127.0.0.1:{port}", compression=None) as c2:
                assert msgpack_numpy.unpackb(await c1.recv())["model_id"] == adapter.metadata().model_id
                await c2.recv()
                for k in (100, 2, 1):
                    raw = asdict(plain_config(provider, num_steps=k, commit_horizon=5 if k == 1 else 15,
                        allow_lossy_sampler=k == 1,
                        sampler_study_label="P4_WS_K1_plain_step_reduction" if k == 1 else None))
                    ack = await call(c1, {"__ctrl__": "load_cache_config", "bundle_id": f"k{k}.v1",
                                          "yaml_content": yaml.safe_dump(raw)})
                    assert ack["__ack__"] == "load_cache_config", ack
                for c, bid in ((c1, "k100.v1"), (c2, "k2.v1")):
                    assert (await call(c, {"__ctrl__": "select_bundle", "bundle_id": bid}))["__ack__"] == "select_bundle"
                    assert (await call(c, {"__ctrl__": "episode_start", "__episode_id__": 0,
                                           "__extra__": episode_metadata()}))["__ack__"] == "episode_start"
                obs = dict(observation(), __request__={"explicit_seed": 18})
                a, b = await asyncio.gather(call(c1, obs), call(c2, obs))
                assert a["__hit_meta__"]["actual_nfe"] == 100 and b["__hit_meta__"]["actual_nfe"] == 2
                tail = await call(c1, dict(obs, __execution__={"decision_seq": 0, "executed_count": 5}))
                assert tail["__hit_meta__"]["stage_calls"] == [0, 0, 0]
                assert created[-1]._cursor == 0
                old_version = registry_ws._bundle_version
                bad = asdict(plain_config(provider, num_steps=2))
                bad["adapter_binding"]["norm_identity"] = "foreign"
                ack = await call(c1, {"__ctrl__": "load_cache_config", "bundle_id": "bad",
                                     "yaml_content": yaml.safe_dump(bad)})
                assert ack["__ack__"] == "error" and registry_ws._bundle_version == old_version
                unlabelled = asdict(plain_config(provider, num_steps=2))
                unlabelled["adapter_binding"].update(num_steps=1, sampler_identity=provider.sampler_for_steps(1).identity)
                forbidden = asdict(plain_config(provider, num_steps=1, allow_lossy_sampler=True,
                    sampler_study_label="forbidden_K1_cache_MISS"))
                forbidden["checkpoints"] = asdict(plain_config(provider, num_steps=2, retrieval=True))["checkpoints"]
                forbidden["checkpoints"]["cp1"]["judge"]["type"] = "always_miss"
                for name, document in (("unlabelled_k1", unlabelled), ("k1_cache_miss", forbidden)):
                    error = await call(c1, {"__ctrl__": "load_cache_config", "bundle_id": name,
                        "yaml_content": yaml.safe_dump(document)})
                    assert error["__ack__"] == "error" and registry_ws._bundle_version == old_version
                    assert registry_ws.get_current_cache_bundle(name) is None
                assert (await call(c1, {"__ctrl__": "select_bundle", "bundle_id": "unknown"}))["__ack__"] == "error"
                assert (await call(c1, {"__ctrl__": "episode_end",
                             "__execution__": {"decision_seq": 1, "executed_count": 5}}))["__ack__"] == "episode_end"
                assert (await call(c1, {"__ctrl__": "select_bundle", "bundle_id": "k1.v1"}))["__ack__"] == "select_bundle"
                assert (await call(c1, {"__ctrl__": "episode_start", "__episode_id__": 1,
                                         "__extra__": episode_metadata()}))["__ack__"] == "episode_start"
                one = await call(c1, obs)
                assert one["__hit_meta__"]["actual_nfe"] == 1
                # Same ID selection is idempotent; a new generation builds a wrapper.
                count = len(created)
                await call(c1, {"__ctrl__": "select_bundle", "bundle_id": "k1.v1"})
                assert len(created) == count
                assert all(record.actual_nfe in (0, 1, 2, 100) for record in records)
        finally:
            server.request_stop()
            await asyncio.wait_for(task, 5)
        assert all(p._chunk is None for p in created)
    asyncio.run(run())


def test_P6_DecisionSink_factory_closed_native_capture_and_execution_ack(registered, tmp_path):
    from exp.r17_integration.debug.capture_bridge import CaptureBridge, DecisionSink
    from exp.r17_integration.debug.reader_profiles import CaptureReader
    from exp.r17_integration.debug.validate import model_binding
    adapter, provider = registered
    cfg = plain_config(provider, num_steps=2, commit_horizon=10)
    storage = build_shared_storage(cfg)
    binding = model_binding(adapter.metadata(), adapter.metadata().allowed_benchmark_protocols[0])
    bridge = CaptureBridge(tmp_path / "debug", enabled=True, binding=binding)
    sink = DecisionSink(bridge, {"task_uid": "p4_join", "task_data_hash": "fake", "split": "library"})
    policy = make_connection_factory(adapter, provider=provider, default_config=cfg,
                                     default_storage=storage, capture_sink=sink)()
    policy.on_episode_start(extra_metadata={**episode_metadata("reach-v3"), "attempt": 1})
    obs = dict(observation(), __request__={"explicit_seed": 27})
    baseline = make_connection_factory(adapter, provider=provider, default_config=cfg, default_storage=storage)()
    baseline.on_episode_start(extra_metadata={**episode_metadata("reach-v3"), "attempt": 1})
    assert policy.infer(obs)["actions"].tobytes() == baseline.infer(obs)["actions"].tobytes()
    policy.acknowledge_execution({"decision_seq": 0, "executed_count": 3, "terminal": True})
    stats = bridge.close()
    assert stats["drained"] and not stats["errors"]
    reader = CaptureReader(tmp_path / "debug")
    row, arrays, keys = list(reader.decisions())[0]
    assert row["native_snapshots"][0]["native_timestep"] == 49
    assert arrays["snapshot_1_rng"].dtype == np.uint8
    assert arrays["native_grid"].dtype == np.int64
    assert arrays["native%2Fconditioning"].shape[0] == 1
    acknowledgments = [json.loads(line) for line in (tmp_path / "debug" / "execution.jsonl").read_text().splitlines()]
    assert acknowledgments[0]["executed_count"] == 3 and acknowledgments[0]["cursor_end"] == 3
