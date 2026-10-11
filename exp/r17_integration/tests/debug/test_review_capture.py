"""Review A5/A10 and B9: deployed bridge identity, sampler and overhead."""
from dataclasses import asdict, replace
import uuid

import numpy as np
import pytest

from exp.r17_integration.common.contracts import DecisionCapture, RequestContext
from exp.r17_integration.common.fake_adapter import fake_provider
from exp.r17_integration.debug.capture_bridge import CaptureBridge, DecisionSink
from exp.r17_integration.debug.reader_profiles import CaptureReader
from exp.r17_integration.tests.debug.fixtures import arrays, binding, record


def test_bridge_measures_preparation_and_marks_cost_as_capture(tmp_path):
    b = binding(); payload, keys = arrays(b)
    bridge = CaptureBridge(tmp_path, enabled=True, binding=b)
    bridge.capture(record(), payload, rawkeys=keys)
    bridge.close()
    row = next(CaptureReader(tmp_path).decisions())[0]
    assert row["capture_bridge_prepare_ms"] >= 0
    assert row["capture_enabled"] is True and row["pricing_eligible"] is False
    assert "host_wall" in row["capture_timing_scope"]


def test_same_uid_across_arms_has_distinct_block_decision_ids(tmp_path):
    b = binding(); payload, keys = arrays(b)
    bridge = CaptureBridge(tmp_path, enabled=True, binding=b)
    for arm in ("full", "retrieval"):
        bridge.capture(record(arm=arm), payload, rawkeys=keys)
    bridge.close()
    rows = list(CaptureReader(tmp_path).decisions())
    assert len({r[0]["decision_id"] for r in rows}) == 2
    from exp.offline_search.debug.schema import decision_id
    assert all(r['decision_id']==decision_id(r['episode_key'],r['dispatch_gen'],r['decision_seq']) for r,_,_ in rows)


def test_model_capture_refuses_conflicting_episode_provenance(tmp_path):
    b = binding(); bridge = CaptureBridge(tmp_path, enabled=True, binding=b)
    sink = DecisionSink(bridge, record())
    req = RequestContext("cpu-fixture", b["benchmark_protocol"], "reach-v3", 10, 0,
                         bundle_generation="other", arm="full", attempt=1)
    capture = DecisionCapture(req, "policy", 0, 0, (1,1,5), 5,
                              np.zeros((16,4)), np.zeros((5,4)))
    try:
        with pytest.raises(ValueError, match="identity|provenance"):
            sink(capture)
    finally:
        bridge.close()


def test_core_aware_server_capture_uses_episode_metadata_and_actual_sampler(tmp_path):
    from exp.r17_integration.debug.capture_bridge import make_capture_sink_factory
    from exp.r17_integration.tests.debug.test_framework_bridge import config
    from exp.r17_integration.common.serving import make_connection_factory
    from exp.r17_integration.common.config import build_shared_storage
    provider = fake_provider("p6.review."+uuid.uuid4().hex)
    cfg = config(provider, k=2)
    factory = make_capture_sink_factory(provider, tmp_path)
    try:
        core = make_connection_factory(provider, cfg, build_shared_storage(cfg))()
        core.capture_sink = factory(core)
        core.capture = replace(core.capture, enabled=True, snapshots=True)
        ep = dict(record(), benchmark_protocol="fake_mw.v1", original_init_idx=10)
        core.on_episode_start(task_id=0, init_state_id=10, extra_metadata=ep)
        obs = dict(state=np.arange(4,dtype=np.float32), __request__=dict(
            campaign="cpu-fixture", benchmark_protocol="fake_mw.v1", task_name="reach-v3", original_init_idx=10,
            decision_seq=0, bundle_generation="generation1", arm="full", attempt=1))
        core.infer(obs)
        core.acknowledge_execution(dict(decision_seq=0, executed_count=2, terminal=True))
        assert len(factory.bridges) == 1
    finally:
        factory.close()
    reader = CaptureReader(next(tmp_path.iterdir()))
    assert reader.manifest["binding"]["sampler"]["num_steps"] == 2
    row = next(reader.decisions())[0]
    assert row["task_uid"] == ep["task_uid"] and row["task_data_hash"] == ep["task_data_hash"]
    assert row["executed_count"] == 2


def test_worker_capture_uses_metadata_contract_and_task_sampler(tmp_path):
    from exp.r17_integration.debug.capture_bridge import PhysicalCaptureFactory
    provider = fake_provider("p6.worker.review."+uuid.uuid4().hex)
    factory = PhysicalCaptureFactory(tmp_path, {"r17_model_spec":asdict(provider.model_spec)},
                                     sampler_resolver=provider.sampler_for_steps)
    try:
        for k in (2,4):
            event = dict(record(), event="control", control_idx=0, num_steps=k,
                         benchmark_protocol="fake_mw.v1", physical={"arrays":{"qpos":np.arange(3,dtype=np.float64)}})
            factory(event)
        assert len(factory.bridges) == 2
        bad = dict(event); bad.pop("num_steps")
        with pytest.raises(ValueError, match="sampler"):
            factory(bad)
    finally:
        factory.close()
    assert {CaptureReader(p).manifest["binding"]["sampler"]["num_steps"] for p in tmp_path.iterdir()} == {2,4}
