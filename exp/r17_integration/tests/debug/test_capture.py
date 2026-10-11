import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from exp.r17_integration.debug.capture_bridge import (
    CaptureBridge, DecisionSink, PhysicalSink, SharedPhysicalTransport, export_h5)
from exp.r17_integration.debug.reader_profiles import CaptureReader, accepted_join, read_h5
from exp.r17_integration.debug.validate import decision_identity, file_sha
from exp.r17_integration.tests.debug.fixtures import arrays, binding, captured, record


@pytest.mark.parametrize("model", ["pi05", "groot", "fastwam", "dp"])
@pytest.mark.parametrize("benchmark", ["mw", "libero"])
def test_lossless_npz_h5_roundtrip(tmp_path, model, benchmark):
    b = binding(model, benchmark)
    path, _ = captured(tmp_path, b, episodes=1)
    original = list(CaptureReader(path, expected_binding=b).decisions())
    export_h5(path, tmp_path / "episode.h5")
    manifest, restored = read_h5(tmp_path / "episode.h5", expected_binding=b)
    assert manifest["binding"] == b
    assert len(original) == len(restored) == 2
    for (r, a, k), (rr, aa, kk) in zip(original, restored):
        assert r == rr
        for values, other in ((a, aa), (k, kk)):
            assert values.keys() == other.keys()
            for name in values:
                assert values[name].dtype == other[name].dtype
                assert values[name].tobytes() == other[name].tobytes()
    assert len(original[0][2]) == len(b["model"]["feature_spec"]["fields"])


@pytest.mark.parametrize("field", ["checkpoint_identity", "norm_identity", "layout", "feature_spec", "default_sampler"])
def test_capture_identity_refuses_mismatch(tmp_path, field):
    b = binding()
    path, _ = captured(tmp_path, b, episodes=1)
    wrong = copy.deepcopy(b)
    wrong["model"][field] = "changed"
    with pytest.raises(ValueError, match="binding mismatch"):
        CaptureReader(path, expected_binding=wrong)


def test_default_off_is_noop_without_writer_import_or_copy(tmp_path):
    class Poison:
        def __array__(self, *args, **kwargs):
            raise AssertionError("off hook copied an array")
    bridge = CaptureBridge(tmp_path / "never_created")
    assert bridge.capture({}, {"value": Poison()}) is None
    assert bridge.capture_physical({}, {"value": Poison()}) is None
    assert bridge.close() is None
    assert not (tmp_path / "never_created").exists()


def test_observe_only_debug_parity_rng_forward_and_copy(tmp_path):
    def run(enabled):
        b = binding()
        bridge = CaptureBridge(tmp_path / str(enabled), enabled=enabled, binding=b)
        rng = np.random.default_rng(123)
        actions, calls = [], 0
        for seq, source in enumerate(("policy", "tail", "tail", "policy")):
            if source == "policy":
                calls += 1
                chunk = rng.normal(size=(16, 4)).astype(np.float32)
            row = record(seq=seq)
            row.update(source=source, stage_calls=[1, 1, 5] if source == "policy" else [0, 0, 0],
                       actual_nfe=5 if source == "policy" else 0)
            bridge.capture(row, {"normalized_actions": chunk})
            actions.append(chunk.tobytes())
        after = rng.bit_generator.state
        bridge.close()
        return actions, calls, after
    assert run(False) == run(True)
    assert [row[0]["actual_nfe"] for row in CaptureReader(tmp_path / "True").decisions()] == [5, 0, 0, 5]


def test_capture_detaches_mutable_arrays_and_records_missing_status(tmp_path):
    b = binding()
    a, keys = arrays(b)
    expected = a["normalized_actions"].copy()
    bridge = CaptureBridge(tmp_path / "capture", enabled=True, binding=b, queue_bytes=128)
    bridge.capture(dict(record(), capabilities={"task_predicates": {"status": "unsupported", "reason": "no map"}}), a, rawkeys=keys)
    a["normalized_actions"][:] = 999
    bridge.close()
    row, values, _ = next(CaptureReader(tmp_path / "capture").decisions())
    np.testing.assert_array_equal(values["normalized_actions"], expected)
    assert row["capabilities"]["task_predicates"]["status"] == "unsupported"
    assert bridge.writer.stats["oversized_items"] == 1


def test_partial_torn_jsonl_and_checksum_tamper(tmp_path):
    b = binding()
    path, _ = captured(tmp_path, b, episodes=1)
    manifest_path = path / "r17_capture.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "partial"
    manifest["files"] = {}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="partial"):
        CaptureReader(path)
    log = next(path.glob("decisions_*.jsonl"))
    with log.open("ab") as f:
        f.write(b'{"unterminated":')
    reader = CaptureReader(path, allow_partial=True)
    assert len(list(reader.decisions())) == 2
    assert len(reader.issues) == 1
    manifest["status"] = "closed"
    manifest["files"] = {str(log.relative_to(path)): "wrong-sha"}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        CaptureReader(path)


def test_writer_errors_cannot_be_promoted_closed(tmp_path):
    class FailedWriter:
        stats = {"drained": True, "errors": ["disk full"]}
        def close(self):
            pass
    bridge = CaptureBridge(tmp_path, enabled=True, binding=binding(), writer=FailedWriter())
    bridge.close()
    assert json.loads((tmp_path / "r17_capture.json").read_text())["status"] == "partial"


def test_object_arrays_rejected_no_pickle(tmp_path):
    bridge = CaptureBridge(tmp_path, enabled=True, binding=binding())
    try:
        with pytest.raises(ValueError, match="unsafe"):
            bridge.capture(record(), {"bad": np.array([{}], dtype=object)})
    finally:
        bridge.close()


def test_accepted_join_excludes_late_rejected_error(tmp_path):
    journal = [dict(record(attempt=1), accepted=False, status="error"),
               dict(record(attempt=2), accepted=True, status="ok", success=False)]
    decisions = [record(attempt=1), record(attempt=2)]
    rows = accepted_join(decisions, journal)
    assert len(rows) == 1 and rows[0]["attempt"] == 2 and rows[0]["success"] is False
    with pytest.raises(ValueError, match="duplicate"):
        accepted_join(decisions + [record(attempt=2)], journal)


def test_stock_journal_done_failure_and_error_join():
    from exp.r17_integration.debug.reader_profiles import enrich_journal, accepted_episodes
    manifest = record()
    raw = dict(task_uid=manifest["task_uid"], attempt=1, accepted=True, status="failed", error=None, success=False)
    enriched = enrich_journal([raw], [manifest])
    assert len(accepted_episodes(enriched)) == 1
    assert len(accepted_episodes([dict(enriched[0], error="renderer error")])) == 0
    with pytest.raises(ValueError, match="exact attempt"):
        enrich_journal([dict(raw, attempt=2)], [manifest])


def test_decision_and_native_physical_seams(tmp_path):
    from exp.r17_integration.common.contracts import DecisionCapture, RequestContext
    b = binding()
    a, keys = arrays(b)
    bridge = CaptureBridge(tmp_path / "debug", enabled=True, binding=b)
    request = RequestContext("cpu-fixture", b["benchmark_protocol"], "reach-v3", 10, 0,
                             arm="full", attempt=1, bundle_generation="generation1")
    capture = DecisionCapture(request, "policy", 0, 0, (1, 1, 5), 5, a["normalized_actions"], a["wire_actions"],
                              keys, {"native_timesteps": (99, 98, 97, 96, 95)})
    DecisionSink(bridge, record())(capture)
    event = dict(record(), event="control", control_idx=0, orig_init_state_idx=10,
                 physical=dict(arrays={"qpos": np.arange(9, dtype=np.float64)},
                               metadata={"success": False}, capabilities={"task_predicates": {"status": "unsupported"}}))
    PhysicalSink(bridge, record())(event)
    bridge.close()
    reader = CaptureReader(tmp_path / "debug")
    row, physical = next(reader.physical())
    assert physical["qpos"].dtype == np.float64
    assert row["capabilities"]["task_predicates"]["status"] == "unsupported"
    assert next(reader.decisions())[1]["native_grid"].dtype.kind == "i"


def test_shared_transport_preserves_existing_allowlist(tmp_path):
    from exp.offline_search.debug.transport.protocol import filename
    events = []
    sink = SimpleNamespace(write_file=lambda name, data: events.append((filename(name), data)),
                           write=lambda text: events.append(("events.jsonl", text)), close=lambda: None)
    block = tmp_path / "block.npz"
    block.write_bytes(b"numeric-fixture")
    transport = SharedPhysicalTransport(sink)
    transport.publish(dict(record(), control_idx=2), block)
    assert events[0][0] == "controls_0000.npz"
    assert json.loads(events[1][1])["control_idx"] == 2


def test_p2_actual_native_physical_runner_sink(tmp_path):
    from exp.r17_integration.benchmarks.metaworld import R17MetaWorldEpisodeRunner, FORMAL_SPEC
    from exp.r17_integration.tests.benchmarks.test_metaworld import Env, Client, task
    b = binding()
    b["benchmark_protocol"] = FORMAL_SPEC.identity
    bridge = CaptureBridge(tmp_path / "p2", enabled=True, binding=b)
    env, client = Env(success=7), Client()
    env._r17_task_data_hash = "cpu-native-taskdata"
    acks = []
    runner = R17MetaWorldEpisodeRunner(env_factory=lambda *a: env, client_factory=lambda _: client,
                                       capture_sink=PhysicalSink(bridge, {}), execution_ack=acks.append)
    try:
        result = runner.run(task(), lambda *a: None)
    finally:
        bridge.close()
    assert result.success and [r["executed_count"] for r in acks] == [5, 2]
    physical = list(CaptureReader(tmp_path / "p2").physical())
    controls = [(r, a) for r, a in physical if r["event"] == "control"]
    assert len(controls) == 7
    assert controls[-1][1]["qpos"].dtype == np.float64
    assert controls[-1][0]["task_data_hash"] == "cpu-native-taskdata"
    assert controls[0][1]["requested_action"][0] == 2 and controls[0][1]["applied_action"][0] == 1
    assert {r["event"] for r, a in physical} >= {"reset", "settled", "terminal", "control"}
