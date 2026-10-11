"""P5's capture hooks roundtrip through P6's real shared writer/library artifact."""
import numpy as np

from exp.r17_integration.models.groot.serve import make_connection_factory
from exp.r17_integration.common.config import build_shared_storage
from .test_shared_factory import config, published


def test_p5_capture_writer_library_artifact_reload_exact(published, manifest, obs, tmp_path):
    from exp.r17_integration.debug.capture_bridge import CaptureBridge, DecisionSink, export_h5
    from exp.r17_integration.debug.reader_profiles import CaptureReader, read_h5
    from exp.r17_integration.debug.validate import model_binding
    from exp.r17_integration.search.build_library import build_library, load_stock_artifact, save_stock_artifact
    from exp.r17_integration.search.store import Store
    binding = model_binding(published.model_spec, manifest.benchmark_protocol)
    bridge = CaptureBridge(tmp_path / "capture", enabled=True, binding=binding)
    metadata = {"task_uid": "P5:eval:0:10", "task_data_hash": "cpu-fixture-sha", "split": "library"}
    sink = DecisionSink(bridge, metadata)
    cfg = config(published, commit=5)
    policy = make_connection_factory(published, cfg, build_shared_storage(cfg), capture_sink=sink)()
    policy.on_task_begin()
    policy.on_episode_start(task_id=0, init_state_id=10, episode_id=0,
        extra_metadata={"task_name": "reach-v3", "original_init_idx": 10, "campaign": "P5-cpu",
                        "arm": "full", "attempt": 1})
    output = policy.infer(obs)
    policy.acknowledge_execution({"decision_seq": 0, "executed_count": 5, "cursor": 5, "terminal": True})
    policy.on_episode_end(success=True)
    policy.on_task_end()
    assert bridge.close()["drained"]
    reader = CaptureReader(tmp_path / "capture", expected_binding=binding)
    rows = list(reader.decisions())
    assert len(rows) == 1
    record, arrays, keys = rows[0]
    assert arrays["normalized_actions"].shape == (16, 32)
    assert arrays["state"].shape == (4,)
    assert set(keys) == {"vision_0", "robot_state"}
    assert len(record["native_snapshots"]) == 3
    assert "native%2Fvision_0_tokens" in arrays
    assert arrays["native%2Fvision_0_tokens"].shape == (256, 8)
    h5 = export_h5(tmp_path / "capture", tmp_path / "capture.h5")
    _, h5rows = read_h5(h5, expected_binding=binding)
    np.testing.assert_array_equal(h5rows[0][1]["normalized_actions"], arrays["normalized_actions"])
    episode = dict(record, closed_ok=True, success=True, n_decisions=1, split="library")
    build_library([tmp_path / "capture"], tmp_path / "store", binding, [episode], key_fields=["vision_0", "robot_state"])
    store = Store(tmp_path / "store", expected_binding=binding)
    assert store.array("active_actions").shape == (1, 16, 4)
    artifact_path = tmp_path / "stock.pkl"
    save_stock_artifact(store, artifact_path)
    backend = load_stock_artifact(artifact_path, binding)
    storage = build_shared_storage(cfg)
    storage._backend = backend
    hit = make_connection_factory(published, cfg, storage)()
    repeated = hit.infer(obs)
    assert repeated["__hit_meta__"]["stage_calls"] == (1, 0, 0)
    np.testing.assert_array_equal(repeated["actions"], output["actions"])
    # The ordinary YAML preload path uses the stock loader, not P6's convenience loader.
    cfg.backend.in_memory.preload_path = str(artifact_path)
    preloaded = build_shared_storage(cfg)
    from_yaml = make_connection_factory(published, cfg, preloaded)().infer(obs)
    assert from_yaml["__hit_meta__"]["stage_calls"] == (1, 0, 0)
    np.testing.assert_array_equal(from_yaml["actions"], output["actions"])
