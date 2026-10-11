"""Q regression: observer storage retains BF16 values, bits and native state."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from exp.r17_integration.common.contracts import DecisionCapture, NativeSnapshot, RequestContext
from exp.r17_integration.debug.capture_bridge import CaptureBridge, DecisionSink, export_h5
from exp.r17_integration.debug.reader_profiles import CaptureReader, read_h5
from exp.r17_integration.debug.validate import numeric_arrays
from exp.r17_integration.tests.debug.fixtures import binding, record


BITS_SUFFIX = "__r17_bfloat16_bits"


def test_numeric_arrays_roundtrip_every_bfloat16_pattern_noncontiguous():
    bits = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    native = torch.from_numpy(bits.copy()).view(torch.bfloat16).T
    assert not native.is_contiguous()
    metadata = {}
    stored = numeric_arrays({"snapshot_x": native}, serialization=metadata)
    raw = stored["snapshot_x" + BITS_SUFFIX]
    np.testing.assert_array_equal(raw, bits.T)
    restored = torch.from_numpy(raw.copy()).view(torch.bfloat16)
    assert torch.equal(restored.view(torch.uint16), native.contiguous().view(torch.uint16))
    assert stored["snapshot_x"].tobytes() == native.float().numpy().copy(order="C").tobytes()
    assert metadata["snapshot_x"] == dict(native_dtype="bfloat16", storage_dtype="float32",
        bits_dtype="uint16", bits_array="snapshot_x" + BITS_SUFFIX, original_array="snapshot_x",
        encoding="torch_uint16_view")


def test_bfloat16_companion_name_collision_rejected():
    with pytest.raises(ValueError, match="collision"):
        numeric_arrays({"x": torch.ones(2, dtype=torch.bfloat16),
                        "x" + BITS_SUFFIX: np.ones(2, dtype=np.uint16)})


def capture_for(normalized, seq=0):
    b = binding()
    req = RequestContext("cpu-fixture", b["benchmark_protocol"], "reach-v3", 10, seq,
                         bundle_generation="generation1", arm="full", attempt=1)
    return DecisionCapture(req, "policy", 0, 0, (1, 1, 5), 5,
                           normalized, normalized[:, :4].detach().float().numpy(),
                           {"robot_state": normalized[0, :4]})


def test_decision_sink_bfloat16_npz_h5_and_nested_rng_preserve_native_state(tmp_path):
    native = torch.linspace(-3, 3, 16 * 4, dtype=torch.bfloat16).reshape(16, 4).requires_grad_()
    raw_before = native.detach().contiguous().view(torch.uint16).clone()
    rng_before = torch.get_rng_state().clone()
    snapshot = NativeSnapshot("weights-sha", "norm-sha", "layout-sha", "sampler-sha", "grid-sha",
        1, 98, native, noise_state={"counter": 7, "posterior": native[0],
                                  "rng": {"state": torch.arange(9, dtype=torch.uint8)}})
    capture = replace(capture_for(native), native_snapshots=(snapshot,),
                      model_capture={"initial_noise": native, "tokens": {"base": native[:2]}})
    bridge = CaptureBridge(tmp_path / "capture", enabled=True, binding=binding())
    try:
        DecisionSink(bridge, record())(capture)
    finally:
        bridge.close()
    assert capture.normalized_chunk is native and snapshot.x_t is native
    assert native.requires_grad and native.grad is None
    assert torch.equal(native.detach().contiguous().view(torch.uint16), raw_before)
    assert torch.equal(torch.get_rng_state(), rng_before)
    row, arrays, keys = next(CaptureReader(tmp_path / "capture").decisions())
    assert row["native_snapshots"][0]["tensor_dtype"] == "bfloat16"
    for namespace, payload in (("arrays", arrays), ("rawkeys", keys)):
        for name, receipt in row["array_serialization"][namespace].items():
            assert payload[name].dtype == np.float32
            assert payload[receipt["bits_array"]].dtype == np.uint16
    np.testing.assert_array_equal(arrays["snapshot_1_x" + BITS_SUFFIX], raw_before.numpy())
    noise = row["native_snapshots"][0]["noise_state"]
    assert noise["counter"] == 7
    np.testing.assert_array_equal(arrays[noise["rng"]["state"]["r17_array"]], np.arange(9, dtype=np.uint8))
    posterior_name = noise["posterior"]["r17_array"]
    np.testing.assert_array_equal(arrays[posterior_name + BITS_SUFFIX], raw_before[0].numpy())
    with torch.no_grad():
        native.fill_(100)  # Serialized buffers must not alias live inference.
    np.testing.assert_array_equal(arrays["snapshot_1_x" + BITS_SUFFIX], raw_before.numpy())
    export_h5(tmp_path / "capture", tmp_path / "capture.h5")
    _, h5_rows = read_h5(tmp_path / "capture.h5")
    rr, aa, kk = h5_rows[0]
    assert rr == row
    for payload, restored in ((arrays, aa), (keys, kk)):
        assert payload.keys() == restored.keys()
        assert all(payload[k].dtype == restored[k].dtype and payload[k].tobytes() == restored[k].tobytes()
                   for k in payload)


def test_bfloat16_observer_on_off_inference_bytes_and_rng_match(tmp_path):
    def run(enabled):
        generator = torch.Generator(device="cpu").manual_seed(183)
        bridge = CaptureBridge(tmp_path / str(enabled), enabled=enabled, binding=binding())
        sink = DecisionSink(bridge, record())
        outputs = []
        try:
            for seq in range(2):
                actions = torch.randn((16, 4), generator=generator, dtype=torch.bfloat16)
                sink(capture_for(actions, seq))
                outputs.append(actions.view(torch.uint16).numpy().tobytes())
        finally:
            bridge.close()
        return outputs, generator.get_state().numpy().tobytes()
    assert run(False) == run(True)
    assert not (tmp_path / "False").exists()


def test_float32_numeric_arrays_preserve_existing_names_bytes_and_dtype():
    original = np.arange(12, dtype=np.float32).reshape(3, 4)
    stored = numeric_arrays({"x": original})
    assert set(stored) == {"x"}
    assert stored["x"].dtype == original.dtype and stored["x"].tobytes() == original.tobytes()
    assert not np.shares_memory(stored["x"], original)
