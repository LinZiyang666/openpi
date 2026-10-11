import json
from pathlib import Path

import numpy as np
import pytest

from exp.r17_integration.debug.capture_bridge import CaptureBridge, capture_native_product
from exp.r17_integration.debug.reader_profiles import CaptureReader
from exp.r17_integration.debug.validate import model_binding
from exp.r17_integration.tests.debug.fixtures import record


@pytest.mark.parametrize("benchmark", ["metaworld", "libero"])
def test_p4_actual_cpu_capture_fixture_lossless(tmp_path, benchmark):
    root = Path(__file__).resolve().parents[2] / "models/dp/samples" / ("real_" + benchmark + "_cpu/capture")
    # This consumes P4's already-produced CPU tensors; no model/weights loaded.
    metadata = json.loads((root / "capture.json").read_text())
    with np.load(root / "capture.npz", allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    protocol = metadata["request"]["benchmark_protocol"]
    binding = model_binding(metadata["model_spec"], protocol, sampler=metadata["sampler"])
    bridge = CaptureBridge(tmp_path / "p4", enabled=True, binding=binding)
    capture_native_product(bridge, {"metadata": metadata, "arrays": arrays},
                           dict(record(), split="diagnostic", source="cpu_native_probe", benchmark_protocol=protocol))
    bridge.close()
    row, restored, keys = next(CaptureReader(tmp_path / "p4").decisions())
    from urllib.parse import quote
    for key, source in arrays.items():
        if key.startswith("keys/"):
            loaded = keys[key[5:]]
        elif key.startswith("key_"):
            loaded = keys[key[4:]]
        else:
            loaded = restored[quote(key, safe="")]
        assert source.dtype == loaded.dtype and source.tobytes() == loaded.tobytes()
    assert row["split"] == "diagnostic" and row["actual_nfe"] == metadata["actual_nfe"]
    assert restored["native_grid"].dtype == np.int64
    assert binding["sampler"]["num_steps"] == 2 and binding["model"]["default_sampler"]["num_steps"] == 100
